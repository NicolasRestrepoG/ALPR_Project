"""The corpus: an immutable spec, a content hash, a directory of 192 x 96 crops.

    spec (CorpusSpec)  +  inputs (glyphs, backgrounds, municipalities, CCPD index,
                           generator version)  →  sha256  →  data/corpus/<hash>/

The same spec and inputs always produce the same hash and, sample by sample, the
same pixels: sample `i` draws from `default_rng([seed, i])`, so the result does not
depend on how many workers ran or in what order. A finished corpus is reused, not
rebuilt. Every checkpoint trained later records the hash of the corpus it saw — a
result that cannot name its corpus is not a result.

    data/corpus/<hash>/
    ├── images/<index>_<class>_<text|none>.png   already rectified, jitter baked in
    ├── labels.jsonl     index, file, class, text, valid, severity
    ├── spec.json        the spec and the input digests the hash was taken over
    ├── stats.json       class counts, glyph counts and spread, timing
    └── DONE

Sampling policy (plan, Sprint 3.2): letters and digits uniform — balanced per
position here, so counts differ by at most one block; plate width log-uniform
70-260 px; class mix 88% particular, 5% público, 2% moto, 3% CCPD, 2% non-plate
(the plan's 7% público is split with the single-line motorbike plates found in
the labelled photos). Municipalities are uniform: the DIVIPOLA list carries no
population, so population weighting waits for a population source.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from alpr.data_sources.storage import DataPaths, atomic_write_text
from alpr.plates.labels import LabelError
from alpr.plates.rectify import RectifyError
from alpr.synth.params import DEFAULT_RANGES, DIGITS, LETTERS, Ranges
from alpr.synth.pipeline import ORDERS

#: Bump whenever a change to the generator changes its output for the same spec.
GENERATOR_VERSION = "3.2.1"

CLASSES = ("particular", "publico", "moto", "ccpd", "nonplate")
RENDERED = ("particular", "publico", "moto")
CCPD_POOL = 5000  # CCPD plates kept available to draw negatives from


def corpus_ranges() -> Ranges:
    return replace(DEFAULT_RANGES, plate_px_follows_severity=False)


@dataclass(frozen=True)
class CorpusSpec:
    size: int = 20_000
    seed: int = 0
    class_mix: tuple[tuple[str, float], ...] = (
        ("particular", 0.88),
        ("publico", 0.05),
        ("moto", 0.02),
        ("ccpd", 0.03),
        ("nonplate", 0.02),
    )
    severity: tuple[float, float] = (0.0, 1.0)
    order: str = "correct"
    ranges: Ranges = field(default_factory=corpus_ranges)
    municipality_weighting: str = "uniform"

    def __post_init__(self) -> None:
        names = [c for c, _ in self.class_mix]
        if set(names) - set(CLASSES):
            raise ValueError(f"unknown classes {set(names) - set(CLASSES)}")
        if abs(sum(p for _, p in self.class_mix) - 1.0) > 1e-9:
            raise ValueError("class_mix must sum to 1")
        if self.order not in ORDERS:
            raise ValueError(f"unknown order {self.order!r}")
        if self.size <= 0:
            raise ValueError("size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"generator_version": GENERATOR_VERSION}


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def _digest_files(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes() if f.exists() else b"<missing>")
    return h.hexdigest()


def input_digests(paths: DataPaths) -> dict[str, str]:
    glyph_dir = paths.assets_dir / "glyphs"
    return {
        "glyphs": _digest_files(sorted(glyph_dir.glob("*.png")) + [glyph_dir / "glyphs.json"]),
        "backgrounds": _digest_files([paths.root / "backgrounds" / "index.jsonl"]),
        "municipalities": _digest_files([paths.assets_dir / "municipios.txt"]),
        "ccpd_index": _digest_files([paths.index("ccpd")]),
    }


def corpus_hash(spec: CorpusSpec, paths: DataPaths) -> str:
    payload = {"spec": spec.to_dict(), "inputs": input_digests(paths)}
    blob = json.dumps(payload, sort_keys=True, default=list).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def corpus_dir(paths: DataPaths, spec: CorpusSpec) -> Path:
    return paths.corpus_dir / corpus_hash(spec, paths)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def _perm(seed: int, position: int, block: int, n: int) -> np.ndarray:
    return np.random.default_rng([seed, 7919 + position, block]).permutation(n)


def balanced_text(seed: int, index: int, moto: bool = False) -> str:
    """Letters and digits uniform *and* balanced: within each block of 26 (letters)
    or 10 (digits) consecutive indices, every symbol appears once per position."""
    letters = [LETTERS[_perm(seed, k, index // 26, 26)[index % 26]] for k in range(3)]
    digits = [DIGITS[_perm(seed, 3 + k, index // 10, 10)[index % 10]] for k in range(3)]
    if moto:
        digits[2] = LETTERS[_perm(seed, 6, index // 26, 26)[index % 26]]
    return "".join(letters + digits)


def choose_class(spec: CorpusSpec, u: float) -> str:
    acc = 0.0
    for name, p in spec.class_mix:
        acc += p
        if u < acc:
            return name
    return spec.class_mix[-1][0]


# Worker state, built once per process.
_STATE: dict[str, Any] = {}


def _init(root: str, spec: CorpusSpec, ccpd_pool: list[tuple[str, list]], out: str) -> None:
    from alpr.synth.backgrounds import BackgroundStore
    from alpr.synth.generate import default_synthesizer

    # One OpenCV thread per worker: the parallelism is across processes, and nested
    # thread pools in every process oversubscribe the cores (2x slower measured).
    cv2.setNumThreads(1)
    paths = DataPaths(root)
    syn = default_synthesizer(paths, quiet=True)
    syn.ranges = spec.ranges
    try:
        store = BackgroundStore(paths)
    except LabelError:
        store = None
    _STATE.update(paths=paths, spec=spec, syn=syn, store=store, ccpd=ccpd_pool, out=Path(out))


def make_sample(index: int) -> dict[str, Any]:
    from alpr.synth.negatives import ccpd_crop, nonplate_crop

    s = _STATE
    spec: CorpusSpec = s["spec"]
    rng = np.random.default_rng([spec.seed, index])
    cls = choose_class(spec, float(rng.random()))
    # Negatives need their source; without it, fall back to a público plate.
    if (cls == "ccpd" and not s["ccpd"]) or (cls == "nonplate" and s["store"] is None):
        cls = "publico"
    severity = float(rng.uniform(*spec.severity))
    text = None
    if cls in RENDERED:
        text = balanced_text(spec.seed, index, moto=cls == "moto")
        crop = s["syn"].sample(rng, severity, text=text, style=cls, order=spec.order).crop
    else:
        # Real-image negatives: a source record whose corners are not a plate (one in
        # ~3,000 CCPD labels) is redrawn, deterministically, from the same stream.
        for attempt in range(8):
            try:
                if cls == "ccpd":
                    image_path, corners = s["ccpd"][int(rng.integers(len(s["ccpd"])))]
                    crop = ccpd_crop(s["paths"], image_path, corners, rng)
                else:
                    crop = nonplate_crop(s["store"], int(rng.integers(len(s["store"]))), rng)
                break
            except RectifyError:
                if attempt == 7:
                    raise
    name = f"{index:06d}_{cls}_{text or 'none'}.png"
    cv2.imwrite(
        str(s["out"] / "images" / name),
        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    )
    return {
        "index": index,
        "file": f"images/{name}",
        "class": cls,
        "text": text,
        "valid": cls == "particular",
        "severity": round(severity, 4),
    }


def _ccpd_pool(paths: DataPaths, seed: int) -> list[tuple[str, list]]:
    index = paths.index("ccpd")
    if not index.exists():
        return []
    rows = []
    with index.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("corners"):
                rows.append((r["image_path"], r["corners"]))
    rng = np.random.default_rng([seed, 31337])
    pick = rng.choice(len(rows), min(CCPD_POOL, len(rows)), replace=False) if rows else []
    return [rows[k] for k in sorted(pick)]


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


@dataclass
class CorpusReport:
    path: Path
    hash: str
    size: int
    seconds: float
    workers: int
    class_counts: dict[str, int]
    letter_spread: float
    digit_spread: float
    reused: bool = False

    @property
    def samples_per_s(self) -> float:
        return self.size / self.seconds if self.seconds else 0.0


def glyph_stats(labels: list[dict[str, Any]]) -> tuple[dict[str, int], float, float]:
    """Glyph counts over the particular plates, and the spread (max - min) / mean
    within letters and within digits — the two groups are uniform separately."""
    counts: Counter[str] = Counter()
    for lb in labels:
        if lb["class"] == "particular" and lb["text"]:
            counts.update(lb["text"])

    def spread(alphabet: str) -> float:
        vals = [counts[c] for c in alphabet]
        mean = sum(vals) / len(vals)
        return (max(vals) - min(vals)) / mean if mean else 0.0

    return dict(counts), spread(LETTERS), spread(DIGITS)


def build_corpus(
    paths: DataPaths, spec: CorpusSpec, *, workers: int | None = None, force: bool = False
) -> CorpusReport:
    h = corpus_hash(spec, paths)
    out = paths.corpus_dir / h
    if (out / "DONE").exists() and not force:
        stats = json.loads((out / "stats.json").read_text())
        return CorpusReport(**(stats["report"] | {"path": out, "reused": True}))
    partial = out.with_name(h + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    (partial / "images").mkdir(parents=True)
    atomic_write_text(
        partial / "spec.json",
        json.dumps(
            {"hash": h, "spec": spec.to_dict(), "inputs": input_digests(paths)},
            indent=2,
            default=list,
        )
        + "\n",
    )
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    pool_args = (str(paths.root), spec, _ccpd_pool(paths, spec.seed), str(partial))
    t0 = time.perf_counter()
    if workers == 1:
        _init(*pool_args)
        labels = [make_sample(i) for i in range(spec.size)]
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=pool_args) as pool:
            labels = list(pool.map(make_sample, range(spec.size), chunksize=32))
    seconds = time.perf_counter() - t0

    counts, l_spread, d_spread = glyph_stats(labels)
    report = CorpusReport(
        path=out,
        hash=h,
        size=spec.size,
        seconds=round(seconds, 2),
        workers=workers,
        class_counts=dict(Counter(lb["class"] for lb in labels)),
        letter_spread=round(l_spread, 4),
        digit_spread=round(d_spread, 4),
    )
    atomic_write_text(partial / "labels.jsonl", "".join(json.dumps(lb) + "\n" for lb in labels))
    report_dict = asdict(report) | {"path": str(out)}
    report_dict.pop("reused")
    atomic_write_text(
        partial / "stats.json",
        json.dumps({"report": report_dict, "glyph_counts": counts}, indent=2) + "\n",
    )
    (partial / "DONE").write_text(f"{h}\n")
    if out.exists():
        shutil.rmtree(out)
    partial.rename(out)
    return report


def load_labels(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (Path(path) / "labels.jsonl").read_text().splitlines()]
