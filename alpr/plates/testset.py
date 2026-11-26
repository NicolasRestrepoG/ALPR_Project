"""Freeze a provisional real test set from labelled public images.

Until the own photos exist (before Phase 5), the V0 recognizer is scored on plates
labelled from the Roboflow sources. A score is only worth something if the set it
comes from cannot drift, so freezing does four things:

* **Glyph sources stay out.** Every photo of a car tagged `glyph_source` anywhere
  is excluded: its characters become the renderer's glyphs, and testing on them
  would measure memory, not reading.
* **Cars, not photos.** The Roboflow sets photograph the same car many times. Each
  entry carries a `group` (its plate text) so results can be reported per car, and
  near-identical photos of the same car are dropped by perceptual hash.
* **Provenance per entry.** Source, original path, licence and attribution URL.
* **Read-only and checksummed.** Images are copied byte for byte, every file is
  made read-only, and `SHA256SUMS` covers them all. `annotate verify` rechecks it.

Layout:

    data/provisional_test/<name>/
    ├── images/<sha256[:16]>.<ext>   original bytes; a label's rotN tag still applies
    ├── manifest.jsonl               one entry per labelled plate
    ├── SHA256SUMS
    └── README.txt                   the rules and the counts at freeze time

`unknown` labels (unreadable plates) carry no ground truth and are left out.
These images must also stay out of any later detector fine-tuning split.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from alpr.data_sources.base import PlateType
from alpr.data_sources.storage import DataPaths, sha256_file
from alpr.plates.labels import (
    CHARSET,
    LabelError,
    LabelStore,
    labels_path,
    list_items,
    rotation_of,
)
from alpr.plates.rectify import load_image

TESTSET_DIR = "provisional_test"
MANIFEST_NAME = "manifest.jsonl"
SUMS_NAME = "SHA256SUMS"
README_NAME = "README.txt"

#: dHash bits that may differ for two photos of the same car to count as the same
#: photo. 64-bit hash; 4 catches re-encodes and near-identical frames, not a car
#: photographed again from a step to the side.
NEAR_DUPLICATE_BITS = 4


class FrozenSetError(LabelError):
    """A frozen set that is missing, already exists, or no longer matches its sums."""


@dataclass(frozen=True, slots=True)
class Entry:
    file: str  # relative to the set's directory
    sha256: str
    source: str
    key: str  # the label key within its source
    image_path: str  # relative to the data root, as published
    license_key: str
    attribution_url: str | None
    plate: str | None
    plate_type: str
    corners: list[list[float]] | None  # TL/TR/BR/BL, in the frame after rotation
    tags: list[str]
    group: str  # one per car: "plate:<text>", or "image:<source>/<key>" without text
    dhash: str


@dataclass
class FreezeReport:
    path: Path
    entries: int = 0
    images: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    cars: int = 0  # distinct particular plates
    excluded_glyph_source: int = 0
    excluded_unknown: int = 0
    near_duplicates: int = 0
    char_counts: dict[str, int] = field(default_factory=dict)

    def thin_chars(self, target: int = 5) -> dict[str, int]:
        return {c: n for c, n in self.char_counts.items() if n < target}


def frozen_dir(paths: DataPaths, name: str) -> Path:
    return paths.root / TESTSET_DIR / name


def dhash(image: np.ndarray) -> int:
    """64-bit difference hash of an RGB image: robust to scale and re-encoding."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _read_only(path: Path) -> None:
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def freeze(paths: DataPaths, sources: Sequence[str], *, name: str = "provisional") -> FreezeReport:
    """Build the frozen set from the labels of `sources`. Refuses to overwrite."""
    dest = frozen_dir(paths, name)
    if dest.exists():
        raise FrozenSetError(
            f"{dest} is already frozen. A frozen set is never rewritten; delete it by "
            "hand (chmod -R u+w, then remove) if you really mean to replace it."
        )
    labelled = []
    for source in sources:
        store = LabelStore(labels_path(paths, source))
        if not len(store):
            raise FrozenSetError(f"no labels for {source!r} — label it with `annotate serve`")
        for item in list_items(paths, source, order="name"):
            label = store.get(item.key)
            if label is not None:
                labelled.append((source, item, label))

    glyph_plates = {lb.plate for _, _, lb in labelled if "glyph_source" in lb.tags and lb.plate}
    report = FreezeReport(path=dest)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    (partial / "images").mkdir(parents=True)

    entries: list[Entry] = []
    hashes: dict[str, list[int]] = {}
    copied: set[str] = set()
    try:
        for source, item, label in labelled:
            if label.plate_type is PlateType.UNKNOWN:
                report.excluded_unknown += 1
                continue
            if label.plate and label.plate in glyph_plates:
                report.excluded_glyph_source += 1
                continue
            group = f"plate:{label.plate}" if label.plate else f"image:{source}/{item.key}"
            h = dhash(load_image(item.image, rotation_of(label.tags)))
            if any(_hamming(h, other) <= NEAR_DUPLICATE_BITS for other in hashes.get(group, [])):
                report.near_duplicates += 1
                continue
            hashes.setdefault(group, []).append(h)

            digest = sha256_file(item.image)
            rel = f"images/{digest[:16]}{item.image.suffix.lower()}"
            if rel not in copied:
                shutil.copyfile(item.image, partial / rel)
                _read_only(partial / rel)
                copied.add(rel)
            rec = item.record
            entries.append(
                Entry(
                    file=rel,
                    sha256=digest,
                    source=source,
                    key=item.key,
                    image_path=rec.image_path if rec else paths.relative(item.image),
                    license_key=rec.license_key if rec else "own",
                    attribution_url=rec.attribution_url if rec else None,
                    plate=label.plate,
                    plate_type=label.plate_type.value,
                    corners=[list(p) for p in label.corners] if label.corners else None,
                    tags=list(label.tags),
                    group=group,
                    dhash=f"{h:016x}",
                )
            )

        entries.sort(key=lambda e: (e.group, e.source, e.key))
        manifest = partial / MANIFEST_NAME
        manifest.write_text("".join(json.dumps(asdict(e), sort_keys=True) + "\n" for e in entries))
        _read_only(manifest)

        report.entries = len(entries)
        report.images = len(copied)
        report.by_type = dict(Counter(e.plate_type for e in entries))
        cars = {e.plate for e in entries if e.plate_type == PlateType.PARTICULAR.value and e.plate}
        report.cars = len(cars)
        counts = Counter({c: 0 for c in CHARSET})
        for plate in cars:
            counts.update(ch for ch in plate if ch in counts)
        report.char_counts = dict(counts)

        readme = partial / README_NAME
        readme.write_text(_readme(report, sources))
        _read_only(readme)

        sums = partial / SUMS_NAME
        files = sorted([*copied, MANIFEST_NAME, README_NAME])
        sums.write_text("".join(f"{sha256_file(partial / f)}  {f}\n" for f in files))
        _read_only(sums)
        partial.rename(dest)
    except BaseException:
        _make_writable(partial)
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return report


def _make_writable(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def _readme(report: FreezeReport, sources: Sequence[str]) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    types = ", ".join(f"{k} {v}" for k, v in sorted(report.by_type.items()))
    thin = report.thin_chars()
    return (
        f"Provisional real test set — frozen {when}\n\n"
        f"Sources: {', '.join(sources)}\n"
        f"Entries: {report.entries} ({types}) from {report.images} images\n"
        f"Distinct particular cars: {report.cars}\n"
        f"Excluded: {report.excluded_glyph_source} photos of glyph-source cars, "
        f"{report.excluded_unknown} unreadable (unknown), "
        f"{report.near_duplicates} near-duplicates\n"
        f"Characters on fewer than 5 cars: "
        f"{', '.join(f'{c}:{n}' for c, n in sorted(thin.items())) or 'none'}\n\n"
        "Rules: report results per car (group), not per photo. Never train on these\n"
        "images, and keep them out of detector fine-tuning splits. Corners are in the\n"
        "frame after any rotN tag is applied. Replaced by the own-photo test set before\n"
        "Phase 5.\n"
    )


def load_entries(paths: DataPaths, name: str = "provisional") -> list[Entry]:
    path = frozen_dir(paths, name) / MANIFEST_NAME
    if not path.exists():
        raise FrozenSetError(f"no frozen test set at {path.parent} — run `annotate freeze`")
    return [Entry(**json.loads(line)) for line in path.read_text().splitlines() if line]


def entry_image(paths: DataPaths, entry: Entry, name: str = "provisional") -> np.ndarray:
    """The entry's image in the frame its corners live in (rotation applied)."""
    return load_image(frozen_dir(paths, name) / entry.file, rotation_of(entry.tags))


def verify(paths: DataPaths, name: str = "provisional") -> list[str]:
    """Problems with a frozen set; an empty list means it is intact."""
    root = frozen_dir(paths, name)
    sums = root / SUMS_NAME
    if not sums.exists():
        raise FrozenSetError(f"no frozen test set at {root} — run `annotate freeze`")
    problems = []
    listed = set()
    for line in sums.read_text().splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        path = root / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"changed: {rel}")
        elif path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            problems.append(f"writable: {rel}")
    extra = (
        {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        - listed
        - {SUMS_NAME}
    )
    problems += [f"not in {SUMS_NAME}: {rel}" for rel in sorted(extra)]
    return problems
