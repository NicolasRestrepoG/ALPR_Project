"""The background corpus: vehicle fronts around a painted-out plate.

Each entry is a crop centred on a real plate, three plate-widths wide, with that
plate inpainted away and its quad recorded. The compositor places the crop so the
synthetic plate lands where the real one was: bumper, grille, lighting and scale
stay plausible, and the plate/vehicle boundary is a real one.

Sources are those registered for `Purpose.BACKGROUNDS` that carry plate corners:
CCPD (corners in the index) and the labelled Roboflow sets (corners from our
labels, rotation applied). **Images in the frozen provisional test set are never
used**: their surroundings would otherwise leak into training.

    data/backgrounds/
    ├── images/<n>.jpg      480 x 360
    └── index.jsonl         file, source, image_path, quad (in crop pixels)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from alpr.data_sources import registry
from alpr.data_sources.base import read_jsonl
from alpr.data_sources.registry import Purpose
from alpr.data_sources.storage import DataPaths, atomic_write_text
from alpr.plates.labels import LabelError, LabelStore, labels_path, list_items, rotation_of
from alpr.plates.rectify import load_image

BG_DIR = "backgrounds"
INDEX_NAME = "index.jsonl"
CROP_SIZE = (480, 360)
CROP_PLATE_WIDTHS = 3.0
INPAINT_SCALE = 4  # inpaint at 1/4 size: the hole is covered by the new plate anyway


@dataclass(frozen=True, slots=True)
class BgEntry:
    file: str
    source: str
    image_path: str
    quad: list[list[float]]  # TL/TR/BR/BL in crop pixels


def backgrounds_dir(paths: DataPaths) -> Path:
    return paths.root / BG_DIR


def _frozen_test_paths(paths: DataPaths) -> set[str]:
    from alpr.plates.testset import FrozenSetError, load_entries

    try:
        return {e.image_path for e in load_entries(paths)}
    except FrozenSetError:
        return set()


def candidates(
    paths: DataPaths, sources: list[str], ccpd_n: int, seed: int = 0
) -> list[dict[str, Any]]:
    """(source, image path, quad, rotation) for every usable plate, test set excluded."""
    excluded = _frozen_test_paths(paths)
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for source in sources:
        info = registry.get(source).info
        if Purpose.BACKGROUNDS not in info.purposes:
            raise LabelError(f"{source!r} is not registered for backgrounds")
        if source == "ccpd":
            index = paths.index("ccpd")
            if not index.exists():
                continue
            recs = [r for r in read_jsonl(index) if r.corners and r.image_path not in excluded]
            pick = rng.choice(len(recs), min(ccpd_n, len(recs)), replace=False) if recs else []
            for k in sorted(pick):
                r = recs[k]
                out.append(
                    {
                        "source": source,
                        "image_path": r.image_path,
                        "quad": [list(p) for p in r.corners],
                        "rotation": 0,
                    }
                )
            continue
        store = LabelStore(labels_path(paths, source))
        for item in list_items(paths, source, order="name"):
            lb = store.get(item.key)
            rec = item.record
            if lb is None or lb.corners is None or rec is None or rec.image_path in excluded:
                continue
            out.append(
                {
                    "source": source,
                    "image_path": rec.image_path,
                    "quad": [list(p) for p in lb.corners],
                    "rotation": rotation_of(lb.tags),
                }
            )
    return out


def make_crop(image: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Crop three plate-widths around `quad`, paint the plate out, resize to CROP_SIZE."""
    q = np.asarray(quad, float)
    pw = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    cw = CROP_PLATE_WIDTHS * pw
    ch = cw * CROP_SIZE[1] / CROP_SIZE[0]
    cx, cy = q.mean(axis=0)
    x0, y0 = cx - cw / 2, cy - ch / 2
    scale = CROP_SIZE[0] / cw
    m = np.float32([[scale, 0, -x0 * scale], [0, scale, -y0 * scale]])
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    crop = cv2.warpAffine(image, m, CROP_SIZE, flags=interp, borderMode=cv2.BORDER_REFLECT_101)
    cq = (q - [x0, y0]) * scale

    # Inpaint the plate (slightly dilated quad) at reduced size, paste it back.
    small = cv2.resize(
        crop, None, fx=1 / INPAINT_SCALE, fy=1 / INPAINT_SCALE, interpolation=cv2.INTER_AREA
    )
    mask = np.zeros(small.shape[:2], np.uint8)
    centre = cq.mean(axis=0)
    grown = centre + (cq - centre) * 1.12
    cv2.fillPoly(mask, [np.round(grown / INPAINT_SCALE).astype(np.int32)], 255)
    filled = cv2.inpaint(small, mask, 3, cv2.INPAINT_TELEA)
    big = cv2.resize(filled, CROP_SIZE, interpolation=cv2.INTER_CUBIC)
    full_mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(full_mask, [np.round(grown).astype(np.int32)], 255)
    soft = cv2.GaussianBlur(full_mask, (0, 0), 3).astype(np.float32)[..., None] / 255.0
    crop = (big * soft + crop * (1 - soft)).astype(np.uint8)
    return crop, cq


def _work(args: tuple[str, int, dict[str, Any], str]) -> dict[str, Any] | None:
    root, n, cand, out_dir = args
    try:
        img = load_image(Path(root) / cand["image_path"], cand["rotation"])
    except Exception:  # noqa: BLE001 — a corrupt file must not stop a 20k build
        return None
    crop, quad = make_crop(img, np.asarray(cand["quad"]))
    name = f"{n:06d}.jpg"
    cv2.imwrite(
        str(Path(out_dir) / name),
        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    return asdict(
        BgEntry(f"images/{name}", cand["source"], cand["image_path"], quad.round(2).tolist())
    )


def build(
    paths: DataPaths,
    sources: list[str],
    *,
    ccpd_n: int = 19000,
    seed: int = 0,
    workers: int | None = None,
    force: bool = False,
) -> tuple[int, int]:
    """Build the corpus. Returns (written, failed)."""
    out = backgrounds_dir(paths)
    if (out / INDEX_NAME).exists() and not force:
        raise LabelError(f"{out} already exists — pass --force to rebuild it")
    cands = candidates(paths, sources, ccpd_n, seed)
    if not cands:
        raise LabelError("no usable background sources (need corners: CCPD index or labels)")
    (out / "images").mkdir(parents=True, exist_ok=True)
    jobs = [(str(paths.root), n, c, str(out / "images")) for n, c in enumerate(cands)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        results = [_work(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(_work, jobs, chunksize=64))
    entries = [r for r in results if r is not None]
    atomic_write_text(out / INDEX_NAME, "".join(json.dumps(e) + "\n" for e in entries))
    return len(entries), len(results) - len(entries)


class BackgroundStore:
    def __init__(self, paths: DataPaths) -> None:
        self.root = backgrounds_dir(paths)
        index = self.root / INDEX_NAME
        if not index.exists():
            raise LabelError(f"no background corpus at {self.root} — run `generate backgrounds`")
        self.entries = [
            BgEntry(**json.loads(line)) for line in index.read_text().splitlines() if line
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        e = self.entries[i]
        bgr = cv2.imread(str(self.root / e.file), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.asarray(e.quad, float)
