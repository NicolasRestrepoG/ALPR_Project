"""Extract the 36 plate characters, and the MinTransporte logo, from labelled photos.

The typeface is not published, so the renderer's characters are cut from real
plates. Sources are labelled plates tagged `glyph_source` (sharp, frontal). Until
the own close-ups exist they are low-resolution Roboflow photos, so instead of
trusting one crop per character, every character is rebuilt from all of its
instances:

1. **Rectify** each source plate at the render resolution (4 px/mm, 1320 x 640).
2. **Find the six characters** by their ink blobs and fit the slot positions, so a
   slightly-off corner label does not cut a character in half.
3. **One instance per car.** The Roboflow sets photograph the same car many times;
   the widest (highest-resolution) photo of each car is used, so one car cannot
   outvote the rest.
4. **Normalise** each crop to ink darkness in [0, 1] (background 0, ink 1), and
   keep only the character's own blob: neighbours, bolt heads and dirt at the crop
   edge are masked out.
5. **Align** the instances of a character to each other (sub-pixel translation,
   iterated against their median) and take the **median**: blur, glare and dirt
   that differ between photos cancel; the letterform they share remains.

Output, in `data/assets/glyphs/`:

    <C>.png        8-bit, 255 = ink, cropped to the ink with a 2 mm margin, 4 px/mm
    logo.png       the MinTransporte logo from the gutter, same convention
    glyphs.json    per glyph: source cars, instance count, confidence, ink size in mm
    qa_sheet.png   each glyph beside its source crops at matched scale

**Ink size, not cell size.** In photos the characters' ink measures ~85 mm tall
against 80 mm on the real plate (embossing seen from above). The glyphs are stored
exactly as photographed, with their measured ink size in `glyphs.json`; scaling to
the cell is the renderer's decision, made once for all 36.

Glyphs from one or two cars are marked `low` confidence: check them first when
Phase 5's per-glyph attribution points at a character.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from alpr.data_sources.base import PlateType
from alpr.data_sources.storage import DataPaths, atomic_write_text
from alpr.plates.geometry import MT001, RENDER_PX_PER_MM, PlateGeometry
from alpr.plates.labels import (
    CHARSET,
    LabelError,
    LabelStore,
    labels_path,
    list_items,
    load_labelled_image,
)
from alpr.plates.rectify import rectify

PX = float(RENDER_PX_PER_MM)

#: Crop around each character slot, in mm: wide enough for a mis-fitted slot and
#: for the ~5 mm the photographed ink runs past the cell, before masking.
WIN_W_MM = 60.0
WIN_H_MM = 100.0
WIN_ABOVE_MM = 8.0  # how far above the characters' top line the window starts

LOGO_WIN_MM = 22.0
MARGIN_MM = 2.0
LOW_CONFIDENCE_BELOW = 3  # cars

GLYPHS_DIR = "glyphs"
MANIFEST_NAME = "glyphs.json"


@dataclass(frozen=True, slots=True)
class Instance:
    char: str
    dark: np.ndarray  # float32 [0, 1], WIN_H x WIN_W at PX
    plate: str
    source: str
    key: str
    plate_width_px: float  # source resolution: the labelled plate's width in the photo


@dataclass
class Glyph:
    """A character (or the logo) as photographed.

    `origin_mm` places the image: its top-left corner relative to the slot's centre
    line and the characters' top line (for the logo: the gutter centre and the
    characters' mid-height), in photographed millimetres. A "1" sits off-centre in
    its cell; without this the renderer would centre it.
    """

    char: str
    image: np.ndarray  # uint8, 255 = ink, cropped
    instances: list[Instance] = field(default_factory=list)
    origin_mm: tuple[float, float] = (0.0, 0.0)

    @property
    def cars(self) -> list[str]:
        return sorted({i.plate for i in self.instances})

    @property
    def confidence(self) -> str:
        return "low" if len(self.cars) < LOW_CONFIDENCE_BELOW else "ok"

    def ink_mm(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.image >= 128)
        if not len(xs):
            return (0.0, 0.0)
        return ((xs.max() - xs.min() + 1) / PX, (ys.max() - ys.min() + 1) / PX)


def glyphs_dir(paths: DataPaths) -> Path:
    return paths.assets_dir / GLYPHS_DIR


# --------------------------------------------------------------------------- #
# Finding characters on a rectified plate
# --------------------------------------------------------------------------- #


def find_characters(gray: np.ndarray, px: float = PX) -> list[np.ndarray] | None:
    """The six character blobs on a rectified plate, left to right, as
    connectedComponents stats rows (x, y, w, h, area), or None if six clean
    characters cannot be isolated."""
    _, th = cv2.threshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, _, stats, _ = cv2.connectedComponentsWithStats(th)
    blobs = [
        s
        for s in stats[1:]
        if 55 * px < s[3] < 100 * px and 3 * px < s[2] < 60 * px and 5 * px < s[1] < 75 * px
    ]
    if len(blobs) < 6:
        return None
    hmed = statistics.median(s[3] for s in blobs)
    chars = sorted((s for s in blobs if abs(s[3] - hmed) < 0.08 * hmed), key=lambda s: s[0])
    return chars if len(chars) == 6 else None


def slot_centres(chars: Sequence[np.ndarray]) -> np.ndarray:
    """Fit x = a + b*k (+ gutter for the digits) through the blob centres.

    Each blob's own centre is unreliable for narrow glyphs (a "1" sits off-centre in
    its cell); the fit uses all six, so the slots follow the plate, not the glyph.
    """
    cx = np.array([s[0] + s[2] / 2 for s in chars], float)
    design = np.array([[1.0, k, 1.0 if k > 2 else 0.0] for k in range(6)])
    coef, *_ = np.linalg.lstsq(design, cx, rcond=None)
    return design @ coef


def darkness(gray: np.ndarray) -> np.ndarray:
    """Map a crop to ink darkness: background 0, ink 1, robust to exposure."""
    g = gray.astype(np.float32)
    bg, ink = np.percentile(g, 90), np.percentile(g, 3)
    return np.clip((bg - g) / max(bg - ink, 1.0), 0.0, 1.0)


def keep_main_blob(dark: np.ndarray, centre: tuple[float, float], px: float = PX) -> np.ndarray:
    """Zero everything but the ink blob nearest `centre` (x, y): the character."""
    binary = (dark >= 0.5).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary)
    if n <= 1:
        return dark * 0
    big = [k for k in range(1, n) if stats[k][4] >= 0.02 * dark.size]
    if not big:
        return dark * 0
    k = min(big, key=lambda k: np.hypot(cents[k][0] - centre[0], cents[k][1] - centre[1]))
    mask = (labels == k).astype(np.uint8)
    radius = max(1, round(1.0 * px))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    return dark * mask


def _crop(gray: np.ndarray, x0: float, y0: float, w: float, h: float) -> np.ndarray | None:
    xi, yi, wi, hi = round(x0), round(y0), round(w), round(h)
    if xi < 0 or yi < 0 or xi + wi > gray.shape[1] or yi + hi > gray.shape[0]:
        return None
    return gray[yi : yi + hi, xi : xi + wi]


# --------------------------------------------------------------------------- #
# Collecting and combining
# --------------------------------------------------------------------------- #


def collect(
    paths: DataPaths, sources: Sequence[str], geometry: PlateGeometry = MT001
) -> tuple[dict[str, list[Instance]], list[Instance], int]:
    """Character instances (one photo per car) and logo instances from the
    `glyph_source` plates of `sources`. Also returns how many plates were skipped."""
    best: dict[str, tuple[float, str, Any, Any]] = {}
    for source in sources:
        store = LabelStore(labels_path(paths, source))
        for item in list_items(paths, source, order="name"):
            lb = store.get(item.key)
            if (
                lb is None
                or lb.corners is None
                or "glyph_source" not in lb.tags
                or lb.plate_type is not PlateType.PARTICULAR
                or not lb.plate
            ):
                continue
            (tl, tr, br, bl) = lb.corners
            width = (np.hypot(*np.subtract(tr, tl)) + np.hypot(*np.subtract(br, bl))) / 2
            if lb.plate not in best or width > best[lb.plate][0]:
                best[lb.plate] = (float(width), source, item, lb)
    if not best:
        raise LabelError("no labelled plates tagged glyph_source — tag some with `annotate serve`")

    size = geometry.render_size(PX)
    chars: dict[str, list[Instance]] = {c: [] for c in CHARSET}
    logos: list[Instance] = []
    skipped = 0
    for plate, (width, source, item, lb) in sorted(best.items()):
        flat = rectify(load_labelled_image(item.image, lb), lb.corners, size)
        gray = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
        found = find_characters(gray)
        if found is None or len(plate) != 6:
            skipped += 1
            continue
        top = statistics.median(s[1] for s in found)
        height = statistics.median(s[3] for s in found)
        centres = slot_centres(found)
        x_off, y_off = WIN_W_MM * PX / 2, WIN_ABOVE_MM * PX
        for k, ch in enumerate(plate):
            crop = _crop(gray, centres[k] - x_off, top - y_off, WIN_W_MM * PX, WIN_H_MM * PX)
            if crop is None:
                continue
            dark = keep_main_blob(darkness(crop), (x_off, y_off + height / 2))
            chars[ch].append(Instance(ch, dark, plate, source, item.key, width))
        # The logo sits in the gutter, at mid character height.
        lx, ly = (centres[2] + centres[3]) / 2, top + height / 2
        half = LOGO_WIN_MM * PX / 2
        crop = _crop(gray, lx - half, ly - half, 2 * half, 2 * half)
        if crop is not None:
            dark = keep_main_blob(darkness(crop), (half, half))
            logos.append(Instance("logo", dark, plate, source, item.key, width))
    return chars, logos, skipped


def _align(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(ref, img, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
    except cv2.error:
        return img  # leave unaligned rather than fail a whole extraction
    return cv2.warpAffine(
        img, warp, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    )


def combine(instances: Sequence[Instance], rounds: int = 3) -> np.ndarray:
    """Align the instances to their median, re-take the median, repeat."""
    imgs = [i.dark.astype(np.float32) for i in instances]
    ref = np.median(imgs, axis=0).astype(np.float32)
    if len(imgs) > 1:
        for _ in range(rounds):
            ref = np.median([_align(i, ref) for i in imgs], axis=0).astype(np.float32)
    return ref


def finish(dark: np.ndarray) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Keep the main blob, crop to the ink with MARGIN_MM. Returns the uint8 image
    (255 = ink) and the crop's top-left (x, y) in the window, or None when there is
    no ink blob at all (e.g. a plate without a logo)."""
    h, w = dark.shape
    ys, xs = np.nonzero(dark >= 0.5)
    if not len(xs):
        return None
    dark = keep_main_blob(dark, (float(np.median(xs)), float(np.median(ys))))
    ys, xs = np.nonzero(dark >= 0.5)
    if not len(xs):
        return None
    m = round(MARGIN_MM * PX)
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + 1 + m)
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + 1 + m)
    image = (np.clip(dark[y0:y1, x0:x1], 0, 1) * 255).round().astype(np.uint8)
    return image, (int(x0), int(y0))


def extract(paths: DataPaths, sources: Sequence[str]) -> tuple[dict[str, Glyph], Glyph | None, int]:
    """Build all glyphs present in the sources. Returns (glyphs, logo, skipped)."""
    chars, logos, skipped = collect(paths, sources)
    anchor = (WIN_W_MM * PX / 2, WIN_ABOVE_MM * PX)  # slot centre, characters' top line
    glyphs = {}
    for c, v in chars.items():
        done = finish(combine(v)) if v else None
        if done is not None:
            image, (x0, y0) = done
            origin = ((x0 - anchor[0]) / PX, (y0 - anchor[1]) / PX)
            glyphs[c] = Glyph(c, image, list(v), origin)
    logo = None
    done = finish(combine(logos)) if logos else None
    if done is not None:
        image, (x0, y0) = done
        half = LOGO_WIN_MM * PX / 2  # gutter centre, characters' mid-height
        logo = Glyph("logo", image, logos, ((x0 - half) / PX, (y0 - half) / PX))
    return glyphs, logo, skipped


# --------------------------------------------------------------------------- #
# Writing, QA
# --------------------------------------------------------------------------- #


def _ink_to_png(image: np.ndarray) -> np.ndarray:
    """Display form: black ink on white."""
    return 255 - image


def qa_sheet(glyphs: dict[str, Glyph], per_glyph: int = 4) -> np.ndarray:
    """Each glyph (left, red frame) beside up to `per_glyph` of its source crops,
    all at 4 px/mm: a glyph that does not look like its sources is visible here."""
    cell_w, cell_h = round(WIN_W_MM * PX), round(WIN_H_MM * PX)
    cap = 28
    rows = []
    for c in CHARSET:
        g = glyphs.get(c)
        row = np.full((cell_h + cap, cell_w * (per_glyph + 1), 3), 255, np.uint8)
        if g is not None:
            gi = _ink_to_png(g.image)
            gh, gw = gi.shape
            y, x = max(0, (cell_h - gh) // 2), max(0, (cell_w - gw) // 2)
            row[y : y + min(gh, cell_h), x : x + min(gw, cell_w)] = cv2.cvtColor(
                gi[:cell_h, :cell_w], cv2.COLOR_GRAY2BGR
            )
            cv2.rectangle(row, (0, 0), (cell_w - 1, cell_h - 1), (0, 0, 220), 2)
            for k, inst in enumerate(g.instances[:per_glyph]):
                src = (255 - inst.dark * 255).astype(np.uint8)
                row[:cell_h, (k + 1) * cell_w : (k + 2) * cell_w] = cv2.cvtColor(
                    src, cv2.COLOR_GRAY2BGR
                )
            text = f"{c}  {len(g.cars)} car(s)  {g.confidence}"
        else:
            text = f"{c}  MISSING"
        cv2.putText(row, text, (6, cell_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1)
        rows.append(row)
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    while len(right) < len(left):
        right.append(np.full_like(left[0], 255))
    return np.hstack([np.vstack(left), np.vstack(right)])


def write(
    paths: DataPaths, glyphs: dict[str, Glyph], logo: Glyph | None, sources: Sequence[str]
) -> Path:
    out = glyphs_dir(paths)
    out.mkdir(parents=True, exist_ok=True)
    entries: dict[str, Any] = {}
    for name, g in [*glyphs.items(), *([("logo", logo)] if logo else [])]:
        fname = f"{name}.png"
        cv2.imwrite(str(out / fname), g.image)
        w, h = g.ink_mm()
        entries[name] = {
            "file": fname,
            "cars": g.cars,
            "instances": [
                {"plate": i.plate, "source": i.source, "key": i.key} for i in g.instances
            ],
            "confidence": g.confidence,
            "ink_mm": [round(w, 2), round(h, 2)],
            "origin_mm": [round(g.origin_mm[0], 2), round(g.origin_mm[1], 2)],
            "size_px": [int(g.image.shape[1]), int(g.image.shape[0])],
        }
    manifest = {
        "px_per_mm": PX,
        "convention": "8-bit, 255 = ink; stored as photographed (ink ~6% taller than "
        "the 80 mm cell, see alpr/plates/glyphs.py)",
        "sources": list(sources),
        "missing": [c for c in CHARSET if c not in glyphs],
        "glyphs": entries,
    }
    atomic_write_text(out / MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")
    sheet = qa_sheet(glyphs)
    cv2.imwrite(str(out / "qa_sheet.png"), sheet)
    return out


def load_manifest(paths: DataPaths) -> dict[str, Any]:
    manifest = glyphs_dir(paths) / MANIFEST_NAME
    if not manifest.exists():
        raise LabelError(f"no glyphs at {manifest.parent} — run `annotate glyphs` first")
    return json.loads(manifest.read_text())


def load_glyphs(paths: DataPaths) -> dict[str, np.ndarray]:
    """The extracted glyphs (and "logo"), 8-bit with 255 = ink."""
    out = glyphs_dir(paths)
    manifest = out / MANIFEST_NAME
    if not manifest.exists():
        raise LabelError(f"no glyphs at {out} — run `annotate glyphs` first")
    entries = json.loads(manifest.read_text())["glyphs"]
    return {
        name: cv2.imread(str(out / e["file"]), cv2.IMREAD_GRAYSCALE) for name, e in entries.items()
    }
