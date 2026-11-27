"""`render(PlateSpec) -> image`: a clean, embossed particular plate at 4 px/mm.

The plate is drawn at its true size from `geometry` (1320 x 640 px for 330 x 160
mm) using the extracted glyphs, then lit:

* **Relief.** Characters, municipality letters and the border rim are raised
  `emboss_mm` above the sheet, with a soft bevel. Their slopes are what catch the
  light: yellow-on-yellow relief reads as embossed rather than printed.
* **Shading.** Lambertian diffuse plus Blinn-Phong specular from a light vector,
  normalised so a flat patch keeps exactly its base colour.
* **Cast shadow.** Raised parts shade the sheet away from the light.
* **Parallax** (`view`). Seen at an angle, the raised face shifts against the sheet
  by `emboss_mm * tan(angle)` and the walls between show. This is why characters
  photographed from above measure ~85 mm tall against 80 mm on the plate: their top
  edge stays put while the face slides down. Phase 3 samples it; frontal is (0, 0).

Glyphs are stored as photographed (ink ~42 x 85 mm). One pair of scale factors,
computed over the whole set, maps them to the 41 x 80 mm cell, so a narrow "1"
stays narrow and keeps its place in the cell.

Everything that does not change between renders (glyph masks at cell size, the
rim, the holes, the logo) is prepared once per `PlateRenderer`; a render composes
and lights. Phase 3 calls this ~500,000 times, so the budget is < 15 ms.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np

from alpr.data_sources.storage import DataPaths
from alpr.plates.geometry import MT001, RENDER_PX_PER_MM, PlateGeometry
from alpr.plates.glyphs import load_glyphs, load_manifest
from alpr.plates.labels import CHARSET, LabelError

PX = float(RENDER_PX_PER_MM)
GLYPH_PX = 4.0  # resolution the glyph images are stored at (alpr.plates.glyphs)
Color = tuple[int, int, int]

#: Painted, raised border along the plate edge (photographed; provisional).
RIM_MM = 3.5
#: Mounting slots: centres (mm) and size, measured on rectified photos (provisional).
SLOT_CENTRES_MM = ((37.0, 15.0), (293.0, 15.0), (37.0, 145.0), (293.0, 145.0))
SLOT_SIZE_MM = (10.0, 5.0)
SLOT_COLOR: Color = (55, 55, 55)
#: Glyphs whose photographed width says nothing about the cell width.
NARROW = frozenset("1IJ")
#: Advance widths (mm) for the punctuation in municipality names.
PUNCT_ADVANCE_MM = {" ": 8.0, ".": 5.0, ",": 5.0, "-": 10.0}
BEVEL_MM = 0.6
#: Municipality letters stand lower than the main characters: in photos taken from
#: above, the characters' ink runs ~5 mm past the cell while the municipality
#: letters' runs ~0.5 mm. Relative to `emboss_mm`; fitted by `annotate overlay`.
MUNI_RELIEF = 0.15


class RenderError(LabelError):
    """A plate that cannot be drawn: unknown characters, missing glyphs."""


@dataclass(frozen=True, slots=True)
class PlateStyle:
    background: Color
    ink: Color
    name: str = ""


PARTICULAR = PlateStyle(background=(247, 196, 22), ink=(24, 24, 24), name="particular")
#: For Phase 3 negatives: same outline and layout, white sheet.
PUBLICO = PlateStyle(background=(236, 236, 230), ink=(24, 24, 24), name="publico")


@dataclass(frozen=True, slots=True)
class Lighting:
    """`direction` points from the plate towards the light, in plate coordinates:
    x right, y down, z out of the plate. Normalised on use."""

    direction: tuple[float, float, float] = (0.0, -0.6, 0.8)
    ambient: float = 0.45
    diffuse: float = 0.55
    specular: float = 0.18
    shininess: float = 24.0
    shadow: float = 0.7  # 0: no cast shadow, 1: shadowed sheet gets ambient only

    def unit(self) -> np.ndarray:
        v = np.asarray(self.direction, np.float32)
        n = float(np.linalg.norm(v))
        if n == 0 or v[2] <= 0:
            raise RenderError("the light must be in front of the plate (z > 0)")
        return v / n


@dataclass(frozen=True, slots=True)
class PlateSpec:
    """What to draw. `text` is six characters from the 36; `view` is the tangent of
    the viewing angle along x and y (0, 0 = frontal; (0, 1) = 45 degrees from above)."""

    text: str
    municipality: str = "BOGOTA D.C."
    style: PlateStyle = PARTICULAR
    lighting: Lighting = field(default_factory=Lighting)
    view: tuple[float, float] = (0.0, 0.0)
    emboss_mm: float = 1.5
    walls_painted: bool = True


def _line_kernel(dx: float, dy: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Kernel whose dilation is the union of shifts by t*(dx, dy), t in [0, 1]."""
    rx, ry = math.ceil(abs(dx)), math.ceil(abs(dy))
    k = np.zeros((2 * ry + 1, 2 * rx + 1), np.uint8)
    steps = max(rx, ry, 1) * 2
    for i in range(steps + 1):
        t = i / steps
        k[round(ry - t * dy), round(rx - t * dx)] = 1
    return k, (rx, ry)


def _sweep(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return mask
    kernel, anchor = _line_kernel(dx, dy)
    return cv2.dilate(mask, kernel, anchor=anchor)


def _shift(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return mask
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, m, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)


def _binarise(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a soft glyph, cut at half darkness, anti-alias: a crisp clean edge."""
    interp = cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_CUBIC
    small = cv2.resize(img.astype(np.float32) / 255.0, (max(1, w), max(1, h)), interpolation=interp)
    crisp = (small >= 0.5).astype(np.float32)
    return cv2.GaussianBlur(crisp, (3, 3), 0.6)


@dataclass
class _Sprite:
    mask: np.ndarray  # float32 [0, 1]
    dx: float  # top-left relative to the anchor, px
    dy: float
    advance: float = 0.0  # municipality letters: horizontal advance, px


class PlateRenderer:
    """Glyphs prepared once at the cell size of `geometry`; `render` composes and lights."""

    def __init__(
        self,
        glyphs: dict[str, np.ndarray],
        origins: dict[str, tuple[float, float]],
        geometry: PlateGeometry = MT001,
        px_per_mm: float = PX,
    ) -> None:
        """`px_per_mm` is the render resolution. The glyph images are 4 px/mm (as
        extracted); they are resampled once, here, to this resolution."""
        missing = [c for c in CHARSET if c not in glyphs]
        if missing:
            raise RenderError(f"glyphs missing for {missing} — run `annotate glyphs`")
        self.geometry = g = geometry
        self.px = float(px_per_mm)
        self._source = (glyphs, origins)
        self._others: dict[float, PlateRenderer] = {}
        k = self.px / GLYPH_PX  # glyph pixels -> render pixels
        self.size = g.render_size(self.px)

        def ink_box(img: np.ndarray) -> tuple[int, int, int, int]:
            ys, xs = np.nonzero(img >= 128)
            return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

        boxes = {c: ink_box(glyphs[c]) for c in CHARSET}
        ink_h = statistics.median((b[3] - b[1]) / GLYPH_PX for b in boxes.values())
        ink_w = statistics.median(
            (b[2] - b[0]) / GLYPH_PX for c, b in boxes.items() if c not in NARROW
        )
        #: photographed mm -> plate mm, one pair for the whole set
        self.scale = (g.char_w / ink_w, g.char_h / ink_h)
        self.photo_ink_mm = (ink_w, ink_h)
        sx, sy = self.scale

        # Main characters: anchor = (slot centre, characters' top line).
        self.chars: dict[str, _Sprite] = {}
        for c in CHARSET:
            img, (ox, oy) = glyphs[c], origins[c]
            x0, y0, _, _ = boxes[c]
            # Place so the ink's top lands on the top line exactly; keep x as measured.
            top_mm = oy + y0 / GLYPH_PX
            w, h = round(img.shape[1] * sx * k), round(img.shape[0] * sy * k)
            self.chars[c] = _Sprite(
                _binarise(img, w, h), ox * sx * self.px, (oy - top_mm) * sy * self.px
            )

        # Municipality letters: the same letterforms at MT-001 strip size, stroke
        # thickened towards the 5 mm the spec gives (scaling alone leaves ~3 mm).
        msx = g.muni_char_w / ink_w
        msy = g.muni_h / ink_h
        stroke_now = g.char_stroke * msx
        grow = max(0, round((g.muni_stroke - stroke_now) / 2 * self.px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2)
        self.muni: dict[str, _Sprite] = {}
        for c in CHARSET:
            img, (ox, oy) = glyphs[c], origins[c]
            x0, y0, x1, y1 = boxes[c]
            crop = img[y0:y1, x0:x1]
            # Shrink first, then thicken: the letter ends up at MT-001 size.
            w = max(1, round(crop.shape[1] * msx * k) - 2 * grow)
            h = max(1, round(crop.shape[0] * msy * k) - 2 * grow)
            m = _binarise(crop, w, h)
            m = cv2.copyMakeBorder(m, grow, grow, grow, grow, cv2.BORDER_CONSTANT, value=0)
            if grow:
                m = cv2.dilate(m, kernel)
            adv = (g.muni_char_w if c not in NARROW else max(w / self.px, 5.0)) * self.px
            self.muni[c] = _Sprite(m, (adv - m.shape[1]) / 2, -grow, adv)
        self.muni_pad = grow
        self.muni.update(self._punctuation())

        # Static layers.
        W, H = self.size
        rim = np.ones((H, W), np.float32)
        r = round(RIM_MM * self.px)
        rim[r:-r, r:-r] = 0
        self.rim = cv2.GaussianBlur(rim, (3, 3), 0.6)
        self.slots = np.zeros((H, W), np.float32)
        for cx, cy in SLOT_CENTRES_MM:
            axes = (round(SLOT_SIZE_MM[0] / 2 * self.px), round(SLOT_SIZE_MM[1] / 2 * self.px))
            cv2.ellipse(
                self.slots, (round(cx * self.px), round(cy * self.px)), axes, 0, 0, 360, 1.0, -1
            )
        self.slot_idx = np.nonzero(self.slots > 0)
        self.logo = np.zeros((H, W), np.float32)
        if "logo" in glyphs:
            img, (ox, oy) = glyphs["logo"], origins["logo"]
            w, h = round(img.shape[1] * sx * k), round(img.shape[0] * sy * k)
            gut = g.logo_gutter_box
            lx = round((gut.center[0] + ox * sx) * self.px)
            ly = round((g.char_top + g.char_h / 2 + oy * sy) * self.px)
            self._paste(self.logo, _binarise(img, w, h), lx, ly)

    def at(self, px_per_mm: float) -> PlateRenderer:
        """This renderer at another resolution, built once and cached."""
        if px_per_mm == self.px:
            return self
        if px_per_mm not in self._others:
            glyphs, origins = self._source
            self._others[px_per_mm] = PlateRenderer(glyphs, origins, self.geometry, px_per_mm)
        return self._others[px_per_mm]

    def _punctuation(self) -> dict[str, _Sprite]:
        g = self.geometry
        h = round(g.muni_h * self.px)
        s = round(g.muni_stroke * self.px)
        out = {}
        for ch, adv_mm in PUNCT_ADVANCE_MM.items():
            adv = round(adv_mm * self.px)
            m = np.zeros((h, adv), np.float32)
            if ch == ".":
                m[h - s :, (adv - s) // 2 : (adv - s) // 2 + s] = 1
            elif ch == ",":
                m[h - s :, (adv - s) // 2 : (adv - s) // 2 + s] = 1
                m[h - s // 2 :, max(0, (adv - s) // 2 - s // 2) : (adv - s) // 2] = 1
            elif ch == "-":
                m[(h - s) // 2 : (h - s) // 2 + s, s // 2 : adv - s // 2] = 1
            out[ch] = _Sprite(m, 0.0, 0.0, float(adv))
        return out

    @staticmethod
    def _paste(canvas: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
        H, W = canvas.shape
        h, w = mask.shape
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        sub = mask[y0 - y : y1 - y, x0 - x : x1 - x]
        np.maximum(canvas[y0:y1, x0:x1], sub, out=canvas[y0:y1, x0:x1])

    # -- composition ------------------------------------------------------- #

    def _municipality(self, canvas: np.ndarray, name: str) -> None:
        g = self.geometry
        sprites = []
        for ch in name.upper():
            if ch not in self.muni:
                raise RenderError(f"municipality {name!r}: no letterform for {ch!r}")
            sprites.append(self.muni[ch])
        gap = g.muni_spacing * self.px
        width = sum(s.advance for s in sprites) + gap * max(0, len(sprites) - 1)
        band = g.muni_band
        pad = self.muni_pad
        h = round(g.muni_h * self.px) + 2 * pad
        strip = np.zeros((h, max(1, math.ceil(width)) + 2 * pad), np.float32)
        x = float(pad)
        for s in sprites:
            self._paste(strip, s.mask, round(x + s.dx), round(pad + s.dy))
            x += s.advance + gap
        band_w = band.w * self.px
        if strip.shape[1] - 2 * pad > band_w:  # auto-fit: condense long names
            new_w = max(1, round(band_w + 2 * pad * band_w / (strip.shape[1] - 2 * pad)))
            strip = cv2.resize(strip, (new_w, h), interpolation=cv2.INTER_AREA)
        sx = round(band.center[0] * self.px - strip.shape[1] / 2)
        sy = round(band.y * self.px) - pad
        self._paste(canvas, strip, sx, sy)

    def relief(self, spec: PlateSpec) -> tuple[np.ndarray, np.ndarray]:
        """The raised, painted parts as float32 masks in [0, 1]: the full-height
        relief (characters and rim) and the low municipality lettering."""
        text = spec.text.upper()
        if len(text) != 6:
            raise RenderError(f"a plate has six characters, got {spec.text!r}")
        bad = [c for c in text if c not in self.chars]
        if bad:
            raise RenderError(f"no glyph for {bad} in {spec.text!r}")
        canvas = self.rim.copy()
        for box, ch in zip(self.geometry.char_boxes(), text, strict=True):
            s = self.chars[ch]
            cx = (box.x + box.w / 2) * self.px
            self._paste(canvas, s.mask, round(cx + s.dx), round(box.y * self.px + s.dy))
        muni = np.zeros_like(canvas)
        self._municipality(muni, spec.municipality)
        return canvas, muni

    def render(self, spec: PlateSpec) -> np.ndarray:
        """RGB uint8, (640, 1320, 3) for MT-001 at 4 px/mm."""
        relief, muni = self.relief(spec)
        light = spec.lighting
        L = light.unit()
        e = spec.emboss_mm * self.px  # emboss height in px

        vx, vy = spec.view
        em = e * MUNI_RELIEF
        face_hi, face_lo = _shift(relief, vx * e, vy * e), _shift(muni, vx * em, vy * em)
        face = cv2.max(face_hi, face_lo)
        swept = cv2.max(_sweep(relief, vx * e, vy * e), _sweep(muni, vx * em, vy * em))
        ink = swept if spec.walls_painted else face

        # Height (px units, so gradients are slopes). Only the bevels have slope;
        # a flat patch keeps its base colour exactly, so light only those pixels.
        height = cv2.GaussianBlur(
            cv2.addWeighted(face_hi, e, face_lo, em, 0.0), (0, 0), BEVEL_MM * self.px
        )
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        flat = light.ambient + light.diffuse * float(L[2])
        shade = np.ones(face.shape, np.float32)
        glint = np.zeros(face.shape, np.float32)
        idx = np.flatnonzero((np.abs(gx) + np.abs(gy)) > 1e-4)
        if idx.size:
            ex, ey = gx.ravel()[idx], gy.ravel()[idx]
            inv = 1.0 / np.sqrt(ex * ex + ey * ey + 1.0)
            n_dot_l = np.clip((-ex * L[0] - ey * L[1] + L[2]) * inv, 0.0, None)
            shade.ravel()[idx] = (light.ambient + light.diffuse * n_dot_l) / flat
            view = np.array([-vx, -vy, 1.0], np.float32)
            view /= np.linalg.norm(view)
            half = L + view
            half /= np.linalg.norm(half)
            n_dot_h = np.clip((-ex * half[0] - ey * half[1] + half[2]) * inv, 0.0, None)
            spec_flat = float(half[2]) ** light.shininess
            glint.ravel()[idx] = (
                255.0 * light.specular * np.clip(n_dot_h**light.shininess - spec_flat, 0.0, None)
            )

        # Cast shadow on the sheet, away from the light: only ambient reaches it.
        sdx, sdy = -L[0] / L[2] * e, -L[1] / L[2] * e
        cast = cv2.max(
            _sweep(face_hi, sdx, sdy),
            _sweep(face_lo, sdx * MUNI_RELIEF, sdy * MUNI_RELIEF),
        )
        shadow = cv2.subtract(cast, face)
        shadow = cv2.GaussianBlur(shadow, (0, 0), 1.0)
        k = light.shadow * (1.0 - light.ambient / flat)
        shade = cv2.subtract(shade, cv2.multiply(shadow, shade, scale=k))

        if not spec.walls_painted:
            walls = cv2.subtract(swept, face)
            shade = cv2.subtract(shade, cv2.multiply(walls, shade, scale=0.45))

        # Colour per channel: (bg + (fg - bg) * paint) * shade + glint.
        paint = cv2.max(ink, self.logo)
        lit_paint = cv2.multiply(paint, shade)
        channels = []
        for c in range(3):
            bgc = float(spec.style.background[c])
            d = float(spec.style.ink[c]) - bgc
            ch = cv2.scaleAdd(lit_paint, d, cv2.multiply(shade, bgc))
            channels.append(cv2.add(ch, glint))
        out = cv2.merge(channels)
        ys, xs = self.slot_idx
        a = self.slots[ys, xs][:, None]
        out[ys, xs] = out[ys, xs] * (1.0 - a) + np.asarray(SLOT_COLOR, np.float32) * a
        return cv2.convertScaleAbs(out)


@lru_cache(maxsize=4)
def _default_renderer(root: str) -> PlateRenderer:
    return renderer_from_assets(DataPaths(root))


def renderer_from_assets(paths: DataPaths, geometry: PlateGeometry = MT001) -> PlateRenderer:
    manifest = load_manifest(paths)
    glyphs = load_glyphs(paths)
    origins = {k: tuple(v.get("origin_mm", (0.0, 0.0))) for k, v in manifest["glyphs"].items()}
    return PlateRenderer(glyphs, origins, geometry)


def render(spec: PlateSpec, renderer: PlateRenderer | None = None) -> np.ndarray:
    """The Phase 2 contract: a PlateSpec in, a 1320 x 640 RGB plate out."""
    r = renderer or _default_renderer(str(DataPaths.default().root))
    return r.render(spec)


# --------------------------------------------------------------------------- #
# The Sprint 2.2 gate: does a rendered plate land where a photographed one does?
# --------------------------------------------------------------------------- #

LANDMARKS = (
    "char_top",
    "char_bottom",
    "slot1",
    "slot2",
    "slot3",
    "slot4",
    "slot5",
    "slot6",
    "muni_top",
    "muni_bottom",
)


def landmarks_from_layout(
    layout: dict[str, list[float]], geometry: PlateGeometry = MT001
) -> dict[str, float]:
    """Landmark positions in mm from `annotate.measure_layout` output (medians):
    the characters' top and bottom lines, the six slot centres, the municipality
    strip's top and bottom."""
    med = {k: statistics.median(v) for k, v in layout.items() if v}
    top, h = med["char_top"], med["char_h"]
    pitch = med["char_gap"] + geometry.char_w
    c1 = med["side_margin"] + geometry.char_w / 2
    centres = [c1, c1 + pitch, c1 + 2 * pitch]
    c4 = centres[2] + med["logo_gutter"] + geometry.char_w
    centres += [c4, c4 + pitch, c4 + 2 * pitch]
    out = {"char_top": top, "char_bottom": top + h}
    out.update({f"slot{k + 1}": c for k, c in enumerate(centres)})
    if "muni_gap" in med:
        out["muni_top"] = top + h + med["muni_gap"]
        out["muni_bottom"] = out["muni_top"] + med["muni_h"]
    return out


def rendered_landmarks(
    renderer: PlateRenderer, texts: list[str], view: tuple[float, float] = (0.0, 0.0)
) -> dict[str, float]:
    from alpr.plates.annotate import measure_layout

    W, H = renderer.size
    full = [(0.0, 0.0), (float(W), 0.0), (float(W), float(H)), (0.0, float(H))]
    samples = [(renderer.render(PlateSpec(t, "MEDELLIN", view=view)), full) for t in texts]
    layout, skipped = measure_layout(samples, renderer.geometry)
    if skipped == len(samples):
        raise RenderError("could not find six characters on any rendered plate")
    return landmarks_from_layout(layout, renderer.geometry)


#: The viewing tilt that reproduces the Roboflow photos (taken from above the
#: bumper), fitted by `annotate overlay`. Frontal is (0, 0).
ROBOFLOW_VIEW = (0.0, 3.0)
GATE_MM = 2.0


@dataclass
class OverlayReport:
    real: dict[str, float]
    frontal: dict[str, float]
    matched: dict[str, float]
    view: tuple[float, float]
    plates: int
    render_ms: float
    sheet: np.ndarray  # real rectified plates beside renders of the same text

    def errors(self, which: str = "matched") -> dict[str, float]:
        other = self.matched if which == "matched" else self.frontal
        return {k: other[k] - v for k, v in self.real.items() if k in other}

    def passes(self) -> bool:
        return max(abs(v) for v in self.errors().values()) <= GATE_MM


def overlay_report(
    paths: DataPaths,
    sources: list[str],
    view: tuple[float, float] = ROBOFLOW_VIEW,
    renderer: PlateRenderer | None = None,
    n_sheet: int = 6,
) -> OverlayReport:
    """Measure real labelled plates and renders of the same strings the same way,
    frontal and at `view`, and time the renderer."""
    import time

    from alpr.data_sources.base import PlateType
    from alpr.plates.annotate import _labelled_quads, measure_layout
    from alpr.plates.labels import load_labelled_image
    from alpr.plates.rectify import rectify

    r = renderer or renderer_from_assets(paths)
    pairs = [
        (item, lb)
        for source in sources
        for item, lb in _labelled_quads(paths, source)
        if lb.plate_type is PlateType.PARTICULAR and lb.plate and len(lb.plate) == 6
    ]
    if not pairs:
        raise RenderError("no labelled particular plates to compare against")
    samples = [(load_labelled_image(item.image, lb), lb.corners) for item, lb in pairs]
    layout, skipped = measure_layout(samples, r.geometry)
    real = landmarks_from_layout(layout, r.geometry)
    texts = sorted({lb.plate for _, lb in pairs})[:40]

    t0 = time.perf_counter()
    reps = 30
    for k in range(reps):
        r.render(PlateSpec(texts[k % len(texts)], view=view))
    render_ms = (time.perf_counter() - t0) / reps * 1000

    tiles = []
    step = max(1, len(pairs) // n_sheet)
    for (_, lb), (image, _) in list(zip(pairs, samples, strict=True))[::step][:n_sheet]:
        real_flat = rectify(image, lb.corners, r.size)
        fake = r.render(PlateSpec(lb.plate, "MEDELLIN", view=view))
        tiles.append(np.hstack([real_flat, fake]))
    sheet = cv2.resize(np.vstack(tiles), None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return OverlayReport(
        real=real,
        frontal=rendered_landmarks(r, texts),
        matched=rendered_landmarks(r, texts, view),
        view=view,
        plates=len(pairs) - skipped,
        render_ms=render_ms,
        sheet=sheet,
    )
