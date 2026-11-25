"""Ficha Técnica MT-001 in millimetres — the single source of truth for plate layout.

Every other module that needs to know where something sits on a plate (the
rectifier's canvas, the glyph extractor's crop boxes, the renderer) reads it from
here, in millimetres, and converts to pixels at the last moment with `to_px`.

Two kinds of number live in this file, and they are kept apart on purpose:

* **SPEC** — transcribed from MT-001 as published (330 x 160 mm plate, 40 x 80 mm
  characters, 10 mm stroke, 8 mm character gap, 22 x 12 mm municipality letters).
* **MEASURED** — layout the published summaries do not give: the logo gutter
  between the letter and digit groups, and the vertical positions. These are
  provisional, measured from near-frontal photographs, and must be re-measured from
  labelled, rectified plates (`annotate measure`) before the renderer is trusted.
  Sprint 2.2's exit gate — a rendered plate within 2 mm of a real rectified one at
  every landmark — is what finally settles them.

`PROVENANCE` records which is which, field by field, and a test keeps it complete.

Coordinates: origin at the plate's top-left outer corner, x to the right, y down,
units in millimetres. The same orientation as image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum

Point = tuple[float, float]

#: The rectifier's output canvas (plan §4): width x height in pixels.
CANONICAL_SIZE: tuple[int, int] = (192, 96)

#: Renderer resolution (Sprint 2.2): 4 px/mm → a 1320 x 640 clean plate.
RENDER_PX_PER_MM = 4

#: Floating-point slack for layout sums, in millimetres.
_EPS = 1e-6


class GeometryError(ValueError):
    """A layout that does not fit on the plate."""


class Provenance(StrEnum):
    SPEC = "spec"  # MT-001 as transcribed
    MEASURED = "measured"  # provisional, from photographs — re-measure before trusting


@dataclass(frozen=True, slots=True)
class Rect:
    """An axis-aligned rectangle in millimetres (or pixels, after `PlateGeometry.to_px`)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> Point:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        """TL, TR, BR, BL — the project-wide corner order."""
        return (
            (self.x, self.y),
            (self.right, self.y),
            (self.right, self.bottom),
            (self.x, self.bottom),
        )

    def overlaps(self, other: Rect) -> bool:
        """True if the interiors intersect. Touching edges do not count."""
        return (
            self.x < other.right - _EPS
            and other.x < self.right - _EPS
            and self.y < other.bottom - _EPS
            and other.y < self.bottom - _EPS
        )

    def contains(self, other: Rect) -> bool:
        return (
            self.x <= other.x + _EPS
            and self.y <= other.y + _EPS
            and other.right <= self.right + _EPS
            and other.bottom <= self.bottom + _EPS
        )


@dataclass(frozen=True, slots=True)
class PlateGeometry:
    """A particular (car) plate, front or rear. Immutable.

    Horizontal layout, left to right:

        side_margin | L gap L gap L | logo_gutter | D gap D gap D | side_margin

    Vertical layout, top to bottom:

        char_top | characters (char_h) | muni_gap | municipality (muni_h) | bottom_margin

    `side_margin` and `bottom_margin` are derived so the layout always sums to the
    plate outline exactly; constructing a geometry that cannot fit raises.
    """

    plate_w: float = 330.0
    plate_h: float = 160.0

    char_w: float = 40.0
    char_h: float = 80.0
    char_stroke: float = 10.0
    char_gap: float = 8.0

    logo_gutter: float = 28.0
    char_top: float = 35.0
    muni_gap: float = 11.0

    muni_h: float = 22.0
    muni_char_w: float = 12.0
    muni_stroke: float = 5.0
    muni_spacing: float = 5.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not value > 0:
                raise GeometryError(f"{f.name} must be positive, got {value!r}")
        if self.char_stroke * 2 >= self.char_w:
            raise GeometryError("char_stroke is too thick for char_w")
        if self.side_margin < 0:
            raise GeometryError(
                f"characters and gutter are {self.plate_w - 2 * self.side_margin:.1f} mm wide; "
                f"the plate is {self.plate_w} mm"
            )
        if self.bottom_margin < 0:
            raise GeometryError(
                f"vertical layout needs {self.plate_h - self.bottom_margin:.1f} mm; "
                f"the plate is {self.plate_h} mm"
            )

    # -- derived dimensions ------------------------------------------------ #

    @property
    def aspect(self) -> float:
        return self.plate_w / self.plate_h

    @property
    def group_w(self) -> float:
        """Width of one three-character group, outer edge to outer edge."""
        return 3 * self.char_w + 2 * self.char_gap

    @property
    def side_margin(self) -> float:
        return (self.plate_w - 2 * self.group_w - self.logo_gutter) / 2.0

    @property
    def bottom_margin(self) -> float:
        return self.plate_h - self.char_top - self.char_h - self.muni_gap - self.muni_h

    # -- regions ------------------------------------------------------------ #

    @property
    def outline(self) -> Rect:
        return Rect(0.0, 0.0, self.plate_w, self.plate_h)

    def char_boxes(self) -> tuple[Rect, ...]:
        """The six character cells, left to right: three letters, then three digits."""
        boxes = []
        x = self.side_margin
        for i in range(6):
            if i == 3:
                x += self.logo_gutter - self.char_gap
            boxes.append(Rect(x, self.char_top, self.char_w, self.char_h))
            x += self.char_w + self.char_gap
        return tuple(boxes)

    @property
    def logo_gutter_box(self) -> Rect:
        """The full-height gap between the groups where the MinTransporte logo sits."""
        left = self.side_margin + self.group_w
        return Rect(left, self.char_top, self.logo_gutter, self.char_h)

    @property
    def char_band(self) -> Rect:
        """All six characters and the gutter: the horizontal strip that gets read."""
        return Rect(
            self.side_margin, self.char_top, self.plate_w - 2 * self.side_margin, self.char_h
        )

    @property
    def muni_band(self) -> Rect:
        """The strip holding the municipality name, spanning the character band's width."""
        top = self.char_top + self.char_h + self.muni_gap
        return Rect(self.side_margin, top, self.plate_w - 2 * self.side_margin, self.muni_h)

    def muni_text_width(self, n_chars: int) -> float:
        if n_chars <= 0:
            return 0.0
        return n_chars * self.muni_char_w + (n_chars - 1) * self.muni_spacing

    def muni_text_box(self, text: str) -> Rect:
        """Where a municipality name sits, centred in the band.

        Raises for names too long to fit at MT-001 letter size. Real plates
        abbreviate long names ("PTO. CARREÑO", "C. BOLIVAR"); the caller decides how,
        this module only refuses to silently overflow.
        """
        width = self.muni_text_width(len(text))
        band = self.muni_band
        if width > band.w + _EPS:
            raise GeometryError(
                f"municipality {text!r} needs {width:.0f} mm at MT-001 letter size; "
                f"the band is {band.w:.0f} mm — abbreviate it"
            )
        return Rect(band.x + (band.w - width) / 2.0, band.y, width, band.h)

    # -- landmarks and pixel conversion ------------------------------------- #

    def landmarks(self) -> dict[str, Point]:
        """Named points in millimetres — what Sprint 2.2's 2 mm overlay check compares."""
        out: dict[str, Point] = {}
        for name, pt in zip(("tl", "tr", "br", "bl"), self.outline.corners, strict=True):
            out[f"plate_{name}"] = pt
        for i, box in enumerate(self.char_boxes(), start=1):
            for name, pt in zip(("tl", "tr", "br", "bl"), box.corners, strict=True):
                out[f"char{i}_{name}"] = pt
        out["logo_center"] = self.logo_gutter_box.center
        band = self.muni_band
        out["muni_top_center"] = (band.center[0], band.y)
        out["muni_bottom_center"] = (band.center[0], band.bottom)
        return out

    def to_px(self, rect: Rect, size: tuple[int, int]) -> Rect:
        """Map a millimetre rectangle onto a canvas of `size` = (width, height) pixels.

        The scale may differ per axis. On the 192 x 96 canonical canvas it does, by
        3%: 192/96 = 2.000 against the plate's 2.0625. That squeeze is deliberate
        (plan, Sprint 2.1) — constant across train and test, so the network never
        sees it as variation, and every pooling stage stays a power of two.
        """
        sx = size[0] / self.plate_w
        sy = size[1] / self.plate_h
        return Rect(rect.x * sx, rect.y * sy, rect.w * sx, rect.h * sy)

    def point_to_px(self, point: Point, size: tuple[int, int]) -> Point:
        return (point[0] * size[0] / self.plate_w, point[1] * size[1] / self.plate_h)

    def render_size(self, px_per_mm: float = RENDER_PX_PER_MM) -> tuple[int, int]:
        return (round(self.plate_w * px_per_mm), round(self.plate_h * px_per_mm))


#: Which numbers came from the spec and which are provisional measurements.
PROVENANCE: dict[str, Provenance] = {
    "plate_w": Provenance.SPEC,
    "plate_h": Provenance.SPEC,
    "char_w": Provenance.SPEC,
    "char_h": Provenance.SPEC,
    "char_stroke": Provenance.SPEC,
    "char_gap": Provenance.SPEC,
    "logo_gutter": Provenance.MEASURED,
    "char_top": Provenance.MEASURED,
    "muni_gap": Provenance.MEASURED,
    "muni_h": Provenance.SPEC,
    "muni_char_w": Provenance.SPEC,
    "muni_stroke": Provenance.SPEC,
    "muni_spacing": Provenance.SPEC,
}

#: The particular car plate. Import this; do not construct your own.
MT001 = PlateGeometry()
