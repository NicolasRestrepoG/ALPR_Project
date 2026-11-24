"""Core types for the acquisition layer: Record, License, Status, corner ordering.

This module has **zero third-party imports**, deliberately. Phase 1 must be fully
testable with nothing but the standard library and pytest installed — no PyTorch,
no OpenCV, no numpy. Everything downstream depends on `Record`, not on how any
particular source produced it.

The single most important thing in this file is the corner-ordering contract:

    Corners are ALWAYS ordered TL, TR, BR, BL.

CCPD publishes its four vertices starting at the *right-bottom* vertex. Consume
them in file order and the detector trains to predict rotated corners, the
rectifier emits upside-down plates, and the recognizer converges to near-chance
with no error message anywhere. `order_corners` converts once, here, with a test.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

Point = tuple[float, float]
Corners = tuple[Point, Point, Point, Point]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class AcquisitionError(Exception):
    """Base class for every error raised by the acquisition layer."""


class CornerOrderError(AcquisitionError):
    """Four points are not a usable quadrilateral, or are not in TL/TR/BR/BL order."""


class RecordValidationError(AcquisitionError):
    """A Record was constructed with fields that violate the contract."""


# --------------------------------------------------------------------------- #
# Licences
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class License:
    """A licence, with the two facts the project actually has to act on.

    `commercial_ok` gates deployment. `attribution_per_image` gates whether a
    record may exist at all without a source URL (Platesmania's backlink is a
    licence condition, not a nicety).
    """

    key: str
    name: str
    spdx: str | None
    commercial_ok: bool
    attribution_required: bool = False
    attribution_per_image: bool = False
    url: str | None = None
    note: str = ""


#: The canonical licence table. Records store the *key*; `commercial_ok` is never
#: duplicated into a record, so there is exactly one place that decides it.
LICENSES: dict[str, License] = {
    lic.key: lic
    for lic in (
        License(
            key="mit",
            name="MIT",
            spdx="MIT",
            commercial_ok=True,
            url="https://opensource.org/license/mit",
        ),
        License(
            key="cc-by-4.0",
            name="Creative Commons Attribution 4.0",
            spdx="CC-BY-4.0",
            commercial_ok=True,
            attribution_required=True,
            url="https://creativecommons.org/licenses/by/4.0/",
        ),
        License(
            key="public-gov-co",
            name="Colombian public government data",
            spdx=None,
            commercial_ok=True,
            note="Published by a Colombian public entity for open reuse.",
        ),
        License(
            key="own-work",
            name="Own work",
            spdx=None,
            commercial_ok=True,
            note="Photographed by the project owner, with written permission where required.",
        ),
        License(
            key="academic-nc",
            name="Academic non-commercial (signed agreement)",
            spdx=None,
            commercial_ok=False,
            note=(
                "UFPR-ALPR and RodoSol-ALPR. Any checkpoint whose weights were "
                "influenced by these cannot be commercially deployed."
            ),
        ),
        License(
            key="platesmania-terms",
            name="Platesmania site terms",
            spdx=None,
            commercial_ok=False,
            attribution_required=True,
            attribution_per_image=True,
            url="https://platesmania.com/about",
            note="Requires an active backlink for every single image used.",
        ),
        License(
            key="youtube-tos",
            name="YouTube Terms of Service (per-video)",
            spdx=None,
            commercial_ok=False,
            attribution_required=True,
            attribution_per_image=True,
            note="Prefer CC-BY videos; per-video terms still govern. Evaluation only.",
        ),
        License(
            key="unknown",
            name="Unverified / unknown",
            spdx=None,
            commercial_ok=False,
            note=(
                "Default for anything whose terms have not been read. Excluded from "
                "--commercial-only by construction, so forgetting to check fails safe."
            ),
        ),
    )
}


def get_license(key: str) -> License:
    try:
        return LICENSES[key]
    except KeyError:
        raise RecordValidationError(
            f"unknown licence key {key!r}; known keys: {sorted(LICENSES)}"
        ) from None


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class PlateType(StrEnum):
    """What kind of plate is in the image.

    Only PARTICULAR is ever *read*. Everything else is a negative that trains the
    rejection head — recognizing that they are not readable is in scope, reading
    them is not (plan §2).
    """

    PARTICULAR = "particular"  # yellow, LLL·logo·DDD — the only readable class
    PUBLICO = "publico"  # white ground, same layout — the hardest negative
    MOTO = "moto"  # two-line layout, out of scope by construction
    DIPLOMATIC = "diplomatic"
    OFFICIAL = "official"
    TRAILER = "trailer"
    TEMPORARY = "temporary"
    FOREIGN = "foreign"  # non-Colombian, e.g. every CCPD plate
    NON_PLATE = "non_plate"  # a patch with no plate at all
    UNKNOWN = "unknown"

    @property
    def readable(self) -> bool:
        """True only for the one class v1 is allowed to emit characters for."""
        return self is PlateType.PARTICULAR

    @property
    def is_negative(self) -> bool:
        """True for classes that are detected and rejected, never read (plan §2).

        Every plate type except `particular` is a negative — motorcycle, público,
        diplomatic, official, trailer and temporary plates all train the rejection
        head, and so does a frame with no plate at all.

        `UNKNOWN` is deliberately excluded. It means "could not determine", not a
        class, so counting it as a negative would inflate the negative tally with
        images that have no ground truth either way.
        """
        return self not in (PlateType.PARTICULAR, PlateType.UNKNOWN)


class Status(StrEnum):
    """Outcome of a fetch or index operation for one source."""

    NOT_STARTED = "not_started"
    READY = "ready"  # files present on disk and verified
    DOWNLOADED = "downloaded"  # fetched this run
    INDEXED = "indexed"
    MANUAL_REQUIRED = "manual_required"  # instructions printed; not an error
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"  # upstream gone / credentials missing
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


class Winding(StrEnum):
    """Traversal direction of a polygon, named as it appears **on screen**.

    Image coordinates put y downward, which flips the usual sign convention. A
    positive shoelace sum here means the points trace clockwise as a person looking
    at the image would describe it. TL → TR → BR → BL is clockwise on screen.
    """

    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"
    DEGENERATE = "degenerate"


def signed_area(points: Sequence[Point]) -> float:
    """Shoelace signed area in image coordinates (y down, positive = clockwise on screen)."""
    n = len(points)
    if n < 3:
        return 0.0
    total = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def winding(points: Sequence[Point], *, tol: float = 1e-9) -> Winding:
    area = signed_area(points)
    if abs(area) <= tol:
        return Winding.DEGENERATE
    return Winding.CLOCKWISE if area > 0 else Winding.COUNTERCLOCKWISE


def _as_points(points: Iterable[Any]) -> list[Point]:
    out: list[Point] = []
    for p in points:
        pair = (p["x"], p["y"]) if isinstance(p, dict) else tuple(p)
        if len(pair) != 2:
            raise CornerOrderError(f"expected an (x, y) pair, got {p!r}")
        x, y = pair
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            raise CornerOrderError(f"corner coordinates must be numeric, got {p!r}")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise CornerOrderError(f"corner coordinates must be finite, got {p!r}")
        out.append((float(x), float(y)))
    return out


def order_corners(points: Iterable[Any]) -> Corners:
    """Order any four quadrilateral vertices as TL, TR, BR, BL.

    Works regardless of the input ordering or winding direction. The algorithm is
    geometric rather than index-based, so it is the same function whether the input
    came from CCPD's right-bottom-first convention, from the Phase 2 annotation
    tool, or from a detector's keypoint head:

    1. sort the vertices cyclically by angle about their centroid — this recovers a
       simple (non-self-intersecting) polygon whatever order they arrived in;
    2. force clockwise-on-screen traversal;
    3. rotate the cycle so it starts at the vertex nearest the image origin.

    Raises `CornerOrderError` for fewer/more than four points, for non-finite or
    non-numeric coordinates, and for degenerate quads (collinear or coincident
    vertices) — which are the shapes that silently produce a garbage homography.
    """
    pts = _as_points(points)
    if len(pts) != 4:
        raise CornerOrderError(f"expected exactly 4 corners, got {len(pts)}")

    if len(set(pts)) != 4:
        raise CornerOrderError(f"corners must be four distinct points, got {pts}")

    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    cyclic = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    area = signed_area(cyclic)
    if abs(area) < 1e-6:
        raise CornerOrderError(f"degenerate quadrilateral (near-zero area): {pts}")
    if area < 0:
        cyclic.reverse()

    # Start at the top-left: of the four cyclic rotations, the one whose first
    # vertex minimises x + y. Within the project's viewpoint envelope (roll ±8°,
    # yaw ±25°) this is unambiguous.
    start = min(range(4), key=lambda i: cyclic[i][0] + cyclic[i][1])
    ordered = cyclic[start:] + cyclic[:start]
    return (ordered[0], ordered[1], ordered[2], ordered[3])


def assert_corner_order(points: Sequence[Any], *, strict: bool = False) -> Corners:
    """Verify corners are already in TL/TR/BR/BL order; return them normalised.

    Checking the winding alone is **not sufficient** and this is the trap the plan
    warns about: CCPD's file order (RB, LB, LT, RT) has the *same* clockwise winding
    as TL, TR, BR, BL — it is a rotation of the same cycle, not a reversal. So the
    check is that the sequence equals its own canonical ordering, which pins the
    starting vertex as well as the direction.

    `strict=True` additionally requires the quad to be roughly upright
    (TL left of TR, TL above BL, and so on). That holds for any plate inside this
    project's viewpoint envelope but not for CCPD's Rotate/Tilt subsets, so it is
    opt-in.
    """
    canonical = order_corners(points)
    given = tuple(_as_points(points))
    if given != canonical:
        raise CornerOrderError(
            "corners are not in TL/TR/BR/BL order.\n"
            f"  given:     {given}\n"
            f"  canonical: {canonical}\n"
            f"  winding of given: {winding(given)}"
        )
    if strict:
        tl, tr, br, bl = canonical
        problems = []
        if not tl[0] < tr[0]:
            problems.append("TL.x < TR.x")
        if not bl[0] < br[0]:
            problems.append("BL.x < BR.x")
        if not tl[1] < bl[1]:
            problems.append("TL.y < BL.y")
        if not tr[1] < br[1]:
            problems.append("TR.y < BR.y")
        if problems:
            raise CornerOrderError(
                f"corners are not upright; violated: {', '.join(problems)} in {canonical}"
            )
    return canonical


# --------------------------------------------------------------------------- #
# Box
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned bounding box in absolute pixels, top-left origin."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise RecordValidationError(f"Box.{name} must be a finite number, got {v!r}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise RecordValidationError(
                f"Box must have positive extent, got {self.as_list()} "
                "(expected x1 < x2 and y1 < y2)"
            )

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, values: Sequence[float]) -> Box:
        if len(values) != 4:
            raise RecordValidationError(f"Box needs 4 values, got {list(values)!r}")
        return cls(*(float(v) for v in values))

    def iou(self, other: Box) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        return inter / (self.area + other.area - inter)


def corners_to_box(corners: Sequence[Point]) -> Box:
    """Tight axis-aligned box around four corners."""
    pts = _as_points(corners)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return Box(min(xs), min(ys), max(xs), max(ys))


# NOTE: there is deliberately no `box_to_corners`. Fabricating corners from a
# bounding box teaches the detector that every plate is axis-aligned. Box-only
# sources set `corners=None` and the keypoint loss is masked on those records
# (plan §7).


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Record:
    """One annotated image from one source. The Phase 1 → everything contract.

    `image_path` is relative to the data root and stored POSIX-style, so an index
    written on one machine reads on another.

    `corners` is either None (the source has no keypoint annotation — do not
    fabricate them) or exactly four points, **normalised to TL/TR/BR/BL here**.

    Normalising rather than validating is safe because `order_corners` is purely
    geometric: it reads the coordinates and ignores whatever convention the source
    published. So a Record that exists is guaranteed to carry canonical corners,
    unconditionally. Source modules still convert explicitly, with a test — that is
    what catches a convention changing upstream, which this cannot see.
    """

    source: str
    image_path: str
    license_key: str
    plate: str | None = None
    plate_type: PlateType = PlateType.UNKNOWN
    box: Box | None = None
    corners: Corners | None = None
    image_size: tuple[int, int] | None = None  # (width, height)
    attribution_url: str | None = None
    tags: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise RecordValidationError("Record.source must be a non-empty string")
        if not self.image_path:
            raise RecordValidationError("Record.image_path must be a non-empty string")
        if Path(self.image_path).is_absolute():
            raise RecordValidationError(
                f"Record.image_path must be relative to the data root, got {self.image_path!r}"
            )
        object.__setattr__(self, "image_path", Path(self.image_path).as_posix())

        lic = get_license(self.license_key)

        if not isinstance(self.plate_type, PlateType):
            object.__setattr__(self, "plate_type", PlateType(self.plate_type))

        if self.plate is not None:
            plate = self.plate.strip().upper()
            if not plate:
                raise RecordValidationError("Record.plate must be None or a non-empty string")
            object.__setattr__(self, "plate", plate)

        if self.corners is not None:
            object.__setattr__(self, "corners", order_corners(self.corners))

        if self.box is not None and not isinstance(self.box, Box):
            object.__setattr__(self, "box", Box.from_list(self.box))

        if self.image_size is not None:
            size = tuple(int(v) for v in self.image_size)
            if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
                raise RecordValidationError(
                    f"Record.image_size must be a positive (width, height), got {self.image_size!r}"
                )
            object.__setattr__(self, "image_size", size)

        if lic.attribution_per_image and not self.attribution_url:
            raise RecordValidationError(
                f"licence {lic.key!r} requires a per-image attribution URL, but "
                f"{self.image_path!r} has none. The backlink is a licence condition: an "
                "image that cannot be attributed must not enter the corpus."
            )

        object.__setattr__(self, "tags", tuple(self.tags))

    # -- derived ---------------------------------------------------------- #

    @property
    def license(self) -> License:
        return get_license(self.license_key)

    @property
    def commercial_ok(self) -> bool:
        return self.license.commercial_ok

    @property
    def has_corners(self) -> bool:
        return self.corners is not None

    def resolve(self, data_root: Path) -> Path:
        return Path(data_root) / self.image_path

    # -- serialization ---------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": self.source,
            "image_path": self.image_path,
            "license_key": self.license_key,
            "plate": self.plate,
            "plate_type": str(self.plate_type),
            "box": self.box.as_list() if self.box is not None else None,
            "corners": [list(p) for p in self.corners] if self.corners is not None else None,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "attribution_url": self.attribution_url,
            "tags": list(self.tags),
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Record:
        corners = d.get("corners")
        box = d.get("box")
        size = d.get("image_size")
        return cls(
            source=d["source"],
            image_path=d["image_path"],
            license_key=d["license_key"],
            plate=d.get("plate"),
            plate_type=PlateType(d.get("plate_type", PlateType.UNKNOWN)),
            box=Box.from_list(box) if box is not None else None,
            corners=order_corners(corners) if corners is not None else None,
            image_size=(int(size[0]), int(size[1])) if size is not None else None,
            attribution_url=d.get("attribution_url"),
            tags=tuple(d.get("tags", ())),
            extra=dict(d.get("extra", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Record:
        return cls.from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# JSON Lines I/O
# --------------------------------------------------------------------------- #


def write_jsonl(path: Path | str, records: Iterable[Record]) -> int:
    """Write records to `path` atomically. Returns the number written.

    Atomic because a half-written index that looks complete on the next run is
    exactly the kind of failure that costs an afternoon.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(rec.to_json())
                fh.write("\n")
                count += 1
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; an index is not a secret
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: Path | str) -> Iterator[Record]:
    """Stream records from a JSON Lines index. Blank lines are skipped."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Record.from_json(line)
            except (AcquisitionError, KeyError, ValueError) as exc:
                raise AcquisitionError(f"{path}:{lineno}: {exc}") from exc
