"""Four corners → a flat, upright plate on the canonical 192 x 96 RGB canvas.

This is half of the Phase 2 contract (plan §7): `rectify(image, corners)`.

**Rectify the full plate outline, not the character band.** The municipality strip
and the MinTransporte logo are evidence for the validity head: a white público
plate has the same outline and the same 3+3 layout, and colour, logo and strip are
what separate them.

Corner order is resolved by `alpr.data_sources.base.order_corners` — the same helper
the annotation tool and every source module use. One function, one convention,
tested by winding direction. A second implementation here is exactly how a
rectifier ends up emitting upside-down plates with no error anywhere.

Pixel convention: corner coordinates are continuous, with pixel (i, j) covering
[i, i+1) x [j, j+1) — what a click on an image reports. The plate's outer corners
land exactly on the output's outer edges.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from alpr.data_sources.base import CornerOrderError, Corners, order_corners
from alpr.plates.geometry import CANONICAL_SIZE

#: Shortest plate edge, in source pixels, that can still be rectified meaningfully.
MIN_SIDE_PX = 8.0

#: Interior angles outside this range mean a mis-click or a degenerate quad, not a
#: plate seen from an awkward angle. The project's viewpoint envelope (yaw ±25°,
#: roll ±8°, camera above the bumper) stays well inside it.
MIN_ANGLE_DEG = 25.0
MAX_ANGLE_DEG = 155.0

#: Width / height of the quad, averaged over opposite sides. A real plate is 2.06;
#: foreshortening moves it, but never this far.
MIN_ASPECT = 0.8
MAX_ASPECT = 8.0

#: Opposite sides differing by more than this is perspective a homography cannot
#: invert usefully — the far side would be smeared across most of the canvas.
MAX_OPPOSITE_RATIO = 4.0

#: Cap on supersampling when the source plate is much larger than the canvas.
MAX_SUPERSAMPLE = 8


class RectifyError(ValueError):
    """Corners that would produce a garbage homography, or an unreadable image."""


def load_image(path: Path | str) -> np.ndarray:
    """Read an image as RGB uint8, honouring EXIF orientation.

    Phone photos store rotation in EXIF. OpenCV applies it on read, and every tool
    in this package reads through this function, so corners clicked in the
    annotation tool and corners consumed by `rectify` live in the same frame.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RectifyError(f"could not decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(prev: Sequence[float], at: Sequence[float], nxt: Sequence[float]) -> float:
    v1 = (prev[0] - at[0], prev[1] - at[1])
    v2 = (nxt[0] - at[0], nxt[1] - at[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def check_quad(corners: Sequence[Any], *, min_side: float = MIN_SIDE_PX) -> Corners:
    """Order four corners TL/TR/BR/BL and refuse shapes that cannot be a plate.

    Returns the ordered corners. Raises `RectifyError` naming the problem: the
    guard exists so a mis-click or a collapsed detector output fails loudly here
    instead of producing a smeared crop that trains the recognizer on noise.
    """
    try:
        quad = order_corners(corners)
    except CornerOrderError as exc:
        raise RectifyError(str(exc)) from exc

    tl, tr, br, bl = quad
    sides = {
        "top": _dist(tl, tr),
        "right": _dist(tr, br),
        "bottom": _dist(br, bl),
        "left": _dist(bl, tl),
    }
    short = {k: round(v, 1) for k, v in sides.items() if v < min_side}
    if short:
        raise RectifyError(f"plate edges shorter than {min_side} px: {short}")

    # Convexity: every turn the same way. order_corners already made the cycle
    # clockwise on screen, so all cross products must be positive.
    for i in range(4):
        a, b, c = quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross <= 0:
            raise RectifyError(f"corners do not form a convex quadrilateral: {quad}")

    for i, name in enumerate(("TL", "TR", "BR", "BL")):
        angle = _angle_deg(quad[i - 1], quad[i], quad[(i + 1) % 4])
        if not MIN_ANGLE_DEG <= angle <= MAX_ANGLE_DEG:
            raise RectifyError(
                f"interior angle at {name} is {angle:.0f}°; expected "
                f"{MIN_ANGLE_DEG:.0f}–{MAX_ANGLE_DEG:.0f}° for a plate"
            )

    width = (sides["top"] + sides["bottom"]) / 2.0
    height = (sides["left"] + sides["right"]) / 2.0
    aspect = width / height
    if not MIN_ASPECT <= aspect <= MAX_ASPECT:
        raise RectifyError(
            f"quad aspect {aspect:.2f} is outside {MIN_ASPECT}–{MAX_ASPECT}; "
            "a plate is 2.06 before foreshortening"
        )

    for a, b in (("top", "bottom"), ("left", "right")):
        ratio = max(sides[a], sides[b]) / min(sides[a], sides[b])
        if ratio > MAX_OPPOSITE_RATIO:
            raise RectifyError(
                f"{a} and {b} edges differ {ratio:.1f}x; too much perspective to rectify"
            )
    return quad


def homography(corners: Sequence[Any], size: tuple[int, int] = CANONICAL_SIZE) -> np.ndarray:
    """3x3 matrix taking source pixels to the canvas, in OpenCV's pixel-centre frame."""
    quad = check_quad(corners)
    w, h = size
    # Continuous (edge) coordinates → OpenCV centre coordinates: shift both by -0.5.
    src = np.array(quad, dtype=np.float32) - 0.5
    dst = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32) - 0.5
    return cv2.getPerspectiveTransform(src, dst)


def jitter_corners(
    corners: Sequence[Any], sigma: float, rng: np.random.Generator | None = None
) -> Corners:
    """Perturb each corner, simulating an imperfect detector.

    `sigma` is a fraction of the plate's mean width, so the same setting means the
    same thing for a 60 px plate and a 1,400 px one. Offsets are Gaussian, clipped
    at 3 sigma so one unlucky draw cannot fold the quad. The result is re-checked;
    if it is degenerate, the original corners come back unchanged rather than a
    crash in the middle of corpus generation.
    """
    quad = check_quad(corners)
    if sigma <= 0:
        return quad
    rng = rng if rng is not None else np.random.default_rng()
    tl, tr, br, bl = quad
    scale = sigma * (_dist(tl, tr) + _dist(bl, br)) / 2.0
    offsets = np.clip(rng.normal(0.0, scale, size=(4, 2)), -3 * scale, 3 * scale)
    moved = [
        (p[0] + float(dx), p[1] + float(dy)) for p, (dx, dy) in zip(quad, offsets, strict=True)
    ]
    try:
        return check_quad(moved)
    except RectifyError:
        return quad


def rectify(
    image: np.ndarray,
    corners: Sequence[Any],
    size: tuple[int, int] = CANONICAL_SIZE,
    *,
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Warp the plate inside `corners` to a flat `size` = (width, height) canvas.

    Returns an array of shape (height, width, channels) with the input's dtype.
    Corners may arrive in any order; they are normalised to TL/TR/BR/BL first.

    A large source plate is warped to a supersampled canvas and then area-averaged
    down, rather than point-sampled straight to 192 x 96 — otherwise a 1,400 px
    plate aliases into a crop that looks nothing like a 150 px one, and real crops
    stop resembling synthetic ones for a reason that has nothing to do with the
    generator.
    """
    if image.ndim not in (2, 3) or image.size == 0:
        raise RectifyError(f"expected an HxW or HxWxC image, got shape {image.shape}")
    quad = jitter_corners(corners, jitter, rng) if jitter > 0 else check_quad(corners)

    w, h = size
    tl, tr, br, bl = quad
    src_w = max(_dist(tl, tr), _dist(bl, br))
    src_h = max(_dist(tl, bl), _dist(tr, br))
    factor = int(min(MAX_SUPERSAMPLE, max(1, math.floor(min(src_w / w, src_h / h)))))

    big = (w * factor, h * factor)
    matrix = homography(quad, big)
    warped = cv2.warpPerspective(
        image, matrix, big, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    if factor > 1:
        warped = cv2.resize(warped, size, interpolation=cv2.INTER_AREA)
    return warped
