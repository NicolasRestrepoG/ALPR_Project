"""Stage 3 — downsample to the sampled plate width: the single band-limit.

Area (box) interpolation stands in for the pixel aperture. Lanczos rings, and the
ringing is a synthetic signature; repeated resizes compound the filtering.
"""

from __future__ import annotations

import cv2
import numpy as np


def plate_width(corners: np.ndarray) -> float:
    c = np.asarray(corners, float)
    return float((np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2)


def downsample(
    image: np.ndarray, corners: np.ndarray, target_plate_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink so the plate's width is `target_plate_px`. Never enlarges.

    Returns the image and the corners in its coordinates (continuous, edges on
    integers, so scaling is exact)."""
    h, w = image.shape[:2]
    factor = max(1.0, plate_width(corners) / target_plate_px)
    nw, nh = max(1, round(w / factor)), max(1, round(h / factor))
    # Area filtering by a non-integer factor is slow in OpenCV; an exact integer box
    # reduction first, then area for the remaining < 2x, is a box filter still and
    # ~10x faster on render-size canvases.
    k = int(factor // 2) * 2 if factor >= 4 else 1
    src = image
    if k > 1:
        cw, ch = (w // k) * k, (h // k) * k
        src = cv2.resize(image[:ch, :cw], (cw // k, ch // k), interpolation=cv2.INTER_AREA)
    out = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
    # Corners: /k for the integer step (the trimmed right/bottom strip is margin),
    # then the remaining ratio.
    sx, sy = nw / src.shape[1] / k, nh / src.shape[0] / k
    return out, np.asarray(corners, float) * np.array([sx, sy])


def plate_mask(shape: tuple[int, int], corners: np.ndarray) -> np.ndarray:
    """The plate's coverage at this resolution, anti-aliased, from its corners."""
    mask = np.zeros(shape, np.float32)
    pts = np.round((np.asarray(corners, float) - 0.5) * 16).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 1.0, lineType=cv2.LINE_AA, shift=4)
    return mask
