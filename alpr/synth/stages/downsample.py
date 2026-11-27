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
    out = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    sx, sy = nw / w, nh / h
    return out, np.asarray(corners, float) * np.array([sx, sy])
