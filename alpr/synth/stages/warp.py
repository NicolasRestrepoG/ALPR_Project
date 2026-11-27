"""Stage 1 — perspective, at full render resolution.

Warping the 4 px/mm render before any downsampling means the only resampling that
band-limits the image is stage 3's area filter, as in a camera. Warping a small
image resamples twice and staircases glyph edges in a way no lens produces.

The plate is a 330 x 160 mm rectangle rotated by yaw (about the vertical axis),
pitch (about the horizontal axis) and roll, `distance_mm` in front of a pinhole
camera whose focal length keeps a frontal plate at render scale. Corners are
returned TL/TR/BR/BL in continuous pixel coordinates (edges on integers).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

Corners = np.ndarray  # (4, 2) float64, TL/TR/BR/BL

#: Context kept around the plate, as a fraction of its projected width.
MARGIN = 0.25


def _rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def project_corners(
    plate_size: tuple[int, int],
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    distance_mm: float,
    px_per_mm: float,
) -> np.ndarray:
    """Image positions of the plate's corners (relative to the projected centre)."""
    w_px, h_px = plate_size
    w_mm, h_mm = w_px / px_per_mm, h_px / px_per_mm
    obj = np.array(
        [
            [-w_mm / 2, -h_mm / 2, 0],
            [w_mm / 2, -h_mm / 2, 0],
            [w_mm / 2, h_mm / 2, 0],
            [-w_mm / 2, h_mm / 2, 0],
        ]
    )
    cam = obj @ _rotation(yaw_deg, pitch_deg, roll_deg).T + np.array([0.0, 0.0, distance_mm])
    if np.any(cam[:, 2] <= 1.0):
        raise ValueError("plate behind the camera")
    f = distance_mm * px_per_mm  # a frontal plate stays at render scale
    return f * cam[:, :2] / cam[:, 2:3]


def warp(
    plate: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    distance_mm: float,
    px_per_mm: float,
    margin: float = MARGIN,
) -> tuple[np.ndarray, np.ndarray, Corners]:
    """Warp the rendered plate into a canvas with `margin` context around it.

    Returns (warped uint8 RGB, alpha float32 in [0, 1], corners in the canvas)."""
    h, w = plate.shape[:2]
    pts = project_corners((w, h), yaw_deg, pitch_deg, roll_deg, distance_mm, px_per_mm)
    span = pts.max(axis=0) - pts.min(axis=0)
    pad = margin * span[0]
    origin = pts.min(axis=0) - pad
    corners = pts - origin
    size = (math.ceil(span[0] + 2 * pad), math.ceil(span[1] + 2 * pad))
    src = np.array([(0, 0), (w, 0), (w, h), (0, h)], np.float32) - 0.5
    m = cv2.getPerspectiveTransform(src, (corners - 0.5).astype(np.float32))
    warped = cv2.warpPerspective(plate, m, size, flags=cv2.INTER_LINEAR, borderValue=0)
    ones = np.ones((h, w), np.float32)
    alpha = cv2.warpPerspective(ones, m, size, flags=cv2.INTER_LINEAR, borderValue=0)
    return warped, alpha, corners
