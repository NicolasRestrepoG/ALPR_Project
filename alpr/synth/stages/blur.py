"""Stage 5 — blur: motion and defocus, acting on the radiance field.

Before noise, always: blurring noise correlates neighbouring pixels, and real
sensor noise is independent per pixel.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    """A normalised line of `length` px at `angle_deg`, anti-aliased by supersampling."""
    if length < 1.0:
        return np.ones((1, 1), np.float32)
    r = math.ceil(length / 2) + 1
    size = 2 * r + 1
    ss = 8
    big = np.zeros((size * ss, size * ss), np.float32)
    a = math.radians(angle_deg)
    c = (size * ss) / 2
    dx, dy = math.cos(a) * length * ss / 2, math.sin(a) * length * ss / 2
    cv2.line(big, (round(c - dx), round(c - dy)), (round(c + dx), round(c + dy)), 1.0, ss)
    k = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    return k / k.sum()


def defocus_kernel(radius: float) -> np.ndarray:
    """A normalised, anti-aliased disk: the circle of confusion."""
    if radius < 0.3:
        return np.ones((1, 1), np.float32)
    r = math.ceil(radius) + 1
    ss = 8
    big = np.zeros(((2 * r + 1) * ss,) * 2, np.float32)
    c = big.shape[0] // 2
    cv2.circle(big, (c, c), round(radius * ss), 1.0, -1)
    k = cv2.resize(big, ((2 * r + 1),) * 2, interpolation=cv2.INTER_AREA)
    return k / k.sum()


def blur(
    image: np.ndarray,
    defocus_px: float = 0.0,
    motion_px: float = 0.0,
    motion_angle_deg: float = 0.0,
) -> np.ndarray:
    out = image
    for k in (motion_kernel(motion_px, motion_angle_deg), defocus_kernel(defocus_px)):
        if k.size > 1:
            out = cv2.filter2D(out, -1, k, borderType=cv2.BORDER_REFLECT_101)
    return out.astype(np.float32)
