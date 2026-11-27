"""Stage 2 — composite onto a background, still at render resolution.

Compositing before the downsample means the plate/vehicle boundary is band-limited
by the same filter as the plate interior. Composite after, and the plate gets a
pixel-sharp alpha edge: a free "plate is here" cue that no real photo has.

The background is a vehicle-front crop whose own plate was painted out
(`alpr.synth.backgrounds`); it is placed so that plate's position and size land
under the synthetic one, which keeps bumper, grille and scale plausible.
"""

from __future__ import annotations

import cv2
import numpy as np


def place_background(
    background: np.ndarray,
    bg_quad: np.ndarray,
    corners: np.ndarray,
    size: tuple[int, int],
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Scale and shift `background` so its plate quad sits under `corners`.

    `offset` nudges the placement by a fraction of the plate's width/height, so the
    synthetic plate is not always exactly where the painted-out one was."""
    bq, cq = np.asarray(bg_quad, float), np.asarray(corners, float)
    bw = np.linalg.norm(bq[1] - bq[0]) + np.linalg.norm(bq[2] - bq[3])
    cw = np.linalg.norm(cq[1] - cq[0]) + np.linalg.norm(cq[2] - cq[3])
    scale = cw / max(bw, 1e-6)
    bc, cc = bq.mean(axis=0), cq.mean(axis=0)
    plate_w, plate_h = cw / 2, cw / 4
    tx = cc[0] - scale * bc[0] + offset[0] * plate_w
    ty = cc[1] - scale * bc[1] + offset[1] * plate_h
    m = np.float32([[scale, 0, tx], [0, scale, ty]])
    return cv2.warpAffine(
        background, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def plain_background(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A smooth random field: the fallback when no background corpus is built."""
    w, h = size
    small = rng.integers(20, 200, (4, 6, 3)).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def composite(plate: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    """plate over background, uint8 in and out.

    uint8 is enough here: the downsample that follows averages many render pixels
    into each output pixel, and blending in uint8 (OpenCV's SIMD path) is what keeps
    full-resolution compositing affordable at corpus scale."""
    if plate.shape != background.shape or plate.shape[:2] != alpha.shape:
        raise ValueError(f"shapes differ: {plate.shape} {alpha.shape} {background.shape}")
    a = np.ascontiguousarray(alpha, np.float32)
    return cv2.blendLinear(plate, background, a, 1.0 - a)
