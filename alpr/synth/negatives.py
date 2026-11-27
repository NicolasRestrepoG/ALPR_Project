"""Negatives the recognizer must learn to reject.

Rendered through the full pipeline (in `corpus`, via `Synthesizer` styles):

    publico     white sheet, same outline and 3+3 layout: rejected by colour
    moto        LLLDDL with COLOMBIA in the strip, as photographed: rejected by the
                last character and the strip

Real crops, rectified the way the detector's output will be (this module):

    ccpd        a real Chinese plate from CCPD, rectified from its labelled corners
    nonplate    a plate-shaped patch of a vehicle front (grille, bumper, lamp),
                taken from the background corpus away from the painted-out plate
"""

from __future__ import annotations

import math

import numpy as np

from alpr.data_sources.storage import DataPaths
from alpr.plates.geometry import CANONICAL_SIZE
from alpr.plates.rectify import load_image, rectify
from alpr.synth.backgrounds import BackgroundStore


def ccpd_crop(
    paths: DataPaths,
    image_path: str,
    corners: list[list[float]],
    rng: np.random.Generator,
    jitter: float = 0.02,
) -> np.ndarray:
    corners = np.asarray(corners, float)
    width = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 2
    jittered = corners + rng.normal(0.0, jitter, (4, 2)) * width
    return rectify(load_image(paths.root / image_path), jittered, CANONICAL_SIZE)


#: Minimum pixel standard deviation for a non-plate patch to be worth learning from.
MIN_TEXTURE = 14.0


def nonplate_crop(store: BackgroundStore, index: int, rng: np.random.Generator) -> np.ndarray:
    """A plate-shaped quad that does not overlap the painted-out plate."""
    img, quad = store.get(index)
    h, w = img.shape[:2]
    q = np.asarray(quad, float)
    qx0, qy0 = q.min(axis=0)
    qx1, qy1 = q.max(axis=0)
    pw = qx1 - qx0
    fallback = None
    for _ in range(30):
        cw = pw * rng.uniform(0.6, 1.3)
        ch = cw / rng.uniform(1.7, 2.6)
        cx, cy = rng.uniform(cw / 2, w - cw / 2), rng.uniform(ch / 2, h - ch / 2)
        x0, y0, x1, y1 = cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2
        if x1 < qx0 or x0 > qx1 or y1 < qy0 or y0 > qy1:  # clear of the plate
            a = math.radians(rng.normal(0, 4))
            ca, sa = math.cos(a), math.sin(a)
            pts = np.array(
                [(-cw / 2, -ch / 2), (cw / 2, -ch / 2), (cw / 2, ch / 2), (-cw / 2, ch / 2)]
            )
            rot = pts @ np.array([[ca, sa], [-sa, ca]]) + (cx, cy)
            crop = rectify(img, rot, CANONICAL_SIZE)
            if crop.std() >= MIN_TEXTURE:  # flat sky/paint/night teaches nothing
                return crop
            fallback = crop
    if fallback is not None:
        return fallback
    # Crowded crop (rare): the top-left quarter, plate-shaped.
    pts = np.array([(0, 0), (w / 2, 0), (w / 2, w / 4), (0, w / 4)], float)
    return rectify(img, pts, CANONICAL_SIZE)
