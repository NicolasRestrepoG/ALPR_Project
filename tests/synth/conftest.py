"""A renderer from synthetic glyphs, so synth tests never need the extracted assets."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from alpr.plates.glyphs import PX
from alpr.plates.labels import CHARSET
from alpr.plates.render import PlateRenderer


def _glyph(ch: str) -> tuple[np.ndarray, tuple[float, float]]:
    w_mm = 14.0 if ch in "1IJ" else 42.0
    canvas = np.zeros((420, 300), np.uint8)
    cv2.putText(canvas, ch, (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 12, 255, 34, cv2.LINE_AA)
    ys, xs = np.nonzero(canvas > 127)
    tight = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    ink = cv2.resize(tight, (round(w_mm * PX), round(85 * PX)), interpolation=cv2.INTER_AREA)
    m = round(2 * PX)
    img = cv2.copyMakeBorder(ink, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    return img, (-(w_mm / 2 + 2), -2.0)


@pytest.fixture(scope="session")
def renderer() -> PlateRenderer:
    glyphs, origins = {}, {}
    for c in CHARSET:
        glyphs[c], origins[c] = _glyph(c)
    return PlateRenderer(glyphs, origins)


@pytest.fixture
def scene() -> np.ndarray:
    """A float32 scene in [0, 1] with structure at every scale."""
    rng = np.random.default_rng(0)
    img = cv2.resize(rng.random((12, 16, 3)).astype(np.float32), (320, 240))
    img[100:140, 60:260] = 0.9
    img[110:130, 80:120] = 0.05
    return img


SQUARE = np.array([(60.0, 100.0), (260.0, 100.0), (260.0, 140.0), (60.0, 140.0)])
