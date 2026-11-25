"""Synthetic plates with known geometry, placed into larger scenes by a known warp."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from alpr.plates.geometry import MT001, PlateGeometry

YELLOW = (240, 200, 30)
BLACK = (15, 15, 15)

#: A perspective quad (TL, TR, BR, BL) inside a 1400 x 900 scene: yawed, pitched,
#: slightly rolled — the kind of view the gate camera produces.
SCENE_SIZE = (1400, 900)
SCENE_QUAD = ((212.0, 236.0), (1158.0, 190.0), (1190.0, 655.0), (190.0, 640.0))


def draw_plate(geometry: PlateGeometry = MT001, px_per_mm: float = 4.0) -> np.ndarray:
    """A flat plate: yellow ground, solid black character cells and municipality letters
    exactly where `geometry` says they are."""
    w, h = geometry.render_size(px_per_mm)
    img = np.full((h, w, 3), YELLOW, np.uint8)
    size = (w, h)
    for rect in geometry.char_boxes():
        r = geometry.to_px(rect, size)
        cv2.rectangle(
            img, (round(r.x), round(r.y)), (round(r.right) - 1, round(r.bottom) - 1), BLACK, -1
        )
    muni = geometry.to_px(geometry.muni_text_box("MEDELLIN"), size)
    step = (geometry.muni_char_w + geometry.muni_spacing) * px_per_mm
    for k in range(8):
        x = muni.x + k * step
        cv2.rectangle(
            img,
            (round(x), round(muni.y)),
            (round(x + geometry.muni_char_w * px_per_mm) - 1, round(muni.bottom) - 1),
            BLACK,
            -1,
        )
    return img


def quadrant_plate(w: int = 660, h: int = 320) -> np.ndarray:
    """Four colours, one per quadrant: any rotation or mirror is unmistakable."""
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2, : w // 2] = (255, 0, 0)  # TL red
    img[: h // 2, w // 2 :] = (0, 255, 0)  # TR green
    img[h // 2 :, w // 2 :] = (0, 0, 255)  # BR blue
    img[h // 2 :, : w // 2] = (255, 255, 255)  # BL white
    return img


def place(plate: np.ndarray, quad=SCENE_QUAD, scene_size=SCENE_SIZE) -> np.ndarray:
    """Warp a flat plate into a grey scene so its outer corners land on `quad`
    (continuous pixel coordinates, the same convention `rectify` consumes)."""
    h, w = plate.shape[:2]
    src = np.array([(0, 0), (w, 0), (w, h), (0, h)], np.float32) - 0.5
    dst = np.array(quad, np.float32) - 0.5
    m = cv2.getPerspectiveTransform(src, dst)
    scene = np.full((scene_size[1], scene_size[0], 3), 90, np.uint8)
    return cv2.warpPerspective(plate, m, scene_size, dst=scene, borderMode=cv2.BORDER_TRANSPARENT)


@pytest.fixture
def scene_and_quad():
    return place(quadrant_plate()), SCENE_QUAD
