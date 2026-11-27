"""Front-view augmentations do what they say, where they say."""

from __future__ import annotations

import numpy as np
import pytest

from alpr.plates.render import SLOT_CENTRES_MM
from alpr.synth import augment

PX = 4.0


@pytest.fixture
def plate():
    img = np.full((640, 1320, 3), (240, 190, 30), np.uint8)
    img[120:440, 60:220] = (30, 30, 30)  # a character-like block
    return img


def test_bolts_sit_on_the_mounting_slots(plate):
    spec = [
        {"slot": k, "dx_mm": 0.0, "dy_mm": 0.0, "radius_mm": 6.0, "tone": "chrome"}
        for k in range(4)
    ]
    out = augment.bolts(plate, spec)
    for sx, sy in SLOT_CENTRES_MM:
        x, y = round(sx * PX), round(sy * PX)
        assert not np.array_equal(out[y - 6, x], plate[y - 6, x])  # covered
    assert np.array_equal(out[320, 660], plate[320, 660])  # elsewhere untouched


def test_frame_covers_the_edge_not_the_middle(plate):
    spec = {
        "thickness_mm": 8.0,
        "bottom_extra_mm": 10.0,
        "material": "black",
        "text": True,
        "radius_mm": 6,
    }
    out = augment.frame(plate, spec, np.random.default_rng(0))
    assert out[5, 660].mean() < 60 and out[630, 660].mean() < 120  # rim and bottom bar
    assert np.array_equal(out[320, 660], plate[320, 660])


@pytest.mark.parametrize("kind", ["grime", "spatter"])
def test_dirt_covers_roughly_its_coverage(plate, kind):
    spec = {
        "kind": kind,
        "coverage": 0.2,
        "scale_mm": 20.0 if kind == "grime" else 3.0,
        "opacity": 0.8,
        "tone": 0.5,
    }
    out = augment.dirt(plate, spec, np.random.default_rng(0))
    changed = (np.abs(out.astype(int) - plate.astype(int)).sum(axis=2) > 6).mean()
    assert 0.08 < changed < 0.45
    assert augment.dirt(plate, spec | {"coverage": 0.0}, np.random.default_rng(0)) is plate


def test_wear_fades_ink_only(plate):
    out = augment.wear(
        plate, {"amount": 0.7, "scale_mm": 10.0}, np.random.default_rng(0), (240, 190, 30)
    )
    assert out[120:440, 60:220].mean() > plate[120:440, 60:220].mean()  # ink lighter
    assert np.array_equal(out[20:100, 400:1200], plate[20:100, 400:1200])  # sheet untouched


def test_grille_bars_reuse_the_backgrounds_rows():
    img = np.full((300, 400, 3), 0.5, np.float32)
    bg = np.zeros((300, 400, 3), np.float32)
    bg[:150] = np.linspace(0, 1, 400, dtype=np.float32)[None, :, None]  # above the plate
    corners = np.array([(100.0, 150.0), (300.0, 150.0), (300.0, 250.0), (100.0, 250.0)])
    out = augment.grille(img, bg, corners, {"bars": 1, "thickness": 0.1, "positions": [0.5, 0.0]})
    row = out[200]
    assert row[10, 0] < row[390, 0]  # carries the background's left-to-right ramp
    assert np.array_equal(out[160], img[160])


def test_value_noise_range():
    n = augment.value_noise((50, 80), 10, np.random.default_rng(0))
    assert n.min() == pytest.approx(0) and n.max() == pytest.approx(1) and n.shape == (50, 80)
