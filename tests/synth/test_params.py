"""Parameter sampling: severity scales magnitudes, texts are uniform, all is reproducible."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from alpr.synth.params import DIGITS, LETTERS, random_text, sample_params

MUNIS = ["MEDELLIN", "CALI"]


def test_severity_zero_is_pristine():
    p = sample_params(np.random.default_rng(0), 0.0, municipalities=MUNIS)
    assert (p.yaw_deg, p.pitch_deg, p.roll_deg) == (0, 0, 0)
    assert p.defocus_px == 0 and p.motion_px == 0 and p.read_sigma == 0
    assert p.plate_px == pytest.approx(260) and p.jpeg_quality == 95
    assert not (p.bolts or p.frame or p.dirt or p.grille or p.bloom or p.ir or p.night)
    assert p.relief_smear_mm == 0 and all(j == (0.0, 0.0) for j in p.corner_jitter)


def test_severity_widens_every_magnitude():
    def stats(s):
        ps = [sample_params(np.random.default_rng(k), s, municipalities=MUNIS) for k in range(200)]
        return {
            "yaw": np.mean([abs(p.yaw_deg) for p in ps]),
            "blur": np.mean([p.defocus_px + p.motion_px for p in ps]),
            "px": np.mean([p.plate_px for p in ps]),
            "q": np.mean([p.jpeg_quality for p in ps]),
            "occl": np.mean([bool(p.bolts or p.frame or p.dirt) for p in ps]),
        }

    lo, hi = stats(0.2), stats(1.0)
    assert hi["yaw"] > lo["yaw"] and hi["blur"] > lo["blur"]
    assert hi["px"] < lo["px"] and hi["q"] < lo["q"]
    assert hi["occl"] > 0.4  # probabilities are the plan's at any severity > 0


def test_same_seed_same_params():
    a = sample_params(np.random.default_rng(7), 0.6, municipalities=MUNIS)
    b = sample_params(np.random.default_rng(7), 0.6, municipalities=MUNIS)
    assert a == b and a.to_dict()["text"] == b.text


def test_texts_are_uniform_letters_then_digits():
    rng = np.random.default_rng(0)
    texts = [random_text(rng) for _ in range(5000)]
    assert all(t[:3].isalpha() and t[3:].isdigit() for t in texts)
    letters = Counter(c for t in texts for c in t[:3])
    digits = Counter(c for t in texts for c in t[3:])
    assert set(letters) == set(LETTERS) and set(digits) == set(DIGITS)
    assert max(letters.values()) / min(letters.values()) < 1.35


def test_plate_colour_is_amber_and_varies():
    cols = {
        sample_params(np.random.default_rng(k), 0.0, municipalities=MUNIS).plate_rgb
        for k in range(50)
    }
    assert len(cols) > 20
    for r, g, b in cols:
        assert r > g > b  # yellow-to-amber


def test_bad_severity():
    with pytest.raises(ValueError):
        sample_params(np.random.default_rng(0), 1.5, municipalities=MUNIS)
