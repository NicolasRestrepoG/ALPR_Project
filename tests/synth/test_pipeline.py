"""The composed pipeline: stage order, determinism, severity, and what comes out."""

from __future__ import annotations

import numpy as np
import pytest

import alpr.synth.pipeline as pipeline
from alpr.plates.rectify import rectify
from alpr.synth.pipeline import Synthesizer
from alpr.synth.stages import ORDER


@pytest.fixture
def syn(renderer):
    return Synthesizer(renderer, None, ["MEDELLIN", "SAN JUAN DE RIO SECO"])


def test_sample_shapes(syn):
    s = syn.sample(np.random.default_rng(0), 0.7)
    assert s.crop.shape == (96, 192, 3) and s.crop.dtype == np.uint8
    assert s.image.dtype == np.uint8 and s.corners.shape == (4, 2)


def test_stages_run_in_the_physical_order(syn, monkeypatch):
    calls = []
    modules = {
        "warp": (pipeline._warp, "warp"),
        "composite": (pipeline._composite, "composite"),
        "downsample": (pipeline._downsample, "downsample"),
        "photometric": (pipeline._photometric, "photometric"),
        "blur": (pipeline._blur, "blur"),
        "noise": (pipeline._noise, "noise"),
        "jpeg": (pipeline._jpeg, "jpeg"),
    }
    for name, (mod, fn) in modules.items():
        original = getattr(mod, fn)

        def wrapped(*a, _o=original, _n=name, **k):
            calls.append(_n)
            return _o(*a, **k)

        monkeypatch.setattr(mod, fn, wrapped)
    syn.sample(np.random.default_rng(0), 0.5)
    assert tuple(calls) == ORDER


def test_same_params_same_pixels(syn):
    p = syn.params(np.random.default_rng(3), 0.8)
    assert np.array_equal(syn.run(p).crop, syn.run(p).crop)


def test_severity_degrades_on_average(syn, renderer):
    """Crops drift away from the clean render as severity rises."""
    from alpr.plates.render import PlateSpec

    def error(sev):
        errs = []
        for k in range(6):
            s = syn.sample(np.random.default_rng(k), sev, text="ABC123")
            clean = renderer.render(PlateSpec("ABC123", s.params.municipality))
            h, w = clean.shape[:2]
            ref = rectify(clean, [(0, 0), (w, 0), (w, h), (0, h)], (192, 96)).astype(float)
            g = lambda im: im @ [0.3, 0.6, 0.1]  # noqa: E731
            a, b = g(s.crop.astype(float)), g(ref)
            errs.append(1 - np.corrcoef(a.ravel(), b.ravel())[0, 1])
        return float(np.mean(errs))

    e0, e1 = error(0.0), error(1.0)
    assert e0 < 0.1 < e1 or e1 > 2 * e0


def test_severity_zero_crop_matches_the_clean_plate(syn, renderer):
    s = syn.sample(np.random.default_rng(0), 0.0, text="KDQ742")
    assert s.params.text == "KDQ742"
    g = s.crop.astype(float) @ [0.3, 0.6, 0.1]
    assert g[:, 10:60].min() < 80 and np.median(g) > 100  # dark glyphs on a light sheet
