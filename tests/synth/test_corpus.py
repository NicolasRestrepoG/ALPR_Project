"""Corpus spec, hashing, balanced texts, negatives, and the build itself."""

from __future__ import annotations

import json
from collections import Counter

import cv2
import numpy as np
import pytest

from alpr.data_sources.storage import DataPaths
from alpr.plates.labels import CHARSET
from alpr.synth import corpus as C
from alpr.synth.corpus import CorpusSpec, balanced_text, build_corpus, corpus_hash, glyph_stats

from .conftest import _glyph


@pytest.fixture
def assets(tmp_path):
    paths = DataPaths(tmp_path).ensure()
    out = paths.assets_dir / "glyphs"
    out.mkdir(parents=True)
    entries = {}
    for c in CHARSET:
        img, origin = _glyph(c)
        cv2.imwrite(str(out / f"{c}.png"), img)
        entries[c] = {"file": f"{c}.png", "origin_mm": list(origin)}
    (out / "glyphs.json").write_text(json.dumps({"glyphs": entries}))
    (paths.assets_dir / "municipios.txt").write_text("MEDELLIN\nCALI\n")
    return paths


def test_spec_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        CorpusSpec(class_mix=(("particular", 0.5),))
    with pytest.raises(ValueError, match="unknown classes"):
        CorpusSpec(class_mix=(("truck", 1.0),))
    with pytest.raises(ValueError, match="order"):
        CorpusSpec(order="backwards")
    assert CorpusSpec().ranges.plate_px_follows_severity is False  # corpus policy


def test_hash_follows_spec_and_inputs(assets):
    a = corpus_hash(CorpusSpec(size=10), assets)
    assert a == corpus_hash(CorpusSpec(size=10), assets)
    assert a != corpus_hash(CorpusSpec(size=11), assets)
    assert a != corpus_hash(CorpusSpec(size=10, order="noise_before_blur"), assets)
    (assets.assets_dir / "municipios.txt").write_text("BOGOTA D.C.\n")
    assert a != corpus_hash(CorpusSpec(size=10), assets)  # inputs are part of it


def test_balanced_texts_are_exactly_uniform_per_block():
    texts = [balanced_text(3, i) for i in range(26 * 10)]
    for k in range(3):
        assert set(Counter(t[k] for t in texts).values()) == {10}
    for k in range(3, 6):
        assert set(Counter(t[k] for t in texts).values()) == {26}
    assert balanced_text(3, 5) == balanced_text(3, 5)
    moto = balanced_text(3, 5, moto=True)
    assert moto[:5] == balanced_text(3, 5)[:5] and moto[5].isalpha()


def test_choose_class_follows_the_mix():
    spec = CorpusSpec()
    rng = np.random.default_rng(0)
    counts = Counter(C.choose_class(spec, float(rng.random())) for _ in range(20000))
    assert counts["particular"] / 20000 == pytest.approx(0.88, abs=0.01)
    assert counts["ccpd"] / 20000 == pytest.approx(0.03, abs=0.005)


def test_glyph_stats():
    labels = [{"class": "particular", "text": balanced_text(0, i)} for i in range(260)]
    labels.append({"class": "publico", "text": "AAA111"})  # not counted
    counts, ls, ds = glyph_stats(labels)
    assert ls == 0.0 and ds == 0.0 and counts["A"] == 30


def test_build_is_deterministic_reused_and_worker_independent(assets):
    spec = CorpusSpec(size=12, seed=4)
    rep = build_corpus(assets, spec, workers=1)
    assert rep.size == 12 and (rep.path / "DONE").exists()
    labels = C.load_labels(rep.path)
    assert [lb["index"] for lb in labels] == list(range(12))
    for lb in labels:
        img = cv2.imread(str(rep.path / lb["file"]))
        assert img.shape == (96, 192, 3)
        assert lb["class"] in C.CLASSES and lb["valid"] == (lb["class"] == "particular")
        if lb["text"]:
            assert lb["text"] in lb["file"]
    first = {lb["file"]: cv2.imread(str(rep.path / lb["file"])) for lb in labels}

    again = build_corpus(assets, spec, workers=1)
    assert again.reused and again.hash == rep.hash

    rebuilt = build_corpus(assets, spec, workers=2, force=True)
    for name, img in first.items():
        assert np.array_equal(cv2.imread(str(rebuilt.path / name)), img), name


def test_missing_negative_sources_fall_back_to_publico(assets):
    spec = CorpusSpec(
        size=6, class_mix=(("ccpd", 0.5), ("nonplate", 0.5)), seed=1
    )  # neither CCPD nor backgrounds exist in this data root
    rep = build_corpus(assets, spec, workers=1)
    assert set(rep.class_counts) == {"publico"}


def test_ablation_orders_differ_in_pixels(assets):
    a = build_corpus(assets, CorpusSpec(size=3, seed=2), workers=1)
    b = build_corpus(assets, CorpusSpec(size=3, seed=2, order="jpeg_before_noise"), workers=1)
    assert a.hash != b.hash
    la, lb = C.load_labels(a.path), C.load_labels(b.path)
    assert [x["text"] for x in la] == [x["text"] for x in lb]  # same plates
    diffs = [
        np.abs(
            cv2.imread(str(a.path / x["file"])).astype(int)
            - cv2.imread(str(b.path / y["file"])).astype(int)
        ).mean()
        for x, y in zip(la, lb, strict=True)
    ]
    assert max(diffs) > 0
