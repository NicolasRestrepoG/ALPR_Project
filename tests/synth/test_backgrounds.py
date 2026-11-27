"""The background corpus: plate painted out, quad kept, test set never used."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from alpr.data_sources.base import Box, PlateType, Record, write_jsonl
from alpr.data_sources.storage import DataPaths
from alpr.plates.labels import LabelError, LabelStore, labels_path
from alpr.synth import backgrounds as bgmod
from alpr.synth.backgrounds import CROP_SIZE, BackgroundStore, build, make_crop

QUAD = [(200.0, 300.0), (400.0, 300.0), (400.0, 400.0), (200.0, 400.0)]


def _car(color=(40, 90, 160)):
    img = np.full((600, 600, 3), color, np.uint8)
    cv2.rectangle(img, (200, 300), (399, 399), (250, 200, 20), -1)  # the plate
    cv2.putText(img, "ABC", (215, 375), cv2.FONT_HERSHEY_SIMPLEX, 2.4, (10, 10, 10), 8)
    return img


def test_make_crop_paints_the_plate_out_and_keeps_its_quad():
    crop, quad = make_crop(_car(), np.array(QUAD))
    assert crop.shape == (CROP_SIZE[1], CROP_SIZE[0], 3)
    w = np.linalg.norm(quad[1] - quad[0])
    assert w == pytest.approx(CROP_SIZE[0] / 3, rel=0.01)  # three plate widths
    x0, y0 = quad.min(axis=0).astype(int) + 8
    x1, y1 = quad.max(axis=0).astype(int) - 8
    inside = crop[y0:y1, x0:x1].astype(float)
    assert inside[..., 2].mean() > 100  # yellow and black are gone; the car's blue fills in
    assert inside.std(axis=(0, 1)).max() < 30  # smooth fill, no plate edges left


@pytest.fixture
def source(tmp_path, monkeypatch):
    paths = DataPaths(tmp_path).ensure()
    raw = paths.raw("rf")
    raw.mkdir(parents=True)
    names = ["a.png", "b.png", "t.png"]
    for n in names:
        cv2.imwrite(str(raw / n), _car())
    write_jsonl(
        paths.index("rf"),
        [
            Record(
                source="rf",
                image_path=f"raw/rf/{n}",
                license_key="mit",
                box=Box(200, 300, 400, 400),
            )
            for n in names
        ],
    )
    store = LabelStore(labels_path(paths, "rf"))
    for n in names:
        store.put(f"raw/rf/{n}", plate="ABC123", plate_type=PlateType.PARTICULAR, corners=QUAD)
    # Pretend "t.png" is in the frozen test set, and "rf" is registered for backgrounds.
    monkeypatch.setattr(bgmod, "_frozen_test_paths", lambda p: {"raw/rf/t.png"})

    class Info:
        purposes = {bgmod.Purpose.BACKGROUNDS}

    class Src:
        info = Info

    monkeypatch.setattr(bgmod.registry, "get", lambda name: Src)
    return paths


def test_build_excludes_the_test_set_and_refuses_to_overwrite(source):
    written, failed = build(source, ["rf"], workers=1)
    assert (written, failed) == (2, 0)
    index = [
        json.loads(line)
        for line in (source.root / "backgrounds" / "index.jsonl").read_text().splitlines()
    ]
    assert {e["image_path"] for e in index} == {"raw/rf/a.png", "raw/rf/b.png"}
    store = BackgroundStore(source)
    img, quad = store.get(0)
    assert img.shape == (CROP_SIZE[1], CROP_SIZE[0], 3) and quad.shape == (4, 2)
    with pytest.raises(LabelError, match="force"):
        build(source, ["rf"], workers=1)


def test_unregistered_source_is_refused(source, monkeypatch):
    class Info:
        purposes = set()

    class Src:
        info = Info

    monkeypatch.setattr(bgmod.registry, "get", lambda name: Src)
    with pytest.raises(LabelError, match="not registered"):
        build(source, ["rf"], workers=1)


def test_missing_corpus_is_explained(tmp_path):
    with pytest.raises(LabelError, match="generate backgrounds"):
        BackgroundStore(DataPaths(tmp_path))
