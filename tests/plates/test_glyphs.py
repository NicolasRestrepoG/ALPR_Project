"""Glyph extraction recovers the characters drawn on synthetic plates."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from alpr.data_sources.base import Box, PlateType, Record, write_jsonl
from alpr.data_sources.storage import DataPaths
from alpr.plates.annotate import main
from alpr.plates.geometry import MT001
from alpr.plates.glyphs import (
    PX,
    combine,
    darkness,
    extract,
    find_characters,
    finish,
    glyphs_dir,
    keep_main_blob,
    load_glyphs,
    write,
)
from alpr.plates.labels import LabelError, LabelStore, labels_path

from .conftest import BLACK, SCENE_QUAD, YELLOW, place

SIZE = MT001.render_size(PX)


def _char_bitmap(ch: str, w: int, h: int) -> np.ndarray:
    """A thick character filling a w x h cell (255 = ink)."""
    canvas = np.zeros((420, 300), np.uint8)
    cv2.putText(canvas, ch, (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 12, 255, 34, cv2.LINE_AA)
    ys, xs = np.nonzero(canvas > 127)
    tight = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    return cv2.resize(tight, (w, h), interpolation=cv2.INTER_AREA)


def draw_text_plate(text: str, bolt: bool = False) -> np.ndarray:
    img = np.full((SIZE[1], SIZE[0], 3), YELLOW, np.uint8)
    for rect, ch in zip(MT001.char_boxes(), text, strict=True):
        r = MT001.to_px(rect, SIZE)
        x, y, w, h = round(r.x), round(r.y), round(r.w), round(r.h)
        ink = _char_bitmap(ch, w, h) > 127
        img[y : y + h, x : x + w][ink] = BLACK
    if bolt:  # a bolt head near the first character, not touching it
        r = MT001.to_px(MT001.char_boxes()[0], SIZE)
        cv2.circle(img, (round(r.x) - 14, round(r.y) + 30), 14, BLACK, -1)
    return img


@pytest.fixture
def glyph_source(tmp_path):
    """Three cars tagged glyph_source; car ABC123 photographed twice (once smaller)."""
    paths = DataPaths(tmp_path).ensure()
    raw = paths.raw("rf")
    raw.mkdir(parents=True)
    small = [((x - 700) * 0.6 + 700, (y - 450) * 0.6 + 450) for x, y in SCENE_QUAD]
    shots = [
        ("a.png", "ABC123", SCENE_QUAD, True),
        ("a_far.png", "ABC123", small, False),
        ("b.png", "ABD124", SCENE_QUAD, False),
        ("c.png", "CBA321", SCENE_QUAD, False),
    ]
    rng = np.random.default_rng(0)
    for name, text, quad, bolt in shots:
        scene = place(draw_text_plate(text, bolt), quad).astype(np.int16)
        scene += rng.integers(-12, 13, scene.shape, dtype=np.int16)  # sensor noise
        cv2.imwrite(str(raw / name), cv2.cvtColor(np.clip(scene, 0, 255).astype(np.uint8), 4))
    write_jsonl(
        paths.index("rf"),
        [
            Record(
                source="rf", image_path=f"raw/rf/{n}", license_key="mit", box=Box(50, 50, 1300, 800)
            )
            for n, *_ in shots
        ],
    )
    store = LabelStore(labels_path(paths, "rf"))
    for name, text, quad, _ in shots:
        store.put(
            f"raw/rf/{name}",
            plate=text,
            plate_type=PlateType.PARTICULAR,
            corners=quad,
            tags=["glyph_source"],
        )
    return paths


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two ink images after cropping each to its ink and resizing to match."""

    def tight(img):
        ys, xs = np.nonzero(img >= 128)
        return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    a, b = tight(a), tight(b)
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = a >= 128, b >= 128
    return (a & b).sum() / (a | b).sum()


def test_find_characters_on_a_clean_plate():
    gray = cv2.cvtColor(draw_text_plate("ABC123"), cv2.COLOR_RGB2GRAY)
    chars = find_characters(gray)
    assert chars is not None and len(chars) == 6
    xs = [s[0] for s in chars]
    assert xs == sorted(xs)


def test_extract_recovers_each_drawn_character(glyph_source):
    glyphs, logo, skipped = extract(glyph_source, ["rf"])
    assert skipped == 0
    assert set(glyphs) == set("ABCD1234")
    r = MT001.to_px(MT001.char_boxes()[0], SIZE)
    for ch, g in glyphs.items():
        truth = _char_bitmap(ch, round(r.w), round(r.h))
        assert _iou(g.image, truth) > 0.85, ch


def test_one_instance_per_car_using_its_widest_photo(glyph_source):
    glyphs, _, _ = extract(glyph_source, ["rf"])
    a = glyphs["A"]
    plates = [i.plate for i in a.instances]
    assert sorted(plates) == ["ABC123", "ABD124", "CBA321"]  # four photos, three cars
    abc = next(i for i in a.instances if i.plate == "ABC123")
    assert abc.key == "raw/rf/a.png"  # the closer, wider of its two photos
    assert glyphs["B"].confidence == "ok" and glyphs["4"].confidence == "low"


def test_bolt_next_to_a_character_is_masked_out(glyph_source):
    glyphs, _, _ = extract(glyph_source, ["rf"])
    a = glyphs["A"].image
    # The bolt sits left of and above the A's ink; a clean crop has no ink there.
    ys, xs = np.nonzero(a >= 128)
    assert xs.min() >= round(1.5 * PX) and ys.min() >= round(1.5 * PX)


def test_keep_main_blob_and_combine_are_robust():
    dark = np.zeros((100, 60), np.float32)
    dark[20:80, 25:35] = 1.0  # the character
    dark[2:8, 2:8] = 1.0  # a speck at the edge
    kept = keep_main_blob(dark, (30, 50), px=1.0)
    assert kept[5, 5] == 0 and kept[50, 30] == 1
    shifted = np.roll(dark, 3, axis=1)

    class Inst:  # minimal stand-in for Instance
        def __init__(self, d):
            self.dark = d

    med = combine([Inst(dark), Inst(shifted), Inst(dark)])
    assert finish(med).shape[0] > 0
    assert finish(np.zeros((10, 10), np.float32)) is None
    assert darkness(np.full((10, 10), 200, np.uint8)).max() == 0


def test_write_and_load_round_trip(glyph_source):
    glyphs, logo, _ = extract(glyph_source, ["rf"])
    out = write(glyph_source, glyphs, logo, ["rf"])
    manifest = json.loads((out / "glyphs.json").read_text())
    assert manifest["px_per_mm"] == PX
    assert set(manifest["missing"]) == set("EFGHIJKLMNOPQRSTUVWXYZ056789")
    assert manifest["glyphs"]["B"]["cars"] == ["ABC123", "ABD124", "CBA321"]
    assert (out / "qa_sheet.png").exists()
    loaded = load_glyphs(glyph_source)
    assert np.array_equal(loaded["A"], glyphs["A"].image)


def test_no_glyph_sources_is_explained(tmp_path):
    paths = DataPaths(tmp_path).ensure()
    write_jsonl(paths.index("rf"), [])
    with pytest.raises(LabelError, match="glyph_source"):
        extract(paths, ["rf"])
    with pytest.raises(LabelError, match="annotate glyphs"):
        load_glyphs(paths)


def test_cli_glyphs(glyph_source, monkeypatch, capsys):
    monkeypatch.setenv("ALPR_DATA_ROOT", str(glyph_source.root))
    assert main(["glyphs", "rf"]) == 0
    out = capsys.readouterr().out
    assert "8 of 36 glyphs" in out and "missing:" in out
    assert (glyphs_dir(glyph_source) / "A.png").exists()
