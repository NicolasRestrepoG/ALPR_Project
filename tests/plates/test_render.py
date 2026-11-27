"""The renderer draws MT-001 where geometry says, lights it, and stays fast."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest

from alpr.data_sources.storage import DataPaths
from alpr.plates.annotate import main, measure_layout
from alpr.plates.geometry import MT001
from alpr.plates.glyphs import PX, find_characters
from alpr.plates.labels import CHARSET
from alpr.plates.render import (
    PUBLICO,
    RIM_MM,
    Lighting,
    PlateRenderer,
    PlateSpec,
    RenderError,
    landmarks_from_layout,
    rendered_landmarks,
)


def _glyph(ch: str) -> tuple[np.ndarray, tuple[float, float]]:
    """A photographed-style glyph: ink 42 x 85 mm (narrow ones narrower), soft edge,
    2 mm margin, origin relative to (slot centre, characters' top line)."""
    w_mm = 14.0 if ch in "1IJ" else 42.0
    h_mm = 85.0
    canvas = np.zeros((420, 300), np.uint8)
    cv2.putText(canvas, ch, (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 12, 255, 34, cv2.LINE_AA)
    ys, xs = np.nonzero(canvas > 127)
    tight = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    ink = cv2.resize(tight, (round(w_mm * PX), round(h_mm * PX)), interpolation=cv2.INTER_AREA)
    m = round(2 * PX)
    img = cv2.copyMakeBorder(ink, m, m, m, m, cv2.BORDER_CONSTANT, value=0)
    return cv2.GaussianBlur(img, (5, 5), 1.5), (-(w_mm / 2 + 2), -2.0)


def _logo() -> tuple[np.ndarray, tuple[float, float]]:
    img = np.zeros((76, 76), np.uint8)
    cv2.circle(img, (38, 38), 30, 255, 6)
    return img, (-9.5, -9.5)


@pytest.fixture(scope="module")
def renderer():
    glyphs, origins = {}, {}
    for c in CHARSET:
        glyphs[c], origins[c] = _glyph(c)
    glyphs["logo"], origins["logo"] = _logo()
    return PlateRenderer(glyphs, origins)


def test_render_shape_and_determinism(renderer):
    a = renderer.render(PlateSpec("ABC123", "MEDELLIN"))
    b = renderer.render(PlateSpec("ABC123", "MEDELLIN"))
    assert a.shape == (640, 1320, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_glyphs_are_scaled_to_the_cell_with_one_factor_pair(renderer):
    sx, sy = renderer.scale
    assert sx == pytest.approx(MT001.char_w / 42.0, abs=0.01)
    assert sy == pytest.approx(MT001.char_h / 85.0, abs=0.01)


def test_characters_land_in_their_cells(renderer):
    img = renderer.render(PlateSpec("ABC123"))
    chars = find_characters(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    assert chars is not None
    for s, box in zip(chars, MT001.char_boxes(), strict=True):
        assert s[1] / PX == pytest.approx(box.y, abs=1.0)  # top line
        assert (s[1] + s[3]) / PX == pytest.approx(box.bottom, abs=1.0)
        if s[2] > 30 * PX:  # full-width glyphs: centred in the cell
            assert (s[0] + s[2] / 2) / PX == pytest.approx(box.x + box.w / 2, abs=1.0)


def test_rendered_landmarks_match_geometry_when_frontal(renderer):
    lm = rendered_landmarks(renderer, ["ABC123", "KDQ742", "XYZ908"])
    boxes = MT001.char_boxes()
    assert lm["char_top"] == pytest.approx(MT001.char_top, abs=1.0)
    assert lm["char_bottom"] == pytest.approx(MT001.char_top + MT001.char_h, abs=1.0)
    for k, box in enumerate(boxes, start=1):
        assert lm[f"slot{k}"] == pytest.approx(box.x + box.w / 2, abs=1.0)
    assert lm["muni_top"] == pytest.approx(MT001.muni_band.y, abs=1.5)
    assert lm["muni_bottom"] == pytest.approx(MT001.muni_band.bottom, abs=1.5)


def test_a_flat_patch_keeps_its_base_colour(renderer):
    img = renderer.render(PlateSpec("ABC123"))
    x, y = round(165 * PX), round(20 * PX)  # top margin, between the slots
    assert tuple(img[y, x]) == PARTICULAR_BG
    pub = renderer.render(PlateSpec("ABC123", style=PUBLICO))
    assert tuple(pub[y, x]) == PUBLICO.background


PARTICULAR_BG = PlateSpec("ABC123").style.background


def test_light_direction_decides_which_bevels_shine(renderer):
    above = renderer.render(PlateSpec("HHH111", lighting=Lighting(direction=(0, -0.6, 0.8))))
    below = renderer.render(PlateSpec("HHH111", lighting=Lighting(direction=(0, 0.6, 0.8))))
    # Just below a character's bottom edge: shadowed when lit from above.
    box = MT001.char_boxes()[0]
    y = round((box.bottom + 1.0) * PX)
    xs = slice(round(box.x * PX), round(box.right * PX))
    assert above[y, xs].mean() < below[y, xs].mean() - 5


def test_parallax_moves_the_bottom_edge_not_the_top(renderer):
    texts = ["ABC123", "KDQ742"]
    frontal = rendered_landmarks(renderer, texts)
    tilted = rendered_landmarks(renderer, texts, (0.0, 2.0))
    assert tilted["char_top"] == pytest.approx(frontal["char_top"], abs=0.5)
    assert tilted["char_bottom"] - frontal["char_bottom"] == pytest.approx(1.5 * 2.0, abs=1.0)


def test_unpainted_walls_show_less_ink(renderer):
    painted = renderer.render(PlateSpec("ABC123", view=(0, 2.0)))
    bare = renderer.render(PlateSpec("ABC123", view=(0, 2.0), walls_painted=False))
    dark = lambda im: (cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) < 80).sum()  # noqa: E731
    assert dark(bare) < dark(painted)


def test_long_municipality_is_condensed_into_the_band(renderer):
    img = renderer.render(PlateSpec("ABC123", "SAN JUAN DE RIO SECO, CUNDINAMARCA"))
    band = MT001.muni_band
    strip = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[
        round((band.y + 3) * PX) : round((band.bottom - 3) * PX)
    ]
    rim = round((RIM_MM + 1) * PX)  # the border rim is ink too; look inside it
    cols = np.nonzero((strip[:, rim:-rim] < 80).any(axis=0))[0] + rim
    assert cols.min() >= band.x * PX - 2 and cols.max() <= band.right * PX + 2


@pytest.mark.parametrize(
    "spec, match",
    [
        (PlateSpec("ABC12"), "six characters"),
        (PlateSpec("AB$123"), "no glyph"),
        (PlateSpec("ABC123", "MEDELLÍN"), "letterform"),
        (PlateSpec("ABC123", lighting=Lighting(direction=(0, 0, -1))), "in front"),
    ],
)
def test_bad_specs_are_refused(renderer, spec, match):
    with pytest.raises(RenderError, match=match):
        renderer.render(spec)


def test_missing_glyphs_are_refused():
    with pytest.raises(RenderError, match="missing"):
        PlateRenderer({"A": _glyph("A")[0]}, {"A": (0.0, 0.0)})


def test_render_is_fast(renderer):
    spec = PlateSpec("ABC123", view=(0, 3.0))
    renderer.render(spec)
    t = time.perf_counter()
    for _ in range(10):
        renderer.render(spec)
    ms = (time.perf_counter() - t) / 10 * 1000
    assert ms < 60, ms  # the gate is 15 ms; this only catches a regression on slow CI


def test_landmarks_from_layout_matches_a_measured_render(renderer):
    img = renderer.render(PlateSpec("ABC123"))
    w, h = renderer.size
    layout, skipped = measure_layout([(img, [(0, 0), (w, 0), (w, h), (0, h)])])
    assert skipped == 0
    lm = landmarks_from_layout(layout)
    assert set(lm) >= {"char_top", "char_bottom", "slot1", "slot6", "muni_top"}


def test_cli_render(tmp_path, monkeypatch, capsys):
    paths = DataPaths(tmp_path).ensure()
    out = paths.assets_dir / "glyphs"
    out.mkdir(parents=True)
    entries = {}
    for c in [*CHARSET, "logo"]:
        img, origin = _logo() if c == "logo" else _glyph(c)
        cv2.imwrite(str(out / f"{c}.png"), img)
        entries[c] = {"file": f"{c}.png", "origin_mm": list(origin)}
    (out / "glyphs.json").write_text(json.dumps({"glyphs": entries}))
    monkeypatch.setenv("ALPR_DATA_ROOT", str(tmp_path))
    target = tmp_path / "p.png"
    assert main(["render", "abc123", "--municipality", "CALI", "--out", str(target)]) == 0
    assert "1320 x 640" in capsys.readouterr().out
    assert cv2.imread(str(target)).shape == (640, 1320, 3)
