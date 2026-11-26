"""MT-001 as an immutable spec: the layout sums to the outline, nothing overlaps."""

from __future__ import annotations

import dataclasses

import pytest

from alpr.plates.geometry import (
    CANONICAL_SIZE,
    MT001,
    PROVENANCE,
    GeometryError,
    PlateGeometry,
    Provenance,
    Rect,
)


def test_spec_values_as_transcribed():
    g = MT001
    assert (g.plate_w, g.plate_h) == (330.0, 160.0)
    assert (g.char_w, g.char_h, g.char_stroke, g.char_gap) == (40.0, 80.0, 10.0, 8.0)
    assert (g.muni_h, g.muni_char_w, g.muni_stroke, g.muni_spacing) == (23.0, 12.0, 5.0, 5.0)
    assert g.aspect == pytest.approx(2.0625)


def test_horizontal_layout_sums_to_plate_width():
    g = MT001
    total = 2 * g.side_margin + 6 * g.char_w + 4 * g.char_gap + g.logo_gutter
    assert total == pytest.approx(g.plate_w)


def test_vertical_layout_sums_to_plate_height():
    g = MT001
    total = g.char_top + g.char_h + g.muni_gap + g.muni_h + g.bottom_margin
    assert total == pytest.approx(g.plate_h)
    assert g.bottom_margin > 0


def test_six_cells_inside_the_plate_and_symmetric():
    g = MT001
    boxes = g.char_boxes()
    assert len(boxes) == 6
    assert all(g.outline.contains(b) for b in boxes)
    assert boxes[0].x == pytest.approx(g.plate_w - boxes[5].right)  # centred
    for a, b in zip(boxes, boxes[1:], strict=False):
        assert a.right <= b.x  # strictly left to right, no overlap


def test_character_cells_do_not_overlap_the_logo_gutter():
    g = MT001
    gutter = g.logo_gutter_box
    assert not any(b.overlaps(gutter) for b in g.char_boxes())
    assert g.char_boxes()[2].right == pytest.approx(gutter.x)
    assert g.char_boxes()[3].x == pytest.approx(gutter.right)


def test_gaps_between_cells():
    boxes = MT001.char_boxes()
    gaps = [b.x - a.right for a, b in zip(boxes, boxes[1:], strict=False)]
    assert gaps == pytest.approx([8.0, 8.0, MT001.logo_gutter, 8.0, 8.0])


def test_municipality_band_below_characters_inside_plate():
    g = MT001
    band = g.muni_band
    assert band.y == pytest.approx(g.char_band.bottom + g.muni_gap)
    assert g.outline.contains(band)
    assert not band.overlaps(g.char_band)


def test_municipality_text_is_centred_and_refuses_overflow():
    g = MT001
    box = g.muni_text_box("MEDELLIN")
    assert box.w == pytest.approx(8 * 12 + 7 * 5)
    assert box.center[0] == pytest.approx(g.plate_w / 2)
    with pytest.raises(GeometryError, match="abbreviate"):
        g.muni_text_box("SAN JUAN NEPOMUCENO DEL SUR")


def test_pixel_conversion_on_canonical_canvas_applies_the_three_percent_squeeze():
    g = MT001
    outline = g.to_px(g.outline, CANONICAL_SIZE)
    assert (outline.w, outline.h) == pytest.approx(CANONICAL_SIZE)
    sx, sy = CANONICAL_SIZE[0] / g.plate_w, CANONICAL_SIZE[1] / g.plate_h
    assert sy / sx == pytest.approx(2.0625 / 2.0)  # ≈ 1.031: the deliberate squeeze


def test_render_size_is_4px_per_mm():
    assert MT001.render_size() == (1320, 640)


def test_landmarks_include_plate_cells_logo_and_band():
    lm = MT001.landmarks()
    assert lm["plate_tl"] == (0.0, 0.0)
    assert lm["plate_br"] == (330.0, 160.0)
    assert {f"char{i}_tl" for i in range(1, 7)} <= lm.keys()
    assert lm["logo_center"][0] == pytest.approx(165.0)


def test_geometry_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        MT001.char_w = 41.0  # type: ignore[misc]


def test_layout_that_cannot_fit_raises():
    with pytest.raises(GeometryError):
        PlateGeometry(logo_gutter=80.0)
    with pytest.raises(GeometryError):
        PlateGeometry(char_top=70.0)
    with pytest.raises(GeometryError):
        PlateGeometry(char_w=-1.0)


def test_every_field_has_a_provenance():
    names = {f.name for f in dataclasses.fields(PlateGeometry)}
    assert names == set(PROVENANCE)


def test_non_spec_fields_are_exactly_the_measured_ones():
    by = {p: {k for k, v in PROVENANCE.items() if v is p} for p in Provenance}
    assert by[Provenance.TAPE] == {"char_top", "muni_h"}
    assert by[Provenance.PHOTO] == {"logo_gutter", "muni_gap"}


def test_layout_matches_the_real_plate():
    g = PlateGeometry()
    assert (g.char_top, g.muni_gap, g.muni_h) == (30.0, 15.0, 23.0)
    assert g.bottom_margin == pytest.approx(12.0)
    assert g.side_margin == pytest.approx(15.0)


def test_rect_helpers():
    r = Rect(1, 2, 3, 4)
    assert r.corners == ((1, 2), (4, 2), (4, 6), (1, 6))
    assert not r.overlaps(Rect(4, 2, 1, 1))  # touching is not overlapping
    assert r.overlaps(Rect(3.5, 5.5, 1, 1))
