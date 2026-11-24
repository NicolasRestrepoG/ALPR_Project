"""Roboflow YOLO parsing — especially that corners stay None."""

from __future__ import annotations

import pytest

from alpr.data_sources.base import PlateType
from alpr.data_sources.roboflow import (
    PlacasColombiaItmSource,
    PlacasColombianasSource,
    ProyectoPlacasSource,
    RoboflowError,
    horizontal_flip_enabled,
    parse_yolo_label,
    plate_type_for_class,
    read_class_names,
    yolo_to_box,
)
from alpr.data_sources.storage import DataPaths

from .factories import make_jpeg

# --------------------------------------------------------------------------- #
# Label parsing
# --------------------------------------------------------------------------- #


def test_parse_yolo_label():
    assert parse_yolo_label("0 0.5 0.5 0.25 0.1") == (0, 0.5, 0.5, 0.25, 0.1)


def test_extra_columns_are_ignored():
    """Some exports append a confidence column; the first five are the contract."""
    assert parse_yolo_label("2 0.5 0.5 0.2 0.2 0.98")[0] == 2


@pytest.mark.parametrize(
    "line, match",
    [
        ("0 0.5 0.5", "needs 5 fields"),
        ("x 0.5 0.5 0.2 0.2", "non-numeric"),
        ("-1 0.5 0.5 0.2 0.2", "negative class"),
        ("0 0.5 0.5 0.0 0.2", "degenerate"),
        ("0 640 360 100 40", "outside \\[0, 1\\]"),
    ],
)
def test_malformed_labels_raise(line, match):
    with pytest.raises(RoboflowError, match=match):
        parse_yolo_label(line)


def test_pixel_units_are_caught_not_silently_accepted():
    """An export already in pixels would otherwise produce a plausible tiny box."""
    with pytest.raises(RoboflowError, match="already be in pixel units"):
        parse_yolo_label("0 320 180 64 32")


def test_yolo_to_box_converts_to_pixels():
    box = yolo_to_box(0.5, 0.5, 0.25, 0.125, 640, 480)
    assert box.as_list() == [240.0, 210.0, 400.0, 270.0]
    assert box.width == 160.0
    assert box.height == 60.0


def test_box_is_clipped_to_the_image():
    box = yolo_to_box(0.02, 0.02, 0.2, 0.2, 100, 100)
    assert box.x1 == 0.0
    assert box.y1 == 0.0
    assert box.x2 == pytest.approx(12.0)


def test_box_that_collapses_after_clipping_raises():
    with pytest.raises(RoboflowError, match="collapsed after clipping"):
        yolo_to_box(1.5, 1.5, 0.1, 0.1, 100, 100)


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #


def test_read_class_names_inline(tmp_path):
    (tmp_path / "data.yaml").write_text("nc: 2\nnames: ['placa', 'moto']\n")
    assert read_class_names(tmp_path) == ["placa", "moto"]


def test_read_class_names_block(tmp_path):
    (tmp_path / "data.yaml").write_text("nc: 2\nnames:\n  - placa\n  - moto\ntrain: x\n")
    assert read_class_names(tmp_path) == ["placa", "moto"]


def test_read_class_names_missing_file(tmp_path):
    assert read_class_names(tmp_path) == []


@pytest.mark.parametrize(
    "class_name, expected",
    [
        ("moto", PlateType.MOTO),
        ("placa-moto", PlateType.MOTO),
        ("publico", PlateType.PUBLICO),
        ("taxi", PlateType.PUBLICO),
        ("placa", PlateType.UNKNOWN),
        ("license-plate", PlateType.UNKNOWN),
    ],
)
def test_plate_type_for_class(class_name, expected):
    assert plate_type_for_class(class_name) == expected


def test_a_detection_box_does_not_imply_particular():
    """Nothing about a bounding box says the plate is yellow. UNKNOWN is honest."""
    assert plate_type_for_class("placa") is PlateType.UNKNOWN


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def build_dataset(tmp_path, *, label_text="0 0.5 0.5 0.25 0.125"):
    paths = DataPaths(tmp_path).ensure()
    root = paths.raw("roboflow_placas_colombianas") / "dataset"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: ['placa']\n")
    make_jpeg(root / "train" / "images" / "img001.jpg", 640, 480)
    (root / "train" / "labels" / "img001.txt").write_text(label_text)
    return paths


def test_index_produces_box_only_records(tmp_path):
    paths = build_dataset(tmp_path)
    records = list(PlacasColombianasSource(paths).index())

    assert len(records) == 1
    rec = records[0]
    assert rec.box.as_list() == [240.0, 210.0, 400.0, 270.0]
    assert rec.image_size == (640, 480)
    assert rec.plate is None
    assert "train" in rec.tags
    assert rec.commercial_ok is True  # CC BY 4.0


def test_corners_are_never_fabricated_from_the_box(tmp_path):
    """Plan §7: fabricated corners teach the detector that plates are axis-aligned.

    Phase 6 masks the keypoint loss on exactly these records, which only works if
    they are distinguishable — corners must be None, not a rectangle.
    """
    paths = build_dataset(tmp_path)
    rec = next(iter(PlacasColombianasSource(paths).index()))
    assert rec.corners is None
    assert rec.has_corners is False


def test_multiple_boxes_per_image_yield_multiple_records(tmp_path):
    paths = build_dataset(tmp_path, label_text="0 0.3 0.5 0.1 0.1\n0 0.7 0.5 0.1 0.1\n")
    assert len(list(PlacasColombianasSource(paths).index())) == 2


def test_bad_label_lines_are_skipped_and_counted(tmp_path):
    paths = build_dataset(tmp_path, label_text="0 0.3 0.5 0.1 0.1\ngarbage line here\n")
    source = PlacasColombianasSource(paths)
    records = list(source.index())
    assert len(records) == 1
    assert source.last_index_skipped == 1
    assert source.last_index_errors


def test_image_without_a_label_file_is_skipped(tmp_path):
    paths = build_dataset(tmp_path)
    root = paths.raw("roboflow_placas_colombianas") / "dataset"
    make_jpeg(root / "train" / "images" / "unlabelled.jpg", 640, 480)
    source = PlacasColombianasSource(paths)
    assert len(list(source.index())) == 1
    assert source.last_index_skipped == 1


def test_missing_api_key_reports_instructions_instead_of_raising(tmp_path, monkeypatch):
    from alpr.data_sources.base import Status

    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    result = PlacasColombianasSource(DataPaths(tmp_path).ensure()).fetch()
    assert result.status is Status.UNAVAILABLE
    assert "ROBOFLOW_API_KEY" in result.message


def test_licences_verified_against_the_api_are_recorded():
    """Plan §8 listed two projects as "unverified — check". The check was done.

    The Roboflow API reports MIT for the ITM project and CC BY 4.0 for
    proyecto-placas (read 2026-08-09), so both moved off the `unknown` key and onto
    the commercially clean path. The date is in each source's notes so the claim is
    auditable rather than folklore.
    """
    assert PlacasColombiaItmSource.info.license_key == "mit"
    assert PlacasColombiaItmSource.info.commercial_ok is True
    assert "2026-08-09" in PlacasColombiaItmSource.info.notes

    assert ProyectoPlacasSource.info.license_key == "cc-by-4.0"
    assert ProyectoPlacasSource.info.commercial_ok is True
    assert "2026-08-09" in ProyectoPlacasSource.info.notes


def test_proyecto_placas_is_pinned_to_the_unaugmented_version():
    """v2 inflates 274 real images to 664 by shearing and blurring each three times.

    Phase 3 owns degradation. Baked-in augmentation upstream is both redundant and
    invisible in the record count, so the pin is deliberate, not incidental.
    """
    assert ProyectoPlacasSource.project.version == 1


@pytest.mark.parametrize(
    "augmentation, expected",
    [
        ({}, False),
        ({"blur": {"pixels": 3.8}}, False),
        ({"bbshear": {"horizontal": 10, "vertical": 10}}, False),
        ({"flip": {"horizontal": True}}, True),
        ({"flip": {"horizontal": False, "vertical": True}}, False),
        ({"bbflip": {"horizontal": True}}, True),
        ({"flip": None}, False),
    ],
)
def test_horizontal_flip_detection(augmentation, expected):
    """The one augmentation the project forbids outright (plan §2).

    Note `bbshear.horizontal` is NOT a flip — a naive substring check on
    "horizontal" would reject a perfectly good version.
    """
    assert horizontal_flip_enabled(augmentation) is expected


def test_a_flipped_version_is_refused(tmp_path, monkeypatch):
    """Roboflow augments at export time, so a flipped version cannot be un-flipped."""
    from alpr.data_sources import roboflow as rf
    from alpr.data_sources.base import Status

    monkeypatch.setenv("ROBOFLOW_API_KEY", "pretend")
    monkeypatch.setattr(rf, "resolve_version", lambda project, key: 3)
    flipped = {"augmentation": {"flip": {"horizontal": True}}}
    monkeypatch.setattr(rf, "version_details", lambda project, version, key: flipped)

    result = PlacasColombianasSource(DataPaths(tmp_path).ensure()).fetch()
    assert result.status is Status.FAILED
    assert "HORIZONTAL FLIP" in result.message
    assert "§2" in result.message


def test_a_project_with_no_versions_explains_the_workaround(tmp_path, monkeypatch):
    """placas-colombianas has 1,770 images but zero generated versions upstream."""
    from alpr.data_sources import roboflow as rf

    monkeypatch.setattr(rf, "_get_json", lambda url, **kw: {"versions": []})
    with pytest.raises(RoboflowError, match="fork it into your own workspace"):
        rf.resolve_version(PlacasColombianasSource.project, "pretend")
