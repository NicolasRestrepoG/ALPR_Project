"""The label store writes exactly what Phase 1's reader expects, and nothing it can't read."""

from __future__ import annotations

import pytest

from alpr.data_sources.base import Box, PlateType, Record, write_jsonl
from alpr.data_sources.own_photos import LABELS_HEADER, read_labels
from alpr.data_sources.storage import DataPaths
from alpr.plates.labels import (
    LabelError,
    LabelStore,
    coverage,
    labels_path,
    list_items,
    load_labelled_image,
    rotation_of,
    validate,
    with_rotation,
)
from alpr.plates.rectify import RectifyError

QUAD = [(110.0, 52.0), (412.0, 60.0), (405.0, 205.0), (104.0, 196.0)]


@pytest.mark.parametrize(
    "text, kind, stored",
    [
        ("abc123", PlateType.PARTICULAR, "ABC123"),
        ("ABC-123", PlateType.PARTICULAR, "ABC123"),
        ("abc·123", PlateType.PARTICULAR, "ABC123"),
        ("KSP30C", PlateType.MOTO, "KSP30C"),
        ("", PlateType.UNKNOWN, None),
        ("", PlateType.NON_PLATE, None),
    ],
)
def test_validate_normalises(text, kind, stored):
    assert validate(text, kind) == stored


@pytest.mark.parametrize(
    "text, kind",
    [
        ("AB1234", PlateType.PARTICULAR),
        ("ABC12", PlateType.PARTICULAR),
        ("", PlateType.PARTICULAR),  # a particular plate must have its text
        ("ABC1234", PlateType.MOTO),
    ],
)
def test_validate_rejects(text, kind):
    with pytest.raises(LabelError):
        validate(text, kind)


def test_round_trip_through_the_phase1_reader(tmp_path):
    path = tmp_path / "labels.csv"
    store = LabelStore(path)
    store.put(
        "a.jpg",
        plate="fgo681",
        plate_type=PlateType.PARTICULAR,
        corners=[QUAD[2], QUAD[0], QUAD[3], QUAD[1]],
        tags=["worn", "glyph_source"],
    )
    labels = read_labels(path)
    lb = labels["a.jpg"]
    assert lb.plate == "FGO681"
    assert lb.plate_type is PlateType.PARTICULAR
    assert lb.corners == tuple(QUAD)  # stored in TL/TR/BR/BL whatever the click order
    assert lb.tags == ("glyph_source", "worn")
    assert path.read_text().splitlines()[0] == ",".join(LABELS_HEADER)


def test_store_reloads_and_updates_in_place(tmp_path):
    path = tmp_path / "labels.csv"
    LabelStore(path).put("a.jpg", plate="ABC123", plate_type=PlateType.PARTICULAR, corners=QUAD)
    store = LabelStore(path)
    assert "a.jpg" in store and len(store) == 1
    store.put("a.jpg", plate="ABD123", plate_type=PlateType.PARTICULAR, corners=QUAD)
    assert read_labels(path)["a.jpg"].plate == "ABD123"
    assert store.delete("a.jpg")
    assert read_labels(path) == {}


def test_existing_consent_and_session_survive_a_relabel(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(",".join(LABELS_HEADER) + "\na.jpg,ABC123,particular,,,,,,,,,,written,s1\n")
    LabelStore(path).put("a.jpg", plate="ABC123", plate_type=PlateType.PARTICULAR, corners=QUAD)
    lb = read_labels(path)["a.jpg"]
    assert (lb.consent, lb.session) == ("written", "s1")


def test_bad_corners_are_refused_before_anything_is_written(tmp_path):
    path = tmp_path / "labels.csv"
    store = LabelStore(path)
    with pytest.raises(RectifyError):
        store.put(
            "a.jpg",
            plate="ABC123",
            plate_type=PlateType.PARTICULAR,
            corners=[(0, 0), (1, 1), (2, 2), (3, 3)],
        )
    assert not path.exists()


def test_corners_required_except_for_no_plate(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    with pytest.raises(LabelError, match="corners"):
        store.put("a.jpg", plate="ABC123", plate_type=PlateType.PARTICULAR, corners=None)
    store.put("b.jpg", plate=None, plate_type=PlateType.NON_PLATE, corners=None)
    assert store.get("b.jpg").corners is None


def test_labels_path_follows_the_phase1_contract_for_own_photos(tmp_path):
    paths = DataPaths(tmp_path)
    assert labels_path(paths, "own_photos") == tmp_path / "raw" / "own_photos" / "labels.csv"
    assert labels_path(paths, "roboflow_x") == tmp_path / "labels" / "roboflow_x.csv"


def test_items_from_an_index_largest_first_and_multi_plate_keys(tmp_path):
    paths = DataPaths(tmp_path).ensure()
    recs = [
        Record(
            source="rf", image_path="raw/rf/a.jpg", license_key="cc-by-4.0", box=Box(10, 10, 60, 30)
        ),
        Record(
            source="rf",
            image_path="raw/rf/b.jpg",
            license_key="cc-by-4.0",
            box=Box(500, 10, 900, 200),
        ),
        Record(
            source="rf",
            image_path="raw/rf/b.jpg",
            license_key="cc-by-4.0",
            box=Box(10, 10, 100, 50),
        ),
    ]
    write_jsonl(paths.index("rf"), recs)
    items = list_items(paths, "rf")
    assert [it.key for it in items] == ["raw/rf/b.jpg#1", "raw/rf/b.jpg#0", "raw/rf/a.jpg"]
    assert items[0].image == tmp_path / "raw" / "rf" / "b.jpg"


def test_items_for_own_photos_keyed_by_filename(tmp_path):
    paths = DataPaths(tmp_path)
    for sub, name in (("testset", "t1.jpg"), ("close_ups", "c1.jpg")):
        d = paths.raw("own_photos") / sub
        d.mkdir(parents=True)
        (d / name).write_bytes(b"x")
    assert {it.key for it in list_items(paths, "own_photos")} == {"t1.jpg", "c1.jpg"}


def test_missing_index_is_explained(tmp_path):
    with pytest.raises(LabelError, match="acquire fetch"):
        list_items(DataPaths(tmp_path), "nothing_here")


def test_coverage_counts_distinct_plates_not_images(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    for k in range(5):  # the same car five times
        store.put(f"{k}.jpg", plate="ITZ923", plate_type=PlateType.PARTICULAR, corners=QUAD)
    store.put("m.jpg", plate="KSP30C", plate_type=PlateType.MOTO, corners=QUAD)
    cov = coverage(store.all())
    assert cov.distinct_plates == 1
    assert cov.char_counts["I"] == 1 and cov.char_counts["9"] == 1
    assert cov.char_counts["K"] == 0  # moto plates are not read, so they do not count
    assert cov.by_type == {"particular": 5, "moto": 1}
    assert "A" in cov.missing()


def test_rotation_tags():
    assert rotation_of(["worn"]) == 0
    assert rotation_of(["rot270", "blurry"]) == 270
    assert sorted(with_rotation(["rot90", "worn"], 180)) == ["rot180", "worn"]
    assert with_rotation(["rot90"], 0) == []
    with pytest.raises(LabelError, match="conflicting"):
        rotation_of(["rot90", "rot180"])
    with pytest.raises(LabelError):
        with_rotation([], 45)


def test_store_refuses_two_rotations(tmp_path):
    store = LabelStore(tmp_path / "labels.csv")
    with pytest.raises(LabelError, match="conflicting"):
        store.put(
            "a.jpg",
            plate="ABC123",
            plate_type=PlateType.PARTICULAR,
            corners=QUAD,
            tags=["rot90", "rot270"],
        )


def test_load_labelled_image_applies_the_rotation(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((40, 60, 3), np.uint8))
    store = LabelStore(tmp_path / "labels.csv")
    lb = store.put(
        "a.png", plate=None, plate_type=PlateType.NON_PLATE, corners=None, tags=["rot90"]
    )
    assert read_labels(store.path)["a.png"].tags == ("rot90",)
    assert load_labelled_image(path, lb).shape == (60, 40, 3)
