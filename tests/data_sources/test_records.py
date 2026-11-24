"""Record schema, licence flags, and the JSON Lines round-trip."""

from __future__ import annotations

import pytest

from alpr.data_sources.base import (
    LICENSES,
    Box,
    PlateType,
    Record,
    RecordValidationError,
    get_license,
    read_jsonl,
    write_jsonl,
)

CORNERS = ((10.0, 20.0), (110.0, 22.0), (112.0, 70.0), (12.0, 68.0))


def make_record(**overrides) -> Record:
    kwargs = dict(
        source="own_photos",
        image_path="raw/own_photos/testset/IMG_0001.jpg",
        license_key="own-work",
        plate="ABC123",
        plate_type=PlateType.PARTICULAR,
        box=Box(10, 20, 112, 70),
        corners=CORNERS,
        image_size=(4032, 3024),
        tags=("night", "yaw_high"),
    )
    kwargs.update(overrides)
    return Record(**kwargs)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_plate_is_upper_cased_and_stripped():
    assert make_record(plate="  abc123 ").plate == "ABC123"


def test_absolute_image_path_is_rejected():
    """Paths are relative to the data root so an index is portable between machines."""
    with pytest.raises(RecordValidationError, match="relative to the data root"):
        make_record(image_path="/Users/someone/data/raw/x.jpg")


def test_unknown_licence_key_is_rejected():
    with pytest.raises(RecordValidationError, match="unknown licence key"):
        make_record(license_key="whatever-i-guess")


def test_corners_are_normalized_on_construction():
    """A record built from misordered corners comes out canonical, not wrong."""
    shuffled = (CORNERS[2], CORNERS[0], CORNERS[3], CORNERS[1])
    assert make_record(corners=shuffled).corners == CORNERS


def test_corners_may_be_none_for_box_only_sources():
    rec = make_record(corners=None)
    assert rec.corners is None
    assert rec.has_corners is False


def test_empty_plate_is_rejected():
    with pytest.raises(RecordValidationError, match="non-empty"):
        make_record(plate="   ")


def test_bad_image_size_is_rejected():
    with pytest.raises(RecordValidationError, match="positive"):
        make_record(image_size=(0, 100))


# --------------------------------------------------------------------------- #
# Licences
# --------------------------------------------------------------------------- #


def test_commercial_flag_comes_from_the_licence_table():
    assert make_record(license_key="mit").commercial_ok is True
    assert make_record(license_key="cc-by-4.0").commercial_ok is True
    assert make_record(license_key="academic-nc").commercial_ok is False


def test_unknown_licence_is_not_commercial():
    """Forgetting to verify a licence must exclude the source, never include it."""
    assert get_license("unknown").commercial_ok is False


def test_every_licence_key_matches_its_table_key():
    for key, lic in LICENSES.items():
        assert lic.key == key


def test_per_image_attribution_is_enforced_at_construction():
    """Platesmania's backlink is a licence condition, so a record without one cannot exist."""
    with pytest.raises(RecordValidationError, match="attribution"):
        make_record(license_key="platesmania-terms", attribution_url=None)

    rec = make_record(
        license_key="platesmania-terms",
        attribution_url="https://platesmania.com/co/nomer12345678",
    )
    assert rec.commercial_ok is False


def test_only_particular_plates_are_readable():
    assert PlateType.PARTICULAR.readable is True
    for other in PlateType:
        if other is not PlateType.PARTICULAR:
            assert other.readable is False


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def test_record_json_round_trip_is_exact():
    rec = make_record(extra={"brightness": 37, "note": "ñoño ÁÉÍ"})
    assert Record.from_json(rec.to_json()) == rec


def test_round_trip_preserves_none_fields():
    rec = Record(
        source="roboflow_placas_colombianas",
        image_path="raw/roboflow/train/img_0007.jpg",
        license_key="cc-by-4.0",
        plate=None,
        plate_type=PlateType.PARTICULAR,
        box=Box(1, 2, 3, 4),
        corners=None,
    )
    back = Record.from_json(rec.to_json())
    assert back == rec
    assert back.corners is None
    assert back.plate is None


def test_jsonl_file_round_trip(tmp_path):
    records = [make_record(image_path=f"raw/own_photos/testset/IMG_{i:04d}.jpg") for i in range(5)]
    path = tmp_path / "own_photos.jsonl"

    written = write_jsonl(path, records)
    assert written == 5
    assert list(read_jsonl(path)) == records


def test_jsonl_write_is_atomic_on_failure(tmp_path):
    """A crash mid-write must leave no file at all, not a truncated index."""
    path = tmp_path / "index.jsonl"

    def exploding():
        yield make_record()
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError):
        write_jsonl(path, exploding())

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_jsonl_read_reports_the_offending_line(tmp_path):
    path = tmp_path / "index.jsonl"
    path.write_text(make_record().to_json() + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(Exception, match=r":2:"):
        list(read_jsonl(path))


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "index.jsonl"
    path.write_text("\n" + make_record().to_json() + "\n\n", encoding="utf-8")
    assert len(list(read_jsonl(path))) == 1


def test_resolve_joins_the_data_root(tmp_path):
    rec = make_record()
    assert rec.resolve(tmp_path) == tmp_path / rec.image_path
