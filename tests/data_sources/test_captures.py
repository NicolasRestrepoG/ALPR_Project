"""The capture app's output contract (docs/capture_app_spec.md §2).

These tests ARE the specification. The app should be developed against
`acquire import-captures --dry-run`, and every rule the app must satisfy has a test
here that fails loudly when violated.
"""

from __future__ import annotations

import json

import pytest

from alpr.data_sources.base import PlateType
from alpr.data_sources.captures import (
    ALLOWED_TAGS,
    CAPTURES_NAME,
    COVERAGE_TARGETS,
    CaptureError,
    example_line,
    import_captures,
    parse_capture_line,
    read_captures,
)
from alpr.data_sources.own_photos import LABELS_HEADER, LABELS_NAME, read_labels

from .factories import make_jpeg


def build(tmp_path, lines, images=(("testset", "s1_20260809_193042_0001.jpg", 4032, 3024),)):
    """Lay out a capture folder: images plus a captures.jsonl."""
    for kind, name, w, h in images:
        make_jpeg(tmp_path / kind / name, w, h)
    (tmp_path / CAPTURES_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_a_valid_capture_imports(tmp_path):
    root = build(tmp_path, [example_line()])
    report = import_captures(root)

    assert report.ok, report.problems
    assert len(report.captures) == 1
    capture = report.captures[0]
    assert capture.plate == "ABC123"
    assert capture.plate_type is PlateType.PARTICULAR
    assert capture.tags == ("night", "yaw_high")
    assert capture.session == "s1"


def test_labels_csv_matches_the_own_photos_contract(tmp_path):
    root = build(tmp_path, [example_line()])
    import_captures(root)

    labels = read_labels(root / LABELS_NAME)
    assert "s1_20260809_193042_0001.jpg" in labels
    label = labels["s1_20260809_193042_0001.jpg"]
    assert label.plate == "ABC123"
    assert label.tags == ("night", "yaw_high")
    assert label.consent == "visitor-signage"


def test_corner_columns_are_written_empty(tmp_path):
    """Corners are the desktop tool's job. Emitting zeros here would let a bad
    rectification masquerade as a good one."""
    root = build(tmp_path, [example_line()])
    import_captures(root)

    text = (root / LABELS_NAME).read_text()
    header, row = text.splitlines()[0].split(","), text.splitlines()[1].split(",")
    assert header == list(LABELS_HEADER)
    for column in ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"):
        assert row[header.index(column)] == ""

    assert read_labels(root / LABELS_NAME)["s1_20260809_193042_0001.jpg"].corners is None


def test_dry_run_writes_nothing(tmp_path):
    root = build(tmp_path, [example_line()])
    report = import_captures(root, dry_run=True)
    assert report.ok
    assert not (root / LABELS_NAME).exists()


# --------------------------------------------------------------------------- #
# Append-only semantics
# --------------------------------------------------------------------------- #


def test_last_line_wins(tmp_path):
    """Correcting a typo means appending, never rewriting."""
    root = build(
        tmp_path,
        [example_line(plate="ABC123"), example_line(plate="ABD123")],
    )
    report = import_captures(root)
    assert report.ok
    assert len(report.captures) == 1
    assert report.captures[0].plate == "ABD123"
    assert report.superseded == 1


def test_a_tombstone_removes_a_capture(tmp_path):
    root = build(
        tmp_path,
        [
            example_line(),
            json.dumps({"schema": 1, "filename": "s1_20260809_193042_0001.jpg", "deleted": True}),
        ],
    )
    report = import_captures(root)
    assert report.captures == []
    assert report.tombstoned == 1


def test_a_capture_can_be_resurrected_after_a_tombstone(tmp_path):
    root = build(
        tmp_path,
        [
            example_line(),
            json.dumps({"schema": 1, "filename": "s1_20260809_193042_0001.jpg", "deleted": True}),
            example_line(plate="XYZ789"),
        ],
    )
    report = import_captures(root)
    assert len(report.captures) == 1
    assert report.captures[0].plate == "XYZ789"


def test_blank_lines_are_tolerated(tmp_path):
    root = build(tmp_path, ["", example_line(), ""])
    assert import_captures(root).ok


def test_a_truncated_final_line_is_reported_not_silently_dropped(tmp_path):
    """The app is killed mid-write: everything before must still import."""
    root = build(tmp_path, [example_line(), '{"schema":1,"filename":"s1_2026'])
    report = read_captures(root / CAPTURES_NAME)
    assert not report.ok
    assert any("not valid JSON" in p for p in report.problems)


# --------------------------------------------------------------------------- #
# Field validation — every rule the app must enforce
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"schema": 2}, "not supported"),
        ({"filename": ""}, "missing required field"),
        ({"filename": "sub/dir/x.jpg"}, "bare name"),
        ({"kind": "photos"}, "must be one of"),
        ({"plate_type": "lorry"}, "must be one of"),
        ({"session": ""}, "missing required field"),
        ({"captured_at": ""}, "missing required field"),
        ({"tags": ["nite"]}, "unknown tag"),
        ({"tags": "night"}, "tags must be a list"),
        ({"consent": "probably-fine"}, "must be one of"),
    ],
)
def test_invalid_fields_are_rejected(overrides, match):
    payload = json.loads(example_line(**overrides))
    with pytest.raises(CaptureError, match=match):
        parse_capture_line(payload, 1)


@pytest.mark.parametrize("plate", ["ABC12", "ABC1234", "AB1234", "ABCD12", "abc123!", "123ABC"])
def test_particular_plates_must_be_lll_ddd(plate):
    payload = json.loads(example_line(plate=plate))
    with pytest.raises(CaptureError, match="not LLLDDD"):
        parse_capture_line(payload, 1)


def test_lowercase_plates_are_accepted_and_upcased():
    _, capture = parse_capture_line(json.loads(example_line(plate="abc123")), 1)
    assert capture.plate == "ABC123"


def test_a_particular_plate_with_no_string_is_refused():
    """An unlabelled particular plate cannot be scored, so it must not enter silently."""
    payload = json.loads(example_line())
    del payload["plate"]
    with pytest.raises(CaptureError, match="record it as plate_type 'unknown'"):
        parse_capture_line(payload, 1)


def test_non_particular_types_need_no_plate_string():
    for plate_type in ("publico", "moto", "non_plate", "unknown"):
        payload = json.loads(example_line(plate_type=plate_type))
        payload.pop("plate", None)
        _, capture = parse_capture_line(payload, 1)
        assert capture.plate is None


def test_publico_plates_skip_the_format_check():
    """Público plates have a readable string but are never read, so no LLLDDD rule."""
    _, capture = parse_capture_line(
        json.loads(example_line(plate_type="publico", plate="ABC12D")), 1
    )
    assert capture.plate == "ABC12D"


def test_the_tag_vocabulary_is_closed():
    """A typo must be rejected, not quietly create a tag no slice ever matches."""
    assert "night" in ALLOWED_TAGS
    assert "blur" not in ALLOWED_TAGS  # judged from the image, not by eye
    assert "small" not in ALLOWED_TAGS  # computed from corners in Phase 2


# --------------------------------------------------------------------------- #
# Cross-checks against the images on disk
# --------------------------------------------------------------------------- #


def test_a_label_with_no_image_is_a_problem(tmp_path):
    root = build(tmp_path, [example_line(filename="missing.jpg")])
    report = import_captures(root)
    assert not report.ok
    assert any("no such file on disk" in p for p in report.problems)


def test_an_image_with_no_label_is_a_warning_not_a_problem(tmp_path):
    root = build(
        tmp_path,
        [example_line()],
        images=(
            ("testset", "s1_20260809_193042_0001.jpg", 4032, 3024),
            ("testset", "s1_20260809_193118_0002.jpg", 4032, 3024),
        ),
    )
    report = import_captures(root)
    assert report.ok
    assert any("no label" in w for w in report.warnings)


def test_dimension_mismatch_points_at_exif_orientation(tmp_path):
    """The classic phone bug: pixels landscape, EXIF says rotate.

    Corner coordinates would then not match image dimensions and rectification
    silently produces garbage, so it has to be caught at import.
    """
    root = build(
        tmp_path,
        [example_line(image_width=3024, image_height=4032)],
        images=(("testset", "s1_20260809_193042_0001.jpg", 4032, 3024),),
    )
    report = import_captures(root)
    assert not report.ok
    assert any("EXIF orientation" in p for p in report.problems)


def test_problems_prevent_writing_labels(tmp_path):
    root = build(tmp_path, [example_line(plate="BAD")])
    report = import_captures(root)
    assert not report.ok
    assert not (root / LABELS_NAME).exists()


def test_a_missing_captures_file_says_what_to_do(tmp_path):
    report = import_captures(tmp_path)
    assert not report.ok
    assert any("capture_app_spec" in p for p in report.problems)


# --------------------------------------------------------------------------- #
# Coverage — what the app's live counter must reproduce
# --------------------------------------------------------------------------- #


def test_coverage_counts_characters_across_particular_plates(tmp_path):
    lines, images = [], []
    for i, plate in enumerate(["ABC123", "ABC456", "XYZ789"]):
        name = f"s1_2026080_00000{i}.jpg"
        images.append(("testset", name, 4032, 3024))
        lines.append(example_line(filename=name, plate=plate))
    report = read_captures(build(tmp_path, lines, images) / CAPTURES_NAME)

    counts = report.coverage("testset")
    assert counts["A"] == 2
    assert counts["X"] == 1
    assert counts["1"] == 1
    assert counts["Q"] == 0


def test_shortfall_reports_how_many_more_are_needed(tmp_path):
    root = build(tmp_path, [example_line()])
    report = read_captures(root / CAPTURES_NAME)

    shortfall = report.shortfall("testset")
    assert shortfall["A"] == COVERAGE_TARGETS["testset"] - 1
    assert shortfall["Q"] == COVERAGE_TARGETS["testset"]


def test_the_two_coverage_targets_differ():
    """Test set needs 5x per character (Sprint 2.1); close-ups need 2x (Sprint 2.2)."""
    assert COVERAGE_TARGETS["testset"] == 5
    assert COVERAGE_TARGETS["close_ups"] == 2


@pytest.mark.parametrize(
    "plate_type, is_negative",
    [
        ("particular", False),
        ("publico", True),
        ("moto", True),
        ("non_plate", True),
        ("diplomatic", True),
        ("official", True),
        ("trailer", True),
        ("temporary", True),
        ("unknown", False),  # "could not determine" is not a class
    ],
)
def test_every_non_particular_class_counts_as_a_negative(plate_type, is_negative):
    """Plan §2 lists moto, público, diplomatic, official, trailer and temporary as
    negatives. Counting only the first two would leave a diplomatic plate invisible
    to the >=40 negatives gate, so the shoot could pass while under-covering the
    rejection head."""
    assert PlateType(plate_type).is_negative is is_negative


def test_unknown_counts_as_neither_positive_nor_negative(tmp_path, capsys):
    from alpr.data_sources.acquire import main

    root = build(
        tmp_path,
        [example_line(plate_type="unknown", plate=None)],
    )
    main(["import-captures", str(root), "--dry-run"])
    out = capsys.readouterr().out
    assert "positives         0 / 200" in out
    assert "negatives         0 / 40" in out
    assert "neither positive nor negative" in out


def test_a_diplomatic_plate_counts_toward_the_negatives_gate(tmp_path, capsys):
    from alpr.data_sources.acquire import main

    root = build(tmp_path, [example_line(plate_type="diplomatic", plate=None)])
    main(["import-captures", str(root), "--dry-run"])
    assert "negatives         1 / 40" in capsys.readouterr().out


def test_negatives_do_not_count_toward_character_coverage(tmp_path):
    root = build(tmp_path, [example_line(plate_type="publico", plate="ABC123")])
    report = read_captures(root / CAPTURES_NAME)
    assert report.coverage("testset")["A"] == 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_dry_run_reports_the_gate(tmp_path, capsys):
    from alpr.data_sources.acquire import main

    root = build(tmp_path, [example_line()])
    assert main(["import-captures", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Sprint 2.1 exit gate" in out
    assert "still needed" in out
    assert "dry run: valid" in out


def test_cli_exits_nonzero_on_problems(tmp_path, capsys):
    from alpr.data_sources.acquire import main

    root = build(tmp_path, [example_line(plate="NOPE")])
    assert main(["import-captures", str(root)]) == 1
    assert "PROBLEM" in capsys.readouterr().out
