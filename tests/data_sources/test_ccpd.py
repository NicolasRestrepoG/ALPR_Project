"""CCPD filename parsing, against the example published in the upstream README.

The exit gate for Sprint 1.1:
  - parsing the README's example filename yields corners ordered TL/TR/BR/BL that
    agree with the published bounding box;
  - corner ordering is verified by winding direction, not by eye;
  - malformed filenames raise rather than returning garbage.
"""

from __future__ import annotations

import pytest

from alpr.data_sources.base import PlateType, Winding, assert_corner_order, corners_to_box, winding
from alpr.data_sources.ccpd import (
    ADS,
    ALPHABETS,
    PROVINCES,
    CCPDFilenameError,
    CCPDSource,
    decode_plate,
    parse_ccpd_filename,
)

# Verbatim from https://github.com/detectRecog/CCPD README, "Dataset Annotations".
README_EXAMPLE = (
    "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
)

# CCPD frames are 720 x 1160.
CCPD_FRAME = (720, 1160)


@pytest.fixture
def ann():
    return parse_ccpd_filename(README_EXAMPLE)


def test_readme_example_plate_string(ann):
    assert ann.plate == "皖AY339S"
    assert ann.plate_indices == (0, 0, 22, 27, 27, 33, 16)
    assert ann.plate[0] == PROVINCES[0]
    assert ann.plate[1] == ALPHABETS[0]
    assert ann.plate[2:] == "".join(ADS[i] for i in (22, 27, 27, 33, 16))


def test_readme_example_scalar_fields(ann):
    assert ann.area_code == 25
    assert (ann.tilt_horizontal, ann.tilt_vertical) == (95, 113)
    assert ann.brightness == 37
    assert ann.blurriness == 15


def test_readme_example_box(ann):
    assert ann.box.as_list() == [154.0, 383.0, 386.0, 473.0]
    assert ann.box.width == 232.0
    assert ann.box.height == 90.0


def test_area_ratio_scale_is_consistent_with_the_frame_size(ann):
    """The README never states the area-ratio scale; the example pins it.

    232 x 90 px of a 720 x 1160 frame is 0.0250, and the published code is 025 —
    so the integer is per mille, not per cent.
    """
    measured = ann.box.area / (CCPD_FRAME[0] * CCPD_FRAME[1])
    assert ann.area_ratio == pytest.approx(measured, abs=5e-4)
    assert ann.area_ratio == pytest.approx(0.025)


def test_raw_vertices_are_published_order(ann):
    """As published: right-bottom first, then LB, LT, RT."""
    assert ann.raw_vertices == (
        (386.0, 473.0),
        (177.0, 454.0),
        (154.0, 383.0),
        (363.0, 402.0),
    )


def test_corners_are_tl_tr_br_bl(ann):
    """THE test. File order is RB, LB, LT, RT; we must emit TL, TR, BR, BL."""
    assert ann.corners == (
        (154.0, 383.0),  # TL — was file index 2
        (363.0, 402.0),  # TR — was file index 3
        (386.0, 473.0),  # BR — was file index 0
        (177.0, 454.0),  # BL — was file index 1
    )


def test_corner_order_verified_by_winding_not_by_eye(ann):
    """Winding must be clockwise-on-screen, and the sequence must be canonical.

    Note the trap this encodes: the published order has the *same* winding as the
    canonical one, so winding is necessary but not sufficient. Both are asserted.
    """
    assert winding(ann.corners) is Winding.CLOCKWISE
    assert winding(ann.raw_vertices) is Winding.CLOCKWISE  # identical — hence the second check
    assert assert_corner_order(ann.corners) == ann.corners
    assert_corner_order(ann.corners, strict=True)


def test_corners_agree_with_the_published_bounding_box(ann):
    """The exit-gate check: the corners' extent must reproduce the published box."""
    derived = corners_to_box(ann.corners)
    assert derived.as_list() == ann.box.as_list()
    assert derived.iou(ann.box) == pytest.approx(1.0)


def test_unnormalized_corners_expose_the_raw_rotation():
    raw = parse_ccpd_filename(README_EXAMPLE, normalize_corners=False)
    normalized = parse_ccpd_filename(README_EXAMPLE)
    # Fixed permutation and geometric ordering agree on a well-behaved sample.
    assert raw.corners == normalized.corners


def test_parses_a_full_path_not_just_a_name():
    ann = parse_ccpd_filename(f"/data/raw/ccpd/ccpd_base/{README_EXAMPLE}")
    assert ann.plate == "皖AY339S"


def test_green_subset_eight_character_plate():
    """CCPD-Green energy vehicles carry an eight-character plate."""
    name = README_EXAMPLE.replace("0_0_22_27_27_33_16", "0_0_22_27_27_33_16_24")
    ann = parse_ccpd_filename(name)
    assert len(ann.plate) == 8
    assert ann.plate == "皖AY339S0"


@pytest.mark.parametrize(
    "name, match",
    [
        ("", "7 '-'-separated fields"),
        ("025-95_113-154&383_386&473.jpg", "7 '-'-separated fields"),
        (README_EXAMPLE + "-extra.jpg", "7 '-'-separated fields"),
        (README_EXAMPLE.replace("025-", "abc-"), "area"),
        (README_EXAMPLE.replace("95_113", "95"), "tilt field"),
        (README_EXAMPLE.replace("95_113", "95_x"), "vertical tilt"),
        (README_EXAMPLE.replace("154&383_386&473-", "154&383-"), "bounding box field"),
        (README_EXAMPLE.replace("154&383_386", "154x383_386"), "missing its '&'"),
        (README_EXAMPLE.replace("154&383_386&473-", "154&383_1&2-"), "positive extent"),
        (README_EXAMPLE.replace("_177&454", ""), "exactly 4 'x&y' points"),
        (README_EXAMPLE.replace("0_0_22_27_27_33_16", "0_0_22"), "7 or 8 plate character"),
        (README_EXAMPLE.replace("0_0_22_27_27_33_16", "99_0_22_27_27_33_16"), "outside the table"),
        (README_EXAMPLE.replace("-37-15", "-x-15"), "brightness"),
        (README_EXAMPLE.replace("-37-15", "-37-x"), "blurriness"),
    ],
)
def test_malformed_filenames_raise(name, match):
    """Malformed names raise rather than returning garbage.

    A filename that silently yields bad coordinates is indistinguishable from a
    good one until the detector fails to converge weeks later.
    """
    with pytest.raises(CCPDFilenameError, match=match):
        parse_ccpd_filename(name)


def test_degenerate_vertices_raise():
    collinear = README_EXAMPLE.replace(
        "386&473_177&454_154&383_363&402", "100&100_200&200_300&300_400&400"
    )
    with pytest.raises(CCPDFilenameError, match="degenerate"):
        parse_ccpd_filename(collinear)


def test_decode_plate_rejects_bad_lengths():
    with pytest.raises(CCPDFilenameError, match="7 or 8"):
        decode_plate((0, 0, 1))


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


def test_index_produces_records_with_ordered_corners(tmp_path):
    from alpr.data_sources.storage import DataPaths

    paths = DataPaths(tmp_path).ensure()
    base = paths.raw("ccpd") / "ccpd_base"
    base.mkdir(parents=True)
    (base / README_EXAMPLE).write_bytes(b"not really a jpeg")

    source = CCPDSource(paths)
    records = list(source.index())

    assert len(records) == 1
    rec = records[0]
    assert rec.source == "ccpd"
    assert rec.image_path == f"raw/ccpd/ccpd_base/{README_EXAMPLE}"
    assert rec.plate == "皖AY339S"
    assert rec.plate_type is PlateType.FOREIGN  # never read as a Colombian plate
    assert rec.plate_type.readable is False
    assert rec.commercial_ok is True  # MIT
    assert "base" in rec.tags
    assert_corner_order(rec.corners, strict=True)
    assert corners_to_box(rec.corners).as_list() == rec.box.as_list()


def test_index_skips_unparseable_names_without_aborting(tmp_path):
    from alpr.data_sources.storage import DataPaths

    paths = DataPaths(tmp_path).ensure()
    base = paths.raw("ccpd") / "ccpd_base"
    base.mkdir(parents=True)
    (base / README_EXAMPLE).write_bytes(b"x")
    (base / "garbage-name.jpg").write_bytes(b"x")

    source = CCPDSource(paths)
    records = list(source.index())

    assert len(records) == 1
    assert source.last_index_skipped == 1
    assert source.last_index_errors


def test_extracted_size_is_not_assumed_equal_to_the_archive():
    """Measured, not guessed: 20% extracted to 4.8 GB, so the full tree is ~24 GB.

    The plausible-sounding assumption — "the payload is JPEG, so the archive is
    about the size of the tree" — is wrong by nearly 2x, because xz compresses
    tar's per-member padding across ~350k small files. Under-estimating here fills
    the disk 90% of the way through a one-hour extraction.
    """
    assert CCPDSource.EXTRACTED_BYTES > CCPDSource.ARCHIVE_BYTES * 1.5


def test_preflight_accounts_for_the_extracted_size_not_the_archive_size(tmp_path, monkeypatch):
    from alpr.data_sources import ccpd as ccpd_mod
    from alpr.data_sources.storage import DataPaths

    source = CCPDSource(DataPaths(tmp_path).ensure())
    source.archive_path.parent.mkdir(parents=True, exist_ok=True)
    source.archive_path.write_bytes(b"pretend archive")

    # Room for the archive but not for a full extraction.
    monkeypatch.setattr(ccpd_mod, "free_space", lambda p: 15_000_000_000)
    assert source.preflight(keep_fraction=1.0) is not None
    assert source.preflight(keep_fraction=0.2) is None


def test_preflight_refuses_when_the_disk_is_too_small(tmp_path, monkeypatch):
    """A 13 GB download that dies at 90% with ENOSPC is the worst way to learn this."""
    from alpr.data_sources import ccpd as ccpd_mod
    from alpr.data_sources.storage import DataPaths

    source = CCPDSource(DataPaths(tmp_path).ensure())
    monkeypatch.setattr(ccpd_mod, "free_space", lambda p: 5_000_000_000)

    problem = source.preflight()
    assert problem is not None
    assert "not enough disk space" in problem
    assert "ALPR_DATA_ROOT" in problem

    result = source.fetch()
    assert result.status.value == "failed"
    assert not source.archive_path.exists()


def test_preflight_passes_with_room(tmp_path, monkeypatch):
    from alpr.data_sources import ccpd as ccpd_mod
    from alpr.data_sources.storage import DataPaths

    source = CCPDSource(DataPaths(tmp_path).ensure())
    monkeypatch.setattr(ccpd_mod, "free_space", lambda p: 500_000_000_000)
    assert source.preflight() is None


def test_no_plate_subset_yields_negative_records(tmp_path):
    from alpr.data_sources.storage import DataPaths

    paths = DataPaths(tmp_path).ensure()
    np_dir = paths.raw("ccpd") / "ccpd_np"
    np_dir.mkdir(parents=True)
    (np_dir / "0000.jpg").write_bytes(b"x")

    records = list(CCPDSource(paths).index())
    assert len(records) == 1
    assert records[0].plate_type is PlateType.NON_PLATE
    assert records[0].plate is None
    assert records[0].corners is None
