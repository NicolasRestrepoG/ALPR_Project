"""Platesmania, the gated academic datasets, dashcam and own photos.

The shared property across all four: **a manual source prints instructions and
exits cleanly. It never raises.** A source that raises is a source you stop
running, and two of these have a multi-day lead time.
"""

from __future__ import annotations

import pytest

from alpr.data_sources.base import PlateType, RecordValidationError, Status
from alpr.data_sources.dashcam import DashcamSource, VideoProvenance, read_video_list
from alpr.data_sources.gated import RodosolAlprSource, UfprAlprSource
from alpr.data_sources.own_photos import LabelsError, OwnPhotosSource, read_labels
from alpr.data_sources.platesmania import (
    SIDECAR_NAME,
    PlatesmaniaSource,
    SidecarError,
    read_sidecar,
    write_sidecar_template,
)
from alpr.data_sources.registry import select
from alpr.data_sources.storage import DataPaths

from .factories import make_jpeg


@pytest.mark.parametrize(
    "cls",
    [PlatesmaniaSource, UfprAlprSource, RodosolAlprSource, DashcamSource, OwnPhotosSource],
)
def test_manual_sources_instruct_rather_than_raise(cls, tmp_path):
    """Sprint 1.2 exit gate: manual sources print actionable instructions."""
    result = cls(DataPaths(tmp_path).ensure()).fetch()
    assert result.status in (Status.MANUAL_REQUIRED, Status.READY, Status.UNAVAILABLE)
    assert len(result.message) > 100  # actually instructive, not a one-liner


# --------------------------------------------------------------------------- #
# Platesmania — attribution is a licence condition
# --------------------------------------------------------------------------- #


def make_platesmania(tmp_path, *, sidecar_rows=None, images=("co_001.jpg",)):
    paths = DataPaths(tmp_path).ensure()
    raw = paths.raw("platesmania")
    raw.mkdir(parents=True, exist_ok=True)
    for name in images:
        make_jpeg(raw / name, 800, 600)
    if sidecar_rows is not None:
        lines = ["filename,source_url,plate,plate_type,notes", *sidecar_rows]
        (raw / SIDECAR_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def test_attributed_images_index_normally(tmp_path):
    paths = make_platesmania(
        tmp_path,
        sidecar_rows=["co_001.jpg,https://platesmania.com/co/nomer123,ABC123,particular,"],
    )
    records = list(PlatesmaniaSource(paths).index())
    assert len(records) == 1
    assert records[0].attribution_url == "https://platesmania.com/co/nomer123"
    assert records[0].plate == "ABC123"
    assert records[0].commercial_ok is False


def test_image_missing_from_the_sidecar_is_refused(tmp_path):
    """The plan's rule: an unusable image must not enter the corpus silently."""
    paths = make_platesmania(
        tmp_path,
        images=("co_001.jpg", "co_002.jpg"),
        sidecar_rows=["co_001.jpg,https://platesmania.com/co/nomer123,ABC123,particular,"],
    )
    source = PlatesmaniaSource(paths)
    records = list(source.index())
    assert [r.image_path.rsplit("/", 1)[-1] for r in records] == ["co_001.jpg"]
    assert source.last_index_skipped == 1
    assert "co_002.jpg" in source.last_index_errors[0]


def test_a_row_without_a_url_raises_rather_than_being_dropped(tmp_path):
    paths = make_platesmania(tmp_path, sidecar_rows=["co_001.jpg,,ABC123,particular,"])
    with pytest.raises(SidecarError, match="no source_url"):
        list(PlatesmaniaSource(paths).index())


def test_a_non_url_is_rejected(tmp_path):
    paths = make_platesmania(tmp_path, sidecar_rows=["co_001.jpg,see my notes,ABC123,particular,"])
    with pytest.raises(SidecarError, match="not a URL"):
        read_sidecar(paths.raw("platesmania") / SIDECAR_NAME)


def test_unknown_plate_type_is_rejected(tmp_path):
    paths = make_platesmania(tmp_path, sidecar_rows=["co_001.jpg,https://x.co/1,ABC123,lorry,"])
    with pytest.raises(SidecarError, match="unknown plate_type"):
        read_sidecar(paths.raw("platesmania") / SIDECAR_NAME)


def test_missing_sidecar_produces_a_template(tmp_path):
    paths = make_platesmania(tmp_path, images=("co_001.jpg", "co_002.jpg"))
    result = PlatesmaniaSource(paths).fetch()
    assert result.status is Status.MANUAL_REQUIRED
    template = paths.raw("platesmania") / "sources.template.csv"
    assert template.exists()
    assert "co_002.jpg" in template.read_text()


def test_write_sidecar_template_roundtrips(tmp_path):
    path = write_sidecar_template(tmp_path / "sources.csv", ["a.jpg"])
    with pytest.raises(SidecarError, match="no source_url"):
        read_sidecar(path)


def test_the_record_type_itself_refuses_unattributed_platesmania_images():
    """Belt and braces: even bypassing the sidecar, the Record cannot be built."""
    from alpr.data_sources.base import Record

    with pytest.raises(RecordValidationError, match="attribution"):
        Record(
            source="platesmania",
            image_path="raw/platesmania/co_001.jpg",
            license_key="platesmania-terms",
        )


# --------------------------------------------------------------------------- #
# Gated academic datasets
# --------------------------------------------------------------------------- #


def test_gated_sources_are_excluded_from_the_commercial_path():
    kept = {c.info.name for c in select(commercial_only=True)}
    assert "ufpr_alpr" not in kept
    assert "rodosol_alpr" not in kept


def test_gated_instructions_name_the_contact_and_the_licence_consequence(tmp_path):
    text = UfprAlprSource(DataPaths(tmp_path)).manual_instructions()
    assert "rblsantos@inf.ufpr.br" in text
    assert "CANNOT be commercially deployed" in text
    assert "1-5 day" in text


def test_email_draft_is_ready_to_send(tmp_path):
    draft = RodosolAlprSource(DataPaths(tmp_path)).email_draft()
    assert "RodoSol-ALPR" in draft
    assert "non-commercial" in draft
    assert "To: rblsantos@inf.ufpr.br" in draft


def test_email_draft_signs_from_env_and_never_hardcodes_a_sender(tmp_path, monkeypatch):
    source = RodosolAlprSource(DataPaths(tmp_path))
    assert "<your name>" in source.email_draft()
    assert "<your email>" in source.email_draft()

    monkeypatch.setenv("CONTACT_NAME", "A. Tester")
    monkeypatch.setenv("CONTACT_EMAIL", "tester@example.org")
    draft = source.email_draft()
    assert "A. Tester\ntester@example.org" in draft
    assert "<your" not in draft


def test_filled_draft_is_written_outside_the_repo_docs(tmp_path):
    path = RodosolAlprSource(DataPaths(tmp_path)).draft_path
    assert "docs" not in path.parts
    assert path.is_relative_to(tmp_path)


def test_gated_source_detects_files_once_unpacked(tmp_path):
    paths = DataPaths(tmp_path).ensure()
    raw = paths.raw("ufpr_alpr")
    raw.mkdir(parents=True)
    make_jpeg(raw / "track0001.png", 1920, 1080)

    result = UfprAlprSource(paths).fetch()
    assert result.status is Status.READY

    records = list(UfprAlprSource(paths).index())
    assert len(records) == 1
    assert records[0].plate_type is PlateType.FOREIGN
    assert records[0].commercial_ok is False


# --------------------------------------------------------------------------- #
# Dashcam
# --------------------------------------------------------------------------- #


def test_read_video_list_ignores_comments(tmp_path):
    path = tmp_path / "videos.txt"
    path.write_text("# a comment\n\nhttps://youtu.be/aaa\nhttps://youtu.be/bbb # trailing\n")
    assert read_video_list(path) == ["https://youtu.be/aaa", "https://youtu.be/bbb"]


def test_creative_commons_detection():
    cc = VideoProvenance("a", "u", "t", "up", "Creative Commons Attribution license", 60.0, 3)
    std = VideoProvenance("b", "u", "t", "up", "Standard YouTube License", 60.0, 3)
    assert cc.creative_commons is True
    assert std.creative_commons is False


def test_frames_without_provenance_are_refused(tmp_path):
    """A frame with no video record has no attribution, so it cannot be used."""
    paths = DataPaths(tmp_path).ensure()
    source = DashcamSource(paths)
    frames = source.frames_dir / "vid123"
    frames.mkdir(parents=True)
    make_jpeg(frames / "vid123_00001.jpg", 1280, 720)

    assert list(source.index()) == []
    assert source.last_index_skipped == 1

    source.save_provenance(
        {"vid123": VideoProvenance("vid123", "https://youtu.be/vid123", "t", "u", "CC", 60.0, 1)}
    )
    records = list(source.index())
    assert len(records) == 1
    assert records[0].attribution_url == "https://youtu.be/vid123"


def test_missing_video_list_writes_a_template(tmp_path):
    paths = DataPaths(tmp_path).ensure()
    source = DashcamSource(paths)
    result = source.fetch()
    assert result.status is Status.MANUAL_REQUIRED
    assert source.video_list_path.exists()


# --------------------------------------------------------------------------- #
# Own photos
# --------------------------------------------------------------------------- #


LABELS = """\
filename,plate,plate_type,tl_x,tl_y,tr_x,tr_y,br_x,br_y,bl_x,bl_y,tags,consent,session
IMG_001.jpg,ABC123,particular,10,20,110,22,112,70,12,68,night|yaw_high,resident-signed,s1
IMG_002.jpg,,publico,10,20,110,22,112,70,12,68,,,s1
"""


def make_own_photos(tmp_path, labels=LABELS):
    paths = DataPaths(tmp_path).ensure()
    source = OwnPhotosSource(paths)
    source.testset_dir.mkdir(parents=True, exist_ok=True)
    source.close_ups_dir.mkdir(parents=True, exist_ok=True)
    make_jpeg(source.testset_dir / "IMG_001.jpg", 4032, 3024)
    make_jpeg(source.testset_dir / "IMG_002.jpg", 4032, 3024)
    make_jpeg(source.close_ups_dir / "CLOSE_01.jpg", 4032, 3024)
    if labels is not None:
        source.labels_path.write_text(labels, encoding="utf-8")
    return paths, source


def test_labels_join_onto_images(tmp_path):
    paths, source = make_own_photos(tmp_path)
    records = {r.image_path.rsplit("/", 1)[-1]: r for r in source.index()}

    assert records["IMG_001.jpg"].plate == "ABC123"
    assert records["IMG_001.jpg"].plate_type is PlateType.PARTICULAR
    assert records["IMG_001.jpg"].corners == (
        (10.0, 20.0),
        (110.0, 22.0),
        (112.0, 70.0),
        (12.0, 68.0),
    )
    assert "night" in records["IMG_001.jpg"].tags
    assert records["IMG_001.jpg"].extra["consent"] == "resident-signed"
    assert records["IMG_002.jpg"].plate is None


def test_images_index_before_labels_exist(tmp_path):
    """The annotation tool arrives in Phase 2; the source must work before then."""
    paths, source = make_own_photos(tmp_path, labels=None)
    records = list(source.index())
    assert len(records) == 3
    assert all(r.plate is None and r.corners is None for r in records)
    assert all(r.commercial_ok for r in records)


def test_close_ups_are_tagged_separately(tmp_path):
    paths, source = make_own_photos(tmp_path)
    kinds = {r.image_path.rsplit("/", 1)[-1]: r.tags for r in source.index()}
    assert "close_up" in kinds["CLOSE_01.jpg"]
    assert "testset" in kinds["IMG_001.jpg"]


def test_a_six_character_rule_catches_typos(tmp_path):
    bad = LABELS.replace("ABC123", "ABC12")
    paths, source = make_own_photos(tmp_path, labels=bad)
    with pytest.raises(LabelsError, match="not 6 characters"):
        list(source.index())


def test_partial_corner_columns_raise(tmp_path):
    bad = LABELS.replace("10,20,110,22,112,70,12,68,night", "10,20,110,,112,70,12,68,night")
    paths, source = make_own_photos(tmp_path, labels=bad)
    with pytest.raises(LabelsError, match="partial corner"):
        list(source.index())


def test_corners_from_labels_are_normalized(tmp_path):
    """Even if the annotation tool emits a rotated cycle, the Record is canonical."""
    rotated = LABELS.replace("10,20,110,22,112,70,12,68", "112,70,12,68,10,20,110,22")
    paths, source = make_own_photos(tmp_path, labels=rotated)
    rec = next(r for r in source.index() if r.image_path.endswith("IMG_001.jpg"))
    assert rec.corners == ((10.0, 20.0), (110.0, 22.0), (112.0, 70.0), (12.0, 68.0))


def test_progress_report_tracks_the_sprint_2_1_gate(tmp_path):
    paths, source = make_own_photos(tmp_path)
    result = source.fetch()
    assert "Sprint 2.1 exit gate" in result.message
    assert "positives" in result.message
    assert "night" in result.message


def test_missing_characters_flags_thin_coverage(tmp_path):
    """Per-glyph attribution needs >= 5 occurrences to be signal, not noise."""
    labels = read_labels(make_own_photos(tmp_path)[1].labels_path)
    missing = OwnPhotosSource.missing_characters(labels)
    assert "Q" in missing  # nothing in the fixture contains Q
    assert len(missing) > 30


def test_read_labels_missing_file_is_not_an_error(tmp_path):
    assert read_labels(tmp_path / "nope.csv") == {}
