"""Manifest round-trip, disk verification, and the licence lineage audit."""

from __future__ import annotations

import json

import pytest

from alpr.data_sources.base import PlateType, Record, write_jsonl
from alpr.data_sources.manifest import (
    Manifest,
    ManifestError,
    build_manifest,
    verify_manifest,
)
from alpr.data_sources.storage import DataPaths

from .factories import make_jpeg


def seed_disk(tmp_path, *, with_blocker=False):
    """Put a little of a clean source (and optionally a blocking one) on disk."""
    paths = DataPaths(tmp_path).ensure()

    ccpd = paths.raw("ccpd") / "ccpd_base"
    ccpd.mkdir(parents=True)
    make_jpeg(ccpd / "a.jpg", 720, 1160)
    make_jpeg(ccpd / "b.jpg", 720, 1160)
    write_jsonl(
        paths.index("ccpd"),
        [
            Record(
                source="ccpd",
                image_path="raw/ccpd/ccpd_base/a.jpg",
                license_key="mit",
                plate_type=PlateType.FOREIGN,
            )
        ],
    )

    if with_blocker:
        pm = paths.raw("platesmania")
        pm.mkdir(parents=True)
        make_jpeg(pm / "co_001.jpg", 800, 600)
        write_jsonl(
            paths.index("platesmania"),
            [
                Record(
                    source="platesmania",
                    image_path="raw/platesmania/co_001.jpg",
                    license_key="platesmania-terms",
                    attribution_url="https://platesmania.com/co/nomer1",
                )
            ],
        )
    return paths


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_manifest_survives_a_save_load_round_trip(tmp_path):
    """Sprint 1.2 exit gate."""
    paths = seed_disk(tmp_path)
    manifest = build_manifest(paths)
    manifest.save(paths.manifest_path)

    loaded = Manifest.load(paths.manifest_path)
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.sources["ccpd"].file_count == 2
    assert loaded.sources["ccpd"].record_count == 1


def test_manifest_matches_what_is_on_disk(tmp_path):
    """The other half of the exit gate: recorded state equals actual state."""
    paths = seed_disk(tmp_path)
    manifest = build_manifest(paths)
    manifest.save(paths.manifest_path)

    result = verify_manifest(Manifest.load(paths.manifest_path), paths)
    assert result.ok, str(result)
    assert result.checked == len(manifest.sources)


def test_verify_notices_a_deleted_file(tmp_path):
    paths = seed_disk(tmp_path)
    manifest = build_manifest(paths)
    manifest.save(paths.manifest_path)

    (paths.raw("ccpd") / "ccpd_base" / "b.jpg").unlink()

    result = verify_manifest(Manifest.load(paths.manifest_path), paths)
    assert not result.ok
    assert any("2 files recorded, 1 on disk" in p for p in result.problems)


def test_verify_notices_a_changed_index(tmp_path):
    paths = seed_disk(tmp_path)
    manifest = build_manifest(paths)
    manifest.save(paths.manifest_path)

    paths.index("ccpd").write_text("", encoding="utf-8")

    result = verify_manifest(Manifest.load(paths.manifest_path), paths)
    assert not result.ok
    assert any("index has changed" in p for p in result.problems)


def test_unsupported_version_is_refused(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 99, "sources": {}}))
    with pytest.raises(ManifestError, match="not supported"):
        Manifest.load(path)


def test_malformed_json_is_refused(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{not json")
    with pytest.raises(ManifestError, match="not valid JSON"):
        Manifest.load(path)


def test_missing_manifest_says_what_to_run(tmp_path):
    with pytest.raises(ManifestError, match="acquire manifest"):
        Manifest.load(tmp_path / "nope.json")


def test_in_flight_downloads_are_not_counted(tmp_path):
    """A 13 GB `.part` file is a download in progress, not acquired data."""
    paths = seed_disk(tmp_path)
    (paths.raw("ccpd") / "CCPD2019.tar.xz.part").write_bytes(b"x" * 4096)

    manifest = build_manifest(paths)
    assert manifest.sources["ccpd"].file_count == 2  # the two jpegs, not the .part


def test_fetched_at_is_carried_forward(tmp_path):
    paths = seed_disk(tmp_path)
    first = build_manifest(paths)
    assert first.sources["ccpd"].fetched_at

    second = build_manifest(paths, previous=first)
    assert second.sources["ccpd"].fetched_at == first.sources["ccpd"].fetched_at


# --------------------------------------------------------------------------- #
# The audit
# --------------------------------------------------------------------------- #


def test_audit_says_yes_when_only_clean_sources_are_present(tmp_path):
    manifest = build_manifest(seed_disk(tmp_path))
    assert manifest.commercial_ok() is True
    report = manifest.audit_report()
    assert "COMMERCIAL DEPLOYMENT: YES" in report
    assert "ccpd" in report


def test_audit_says_no_and_names_the_blocker(tmp_path):
    """Sprint 1.2 exit gate: --audit reports NO and names the blocker."""
    manifest = build_manifest(seed_disk(tmp_path, with_blocker=True))
    assert manifest.commercial_ok() is False

    blockers = [e.name for e in manifest.blockers()]
    assert blockers == ["platesmania"]

    report = manifest.audit_report()
    assert "COMMERCIAL DEPLOYMENT: NO" in report
    assert "blocked by platesmania" in report
    assert "backlink" in report  # the licence note explains *why*


def test_registered_but_unacquired_sources_do_not_block(tmp_path):
    """UFPR is registered from day one but blocks nothing until it is on disk.

    Otherwise the audit would report NO permanently and stop meaning anything.
    """
    manifest = build_manifest(seed_disk(tmp_path))
    assert "ufpr_alpr" in manifest.sources
    assert "ufpr_alpr" not in [e.name for e in manifest.blockers()]
    assert "REGISTERED but not acquired" in manifest.audit_report()


def test_audit_reminds_about_agpl_when_otherwise_clean(tmp_path):
    report = build_manifest(seed_disk(tmp_path)).audit_report()
    assert "AGPL" in report


def test_status_reflects_progress(tmp_path):
    paths = seed_disk(tmp_path)
    manifest = build_manifest(paths)
    assert manifest.sources["ccpd"].status == "indexed"
    assert manifest.sources["ufpr_alpr"].status == "manual_required"
    assert manifest.sources["divipola"].status == "not_started"
