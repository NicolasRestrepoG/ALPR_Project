"""The storage contract: layout, atomic downloads, extraction markers, checksums.

The atomicity tests use an injected `fetcher` rather than a network, so they run
offline and in milliseconds.
"""

from __future__ import annotations

import hashlib
import tarfile
import zipfile

import pytest

from alpr.data_sources.storage import (
    DataPaths,
    DownloadError,
    ExtractionError,
    atomic_write_text,
    download,
    extract_archive,
    is_extracted,
    iter_files,
    marker_path,
    read_marker,
    sha256_file,
    verify_checksum,
)

PAYLOAD = b"the quick brown fox" * 100
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def good_fetcher(url, handle):
    handle.write(PAYLOAD)
    return len(PAYLOAD)


def failing_fetcher(url, handle):
    handle.write(PAYLOAD[:50])  # a partial write, then death
    raise ConnectionError("connection reset by peer")


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_layout_matches_the_plan(tmp_path):
    paths = DataPaths(tmp_path)
    assert paths.raw("ccpd") == tmp_path / "raw" / "ccpd"
    assert paths.index("ccpd") == tmp_path / "index" / "ccpd.jsonl"
    assert paths.assets_dir == tmp_path / "assets"
    assert paths.real_test_dir == tmp_path / "real_test"
    assert paths.manifest_path == tmp_path / "manifest.json"


def test_ensure_creates_directories(tmp_path):
    paths = DataPaths(tmp_path).ensure()
    assert paths.raw_dir.is_dir()
    assert paths.index_dir.is_dir()
    assert paths.assets_dir.is_dir()


def test_relative_returns_posix_path_under_the_root(tmp_path):
    paths = DataPaths(tmp_path)
    assert paths.relative(tmp_path / "raw" / "ccpd" / "a.jpg") == "raw/ccpd/a.jpg"


def test_default_data_root_honours_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPR_DATA_ROOT", str(tmp_path / "elsewhere"))
    assert DataPaths.default().root == (tmp_path / "elsewhere").resolve()


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def test_sha256_and_verify(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(PAYLOAD)
    assert sha256_file(p) == PAYLOAD_SHA
    verify_checksum(p, PAYLOAD_SHA)
    verify_checksum(p, PAYLOAD_SHA.upper())
    with pytest.raises(DownloadError, match="checksum mismatch"):
        verify_checksum(p, "0" * 64)


def test_atomic_write_text_leaves_no_temp_files(tmp_path):
    target = tmp_path / "sub" / "out.txt"
    atomic_write_text(target, "hola")
    assert target.read_text() == "hola"
    assert [p.name for p in target.parent.iterdir()] == ["out.txt"]


# --------------------------------------------------------------------------- #
# Download atomicity
# --------------------------------------------------------------------------- #


def test_download_writes_the_file(tmp_path):
    dest = tmp_path / "a" / "payload.bin"
    download("http://example.invalid/x", dest, fetcher=good_fetcher, progress=False)
    assert dest.read_bytes() == PAYLOAD


def test_failed_download_leaves_nothing_behind(tmp_path):
    """The core of the storage contract.

    A killed download that looks complete on the next run costs an afternoon of
    confusing failures — the destination must be all-or-nothing, and the partial
    file must not survive.
    """
    dest = tmp_path / "payload.bin"
    with pytest.raises(ConnectionError):
        download("http://example.invalid/x", dest, fetcher=failing_fetcher, progress=False)

    assert not dest.exists()
    assert not (tmp_path / "payload.bin.part").exists()
    assert list(tmp_path.iterdir()) == []


def test_checksum_failure_does_not_publish_the_file(tmp_path):
    dest = tmp_path / "payload.bin"
    with pytest.raises(DownloadError, match="checksum mismatch"):
        download(
            "http://example.invalid/x",
            dest,
            fetcher=good_fetcher,
            expected_sha256="1" * 64,
            progress=False,
        )
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_not_redownloaded(tmp_path):
    dest = tmp_path / "payload.bin"
    dest.write_bytes(PAYLOAD)

    calls = []

    def counting_fetcher(url, handle):
        calls.append(url)
        handle.write(PAYLOAD)
        return len(PAYLOAD)

    download("http://example.invalid/x", dest, fetcher=counting_fetcher, progress=False)
    assert calls == []

    download("http://example.invalid/x", dest, fetcher=counting_fetcher, progress=False, force=True)
    assert calls == ["http://example.invalid/x"]


def test_existing_file_with_wrong_checksum_raises(tmp_path):
    dest = tmp_path / "payload.bin"
    dest.write_bytes(b"stale contents")
    with pytest.raises(DownloadError, match="checksum mismatch"):
        download(
            "http://example.invalid/x",
            dest,
            fetcher=good_fetcher,
            expected_sha256=PAYLOAD_SHA,
            progress=False,
        )


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def make_tar(tmp_path, name="sample.tar.gz"):
    inner = tmp_path / f"inner-{name}"
    inner.mkdir()
    (inner / "one.jpg").write_bytes(b"a")
    (inner / "two.jpg").write_bytes(b"b")
    archive = tmp_path / name
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(inner / "one.jpg", arcname="ccpd_base/one.jpg")
        tf.add(inner / "two.jpg", arcname="ccpd_base/two.jpg")
    return archive


def test_extract_writes_a_completion_marker(tmp_path):
    archive = make_tar(tmp_path)
    dest = tmp_path / "out"

    assert is_extracted(dest) is False
    extract_archive(archive, dest)

    assert (dest / "ccpd_base" / "one.jpg").exists()
    assert is_extracted(dest) is True
    marker = read_marker(dest)
    assert marker["complete"] is True
    assert marker["archive"] == archive.name
    assert marker["member_count"] == 2


def test_directory_without_a_marker_is_not_considered_extracted(tmp_path):
    """Existence of the destination proves nothing; only the marker does."""
    archive = make_tar(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "ccpd_base").mkdir()
    (dest / "ccpd_base" / "one.jpg").write_bytes(b"truncated leftovers")

    assert is_extracted(dest, archive) is False
    extract_archive(archive, dest)
    assert is_extracted(dest, archive) is True


def test_second_extraction_is_a_no_op(tmp_path):
    archive = make_tar(tmp_path)
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    stamp = marker_path(dest).read_text()

    (dest / "ccpd_base" / "one.jpg").write_bytes(b"locally modified")
    extract_archive(archive, dest)
    assert (dest / "ccpd_base" / "one.jpg").read_bytes() == b"locally modified"
    assert marker_path(dest).read_text() == stamp

    extract_archive(archive, dest, force=True)
    assert (dest / "ccpd_base" / "one.jpg").read_bytes() == b"a"


def test_marker_names_the_archive_it_came_from(tmp_path):
    first = make_tar(tmp_path, "first.tar.gz")
    second = make_tar(tmp_path, "second.tar.gz")
    dest = tmp_path / "out"
    extract_archive(first, dest)
    assert is_extracted(dest, first) is True
    assert is_extracted(dest, second) is False


def test_zip_extraction(tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train/img.jpg", "x")
    dest = tmp_path / "out"
    extract_archive(archive, dest)
    assert (dest / "train" / "img.jpg").exists()
    assert is_extracted(dest) is True


def test_path_traversal_is_refused(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(ExtractionError, match="escapes the destination"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_unsupported_archive_type(tmp_path):
    bogus = tmp_path / "data.7z"
    bogus.write_bytes(b"x")
    with pytest.raises(ExtractionError, match="unsupported archive type"):
        extract_archive(bogus, tmp_path / "out")


def test_missing_archive(tmp_path):
    with pytest.raises(ExtractionError, match="does not exist"):
        extract_archive(tmp_path / "nope.tar.gz", tmp_path / "out")


# --------------------------------------------------------------------------- #
# Walking
# --------------------------------------------------------------------------- #


def test_iter_files_is_sorted_and_filtered(tmp_path):
    (tmp_path / "b").mkdir()
    for name in ("z.jpg", "a.JPEG", "notes.txt", "c.png"):
        (tmp_path / "b" / name).write_bytes(b"x")

    found = [p.name for p in iter_files(tmp_path)]
    assert found == ["a.JPEG", "c.png", "z.jpg"]


def test_iter_files_on_a_missing_directory_is_empty(tmp_path):
    assert list(iter_files(tmp_path / "nope")) == []
