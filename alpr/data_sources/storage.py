"""The storage contract: where files live, how they arrive, how we know they're whole.

Two properties matter more than anything else here.

**Downloads are atomic.** Every download is written to a sibling `.part` file and
moved into place only on success. A killed 12 GB download that looks complete on the
next run costs an afternoon of confusing failures, and the failure surfaces far from
its cause — as a corrupt archive, or worse, as a truncated one that extracts fine.

**Extractions leave a completion marker.** The marker records what was extracted,
from which archive, and how many members. Presence of the destination directory
proves nothing; presence of the marker proves the extraction ran to the end.

Third-party imports (`requests`, `tqdm`, `gdown`) are all lazy, inside the function
that needs them, so the test suite runs without them installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpr.config import get_secret, repo_root
from alpr.data_sources.base import AcquisitionError

MARKER_NAME = ".extracted.json"
_CHUNK = 1 << 20  # 1 MiB


class DownloadError(AcquisitionError):
    """A download failed, was truncated, or failed its checksum."""


class ExtractionError(AcquisitionError):
    """An archive could not be extracted, or contained unsafe member paths."""


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def default_data_root() -> Path:
    """`ALPR_DATA_ROOT` if set (environment or `.env`), else `<repo>/data`."""
    configured = get_secret("ALPR_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return repo_root() / "data"


@dataclass(frozen=True, slots=True)
class DataPaths:
    """The one place that knows the on-disk layout (plan §6).

    data/
    ├── raw/<source>/          files exactly as published
    ├── index/<source>.jsonl   normalized records
    ├── assets/                municipality list, spec PDF, glyph images
    ├── corpus/<config-hash>/  generated synthetic samples   [phase 3]
    ├── real_test/             THE 400 — read-only, checksummed
    └── manifest.json          acquisition record + licence lineage audit
    """

    root: Path

    @classmethod
    def default(cls) -> DataPaths:
        return cls(default_data_root())

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    # -- directories ------------------------------------------------------ #

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def corpus_dir(self) -> Path:
        return self.root / "corpus"

    @property
    def real_test_dir(self) -> Path:
        return self.root / "real_test"

    @property
    def tmp_dir(self) -> Path:
        return self.root / ".tmp"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # -- per source ------------------------------------------------------- #

    def raw(self, source: str) -> Path:
        return self.raw_dir / source

    def index(self, source: str) -> Path:
        return self.index_dir / f"{source}.jsonl"

    def relative(self, path: Path | str) -> str:
        """POSIX path relative to the data root — the form stored in a Record."""
        p = Path(path)
        if not p.is_absolute():
            return p.as_posix()
        return p.resolve().relative_to(self.root.resolve()).as_posix()

    def ensure(self) -> DataPaths:
        for d in (self.raw_dir, self.index_dir, self.assets_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path | str, expected: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise DownloadError(
            f"checksum mismatch for {path}\n"
            f"  expected sha256 {expected}\n"
            f"  actual   sha256 {actual}"
        )


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp creates 0600. These are shared project assets, not secrets, and
        # a 0600 municipality list is a surprise waiting to happen on a second account.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_text(path: Path | str, text: str, *, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _stream_to(url: str, handle: Any, *, progress: bool, timeout: float) -> int:
    """Stream `url` into an open binary handle. Returns bytes written."""
    try:
        import requests  # noqa: PLC0415 — lazy: base layer must import without it
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise DownloadError(
            "requests is not installed. Install the Phase 1 dependencies:\n"
            "  pip install -e '.[dev]'"
        ) from exc

    written = 0
    with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        bar = None
        if progress:
            try:
                from tqdm import tqdm  # noqa: PLC0415 — lazy, optional

                bar = tqdm(total=total or None, unit="B", unit_scale=True, unit_divisor=1024)
            except ImportError:
                bar = None
        try:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        finally:
            if bar is not None:
                bar.close()
    if total and written != total:
        raise DownloadError(
            f"truncated download: got {written} bytes, Content-Length said {total} ({url})"
        )
    return written


def download(
    url: str,
    dest: Path | str,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
    progress: bool = True,
    timeout: float = 60.0,
    fetcher: Callable[[str, Any], int] | None = None,
) -> Path:
    """Download `url` to `dest` atomically. Returns `dest`.

    Skips the download when `dest` already exists (and matches `expected_sha256`,
    if given). Writes to `dest.part`, fsyncs, then `os.replace` — so `dest` either
    does not exist or is a complete file, never anything in between. On any failure
    the partial file is removed rather than left to be mistaken for a good one.

    `fetcher` exists so tests can drive the atomicity logic without a network.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        if expected_sha256:
            verify_checksum(dest, expected_sha256)
        return dest

    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    try:
        with part.open("wb") as fh:
            if fetcher is not None:
                fetcher(url, fh)
            else:
                _stream_to(url, fh, progress=progress, timeout=timeout)
            fh.flush()
            os.fsync(fh.fileno())
        if expected_sha256:
            verify_checksum(part, expected_sha256)
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest


def download_gdrive(file_id: str, dest: Path | str, *, force: bool = False) -> Path:
    """Download a Google Drive file by id via `gdown`, atomically.

    `gdown` handles Drive's large-file confirmation token, which plain HTTP does not.
    Drive rate-limits popular files; the error message names the mirrors so a
    failure at 2 a.m. is self-explanatory.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        return dest
    try:
        import gdown  # noqa: PLC0415 — lazy, heavy
    except ImportError as exc:
        raise DownloadError(
            "gdown is not installed. Install the Phase 1 dependencies:\n  pip install -e '.[dev]'"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.unlink(missing_ok=True)
    try:
        out = gdown.download(id=file_id, output=str(part), quiet=False, resume=True)
        if out is None or not part.exists() or part.stat().st_size == 0:
            raise DownloadError(
                f"gdown returned no file for Drive id {file_id}.\n"
                "Google Drive rate-limits this file when many people fetch it. Options:\n"
                "  1. retry in a few hours;\n"
                "  2. OpenDataLab mirror — https://opendatalab.com/OpenDataLab/CCPD\n"
                "  3. BaiduYun (code hm0u) — https://pan.baidu.com/s/1i5AOjAbtkwb17Zy-NQGqkw\n"
                "  4. download by hand and drop the archive at the destination path above."
            )
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def marker_path(dest_dir: Path | str) -> Path:
    return Path(dest_dir) / MARKER_NAME


def read_marker(dest_dir: Path | str) -> dict[str, Any] | None:
    p = marker_path(dest_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_extracted(dest_dir: Path | str, archive: Path | str | None = None) -> bool:
    """True only if a completion marker exists (and names `archive`, if given)."""
    marker = read_marker(dest_dir)
    if marker is None:
        return False
    if archive is not None and marker.get("archive") != Path(archive).name:
        return False
    return bool(marker.get("complete"))


def _safe_members(names: list[str], dest: Path) -> None:
    """Reject absolute paths and `..` traversal before writing anything."""
    dest = dest.resolve()
    for name in names:
        target = (dest / name).resolve()
        if not str(target).startswith(str(dest)):
            raise ExtractionError(f"archive member escapes the destination directory: {name!r}")


def extract_archive(
    archive: Path | str,
    dest_dir: Path | str,
    *,
    force: bool = False,
    select: Callable[[str], bool] | None = None,
    delete_archive: bool = False,
    progress_every: int = 0,
) -> Path:
    """Extract a .tar.xz / .tar.gz / .tar / .zip into `dest_dir`, once.

    Writes a completion marker on success. Re-running is a no-op unless `force`.
    Extraction happens in place rather than via a staging directory: the CCPD
    archive is 13 GB and a copy would double both the time and the disk needed.
    The marker, not the directory's existence, is what makes that safe.

    `select` filters members by name, which is how a 13 GB archive lands on a disk
    that cannot hold it whole. Tar archives are read as a *stream* (`r|*`) so
    filtering costs one sequential pass rather than a full member scan.

    `delete_archive` removes the archive once extraction has completed and the
    marker is written — never before, so an interrupted run is always resumable
    from the archive.
    """
    archive = Path(archive)
    dest_dir = Path(dest_dir)
    if not archive.exists():
        raise ExtractionError(f"archive does not exist: {archive}")

    if is_extracted(dest_dir, archive) and not force:
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    marker_path(dest_dir).unlink(missing_ok=True)
    started = time.time()
    resolved_dest = dest_dir.resolve()

    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            names = [n for n in zf.namelist() if select is None or select(n)]
            _safe_members(names, dest_dir)
            zf.extractall(dest_dir, members=names)
            member_count = len(names)
    elif ".tar" in name or name.endswith((".tgz", ".txz")):
        member_count = 0
        # Stream mode: members arrive one at a time and are never seeked back to,
        # so a 13 GB xz archive is decompressed exactly once.
        with tarfile.open(archive, "r|*") as tf:
            for member in tf:
                if select is not None and not select(member.name):
                    continue
                target = (dest_dir / member.name).resolve()
                if not str(target).startswith(str(resolved_dest)):
                    raise ExtractionError(
                        f"archive member escapes the destination directory: {member.name!r}"
                    )
                # filter="data" refuses device nodes, absolute paths and links out
                # of the tree. Default in 3.14, passed explicitly to pin behaviour.
                tf.extract(member, dest_dir, filter="data")
                member_count += 1
                if progress_every and member_count % progress_every == 0:
                    elapsed = time.time() - started
                    print(
                        f"  extracted {member_count:,} members in {elapsed / 60:.1f} min",
                        flush=True,
                    )
    else:
        raise ExtractionError(f"unsupported archive type: {archive.name}")

    atomic_write_text(
        marker_path(dest_dir),
        json.dumps(
            {
                "complete": True,
                "archive": archive.name,
                "archive_bytes": archive.stat().st_size,
                "member_count": member_count,
                "selective": select is not None,
                "seconds": round(time.time() - started, 1),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n",
    )
    if delete_archive:
        archive.unlink(missing_ok=True)
    return dest_dir


# --------------------------------------------------------------------------- #
# Walking
# --------------------------------------------------------------------------- #

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def iter_files(root: Path | str, suffixes: frozenset[str] = IMAGE_SUFFIXES) -> Iterator[Path]:
    """Yield files under `root` with a matching suffix, in sorted order.

    Sorted so that indexing is deterministic: the same tree produces a
    byte-identical index, which makes the manifest's checksums meaningful.
    """
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def dir_size(root: Path | str) -> int:
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def free_space(path: Path | str) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free
