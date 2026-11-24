"""The manifest and the licence lineage audit.

`data/manifest.json` answers three questions that get progressively harder to answer
by hand as the project ages:

1. **What is on disk, and is it still what we think it is?** (`verify`)
2. **Where did it come from and under what terms?** (per-source licence, recorded at
   acquisition, never reconstructed)
3. **Can a model trained on this be commercially deployed, and if not, what exactly
   is blocking it?** (`audit`)

Question 3 is the one that matters. Six months in, nobody remembers whether a
checkpoint touched UFPR. The audit is trivial to build now and impossible to
reconstruct later, which is why it exists in Phase 1 rather than Phase 8.

The manifest describes *datasets*. Per-checkpoint lineage — config hash + corpus
hash + every dataset touched — arrives in Phase 4, when there is a checkpoint to
attach it to, and will reference these same source names.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from alpr.data_sources.base import (
    AcquisitionError,
    Status,
    get_license,
    read_jsonl,
)
from alpr.data_sources.registry import Source, SourceInfo, select
from alpr.data_sources.storage import DataPaths, atomic_write_text, sha256_file

MANIFEST_VERSION = 1


class ManifestError(AcquisitionError):
    """The manifest is missing, malformed, or disagrees with the disk."""


@dataclass(slots=True)
class SourceEntry:
    """One source's acquisition record."""

    name: str
    title: str
    license_key: str
    license_name: str
    commercial_ok: bool
    access: str
    purposes: list[str]
    status: str = Status.NOT_STARTED.value
    file_count: int = 0
    bytes_on_disk: int = 0
    record_count: int = 0
    index_sha256: str | None = None
    url: str | None = None
    notes: str = ""
    fetched_at: str | None = None

    @classmethod
    def from_info(cls, info: SourceInfo) -> SourceEntry:
        lic = info.license
        return cls(
            name=info.name,
            title=info.title,
            license_key=info.license_key,
            license_name=lic.name,
            commercial_ok=lic.commercial_ok,
            access=info.access.value,
            purposes=sorted(p.value for p in info.purposes),
            url=info.url,
            notes=info.notes,
        )


@dataclass(slots=True)
class Manifest:
    """What has been acquired, when, and under what terms."""

    version: int = MANIFEST_VERSION
    generated_at: str = ""
    data_root: str = ""
    sources: dict[str, SourceEntry] = field(default_factory=dict)

    # -- serialization ---------------------------------------------------- #

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "data_root": self.data_root,
            "sources": {k: asdict(v) for k, v in sorted(self.sources.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Manifest:
        version = payload.get("version")
        if version != MANIFEST_VERSION:
            raise ManifestError(
                f"manifest version {version!r} is not supported (expected {MANIFEST_VERSION})"
            )
        try:
            sources = {k: SourceEntry(**v) for k, v in (payload.get("sources") or {}).items()}
        except TypeError as exc:
            raise ManifestError(f"malformed source entry in manifest: {exc}") from exc
        return cls(
            version=version,
            generated_at=payload.get("generated_at", ""),
            data_root=payload.get("data_root", ""),
            sources=sources,
        )

    def save(self, path: Path | str) -> Path:
        return atomic_write_text(
            path, json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        path = Path(path)
        if not path.exists():
            raise ManifestError(f"{path} does not exist — run `acquire manifest` first")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path} is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)

    # -- the audit -------------------------------------------------------- #

    @property
    def acquired(self) -> list[SourceEntry]:
        """Sources with something actually on disk — the only ones the audit judges."""
        return [e for e in self.sources.values() if e.file_count > 0 or e.record_count > 0]

    def blockers(self) -> list[SourceEntry]:
        """Acquired sources that would block commercial deployment."""
        return sorted(
            (e for e in self.acquired if not e.commercial_ok),
            key=lambda e: e.name,
        )

    def commercial_ok(self) -> bool:
        return not self.blockers()

    def audit_report(self) -> str:
        clean = sorted(
            (e for e in self.acquired if e.commercial_ok),
            key=lambda e: e.name,
        )
        blockers = self.blockers()
        pending = sorted(e.name for e in self.sources.values() if e not in self.acquired)

        lines = [
            "LICENCE LINEAGE AUDIT",
            f"generated {self.generated_at}   data root {self.data_root}",
            "",
            f"ON DISK, commercially clean ({len(clean)}):",
        ]
        lines += [
            f"  +  {e.name:<30} {e.license_name:<38} "
            f"{e.file_count:>7,} files  {e.record_count:>8,} records"
            for e in clean
        ] or ["  (none)"]

        lines += ["", f"ON DISK, NOT commercially usable ({len(blockers)}):"]
        if blockers:
            for e in blockers:
                lines.append(
                    f"  !  {e.name:<30} {e.license_name:<38} "
                    f"{e.file_count:>7,} files  {e.record_count:>8,} records"
                )
                note = get_license(e.license_key).note
                if note:
                    lines.append(f"       {note}")
        else:
            lines.append("  (none)")

        if pending:
            lines += ["", f"REGISTERED but not acquired ({len(pending)}): {', '.join(pending)}"]

        lines += [""]
        if blockers:
            names = ", ".join(e.name for e in blockers)
            lines += [
                f"COMMERCIAL DEPLOYMENT: NO — blocked by {names}",
                "",
                "To build the clean path, train only on sources listed above as clean:",
                "  acquire fetch --all --commercial-only",
                "  acquire index --all --commercial-only",
                "A checkpoint's lineage is decided by what it TRAINED on, not by what is",
                "on disk — Phase 4 records that per checkpoint. Presence here is a warning,",
                "not a verdict on any particular model.",
            ]
        else:
            lines += [
                "COMMERCIAL DEPLOYMENT: YES — every acquired source is commercially clean.",
                "",
                "Still outstanding for a clean V1 (not dataset issues):",
                "  - Ultralytics YOLOv8 is AGPL-3.0; removed in Sprint 6.3.",
                "  - CC-BY sources require attribution in the writeup.",
            ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Building and verifying
# --------------------------------------------------------------------------- #


def _measure(directory: Path) -> tuple[int, int]:
    """(file count, total bytes) under `directory`.

    Excludes dotfiles (extraction markers) and `.part` files: a download in flight
    is not acquired data, and counting it would make the manifest disagree with
    itself the moment the download finishes.
    """
    if not directory.exists():
        return 0, 0
    count = 0
    total = 0
    for path in directory.rglob("*"):
        if not path.is_file() or path.name.startswith(".") or path.name.endswith(".part"):
            continue
        count += 1
        total += path.stat().st_size
    return count, total


def build_manifest(
    paths: DataPaths | None = None,
    *,
    sources: Iterable[type[Source]] | None = None,
    previous: Manifest | None = None,
) -> Manifest:
    """Scan the disk and produce a manifest of what is actually there.

    `previous` carries `fetched_at` forward, so the manifest remembers when a source
    first arrived even though rebuilding it is a pure scan.
    """
    paths = paths or DataPaths.default()
    classes = list(sources) if sources is not None else select()
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    manifest = Manifest(
        generated_at=now,
        data_root=str(paths.root),
    )
    for cls in classes:
        info = cls.info
        entry = SourceEntry.from_info(info)
        file_count, byte_count = _measure(paths.raw(info.name))
        entry.file_count = file_count
        entry.bytes_on_disk = byte_count

        index_path = paths.index(info.name)
        if index_path.exists():
            entry.index_sha256 = sha256_file(index_path)
            entry.record_count = sum(1 for _ in read_jsonl(index_path))

        if entry.record_count:
            entry.status = Status.INDEXED.value
        elif entry.file_count:
            entry.status = Status.READY.value
        elif not info.automatic:
            entry.status = Status.MANUAL_REQUIRED.value
        else:
            entry.status = Status.NOT_STARTED.value

        prior = (previous.sources.get(info.name) if previous else None) or None
        if prior and prior.fetched_at:
            entry.fetched_at = prior.fetched_at
        if entry.file_count and not entry.fetched_at:
            entry.fetched_at = now

        manifest.sources[info.name] = entry
    return manifest


@dataclass(slots=True)
class VerifyResult:
    """Differences between the recorded manifest and the current disk."""

    problems: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        if self.ok:
            return f"manifest matches disk ({self.checked} source(s) checked)"
        return "\n".join(
            [f"{len(self.problems)} problem(s):", *(f"  - {p}" for p in self.problems)]
        )


def verify_manifest(manifest: Manifest, paths: DataPaths | None = None) -> VerifyResult:
    """Check the manifest against the disk: file counts, sizes, index checksums."""
    paths = paths or DataPaths.default()
    result = VerifyResult()
    current = build_manifest(paths, previous=manifest)

    for name, recorded in sorted(manifest.sources.items()):
        result.checked += 1
        actual = current.sources.get(name)
        if actual is None:
            result.problems.append(f"{name}: in the manifest but no longer registered")
            continue
        if actual.file_count != recorded.file_count:
            result.problems.append(
                f"{name}: {recorded.file_count:,} files recorded, {actual.file_count:,} on disk"
            )
        if actual.bytes_on_disk != recorded.bytes_on_disk:
            result.problems.append(
                f"{name}: {recorded.bytes_on_disk:,} bytes recorded, "
                f"{actual.bytes_on_disk:,} on disk"
            )
        if recorded.index_sha256 and actual.index_sha256 != recorded.index_sha256:
            missing = actual.index_sha256 is None
            result.problems.append(
                f"{name}: index {'is missing' if missing else 'has changed'} "
                f"(recorded {recorded.index_sha256[:12]})"
            )

    for name in sorted(set(current.sources) - set(manifest.sources)):
        if current.sources[name].file_count:
            result.problems.append(f"{name}: on disk but absent from the manifest")

    return result
