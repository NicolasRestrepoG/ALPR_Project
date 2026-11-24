"""Source registration and filtering.

Sources are heterogeneous in *kind*, not just in URL: one is a 12 GB Drive archive
with annotations in the filename, one needs an API key, two require emailing a
signed agreement and waiting days, one requires manual browsing plus a per-image
attribution backlink. The registry is what lets one command speak to all of them,
and what lets `--commercial-only` mean something enforceable.

The important default: anything whose terms have not been read carries the
`unknown` licence, and `unknown` is not commercially usable. Forgetting to check a
licence therefore *excludes* the source rather than silently including it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from alpr.data_sources.base import (
    AcquisitionError,
    License,
    Record,
    Status,
    get_license,
    write_jsonl,
)
from alpr.data_sources.storage import DataPaths


class SourceError(AcquisitionError):
    """A source could not be fetched or indexed."""


class AccessMode(StrEnum):
    """How the bytes are obtained — this is what decides whether `--all` can work."""

    AUTOMATIC = "automatic"  # a URL and nothing else
    API_KEY = "api_key"  # free key in an environment variable
    MANUAL = "manual"  # a human browses, downloads, or photographs
    GATED = "gated"  # signed agreement emailed, multi-day lead time


class Purpose(StrEnum):
    """What a source is *for*. Filtering by purpose keeps later phases honest."""

    DETECTOR_PRETRAIN = "detector_pretrain"
    DETECTOR_FINETUNE = "detector_finetune"
    RECTIFICATION = "rectification"
    RECOGNIZER_EVAL = "recognizer_eval"
    REAL_TEST = "real_test"
    BACKGROUNDS = "backgrounds"
    GLYPHS = "glyphs"
    NEGATIVES = "negatives"
    ASSETS = "assets"


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Everything the orchestrator needs to know without importing the source."""

    name: str
    title: str
    license_key: str
    access: AccessMode
    purposes: frozenset[Purpose]
    url: str | None = None
    size_hint: str = ""
    notes: str = ""

    @property
    def license(self) -> License:
        return get_license(self.license_key)

    @property
    def commercial_ok(self) -> bool:
        return self.license.commercial_ok

    @property
    def automatic(self) -> bool:
        """True if `acquire fetch --all` can obtain it unattended."""
        return self.access in (AccessMode.AUTOMATIC, AccessMode.API_KEY)


@dataclass(slots=True)
class FetchResult:
    """What happened when a source was fetched."""

    source: str
    status: Status
    message: str = ""
    paths: list[str] = field(default_factory=list)
    bytes_written: int = 0

    @property
    def ok(self) -> bool:
        return self.status in (Status.READY, Status.DOWNLOADED, Status.SKIPPED)

    def __str__(self) -> str:
        head = f"[{self.status.value}] {self.source}"
        return f"{head}: {self.message}" if self.message else head


class Source(ABC):
    """One dataset, one module. Subclasses declare `info` and implement `fetch`/`index`."""

    info: ClassVar[SourceInfo]

    def __init__(self, paths: DataPaths | None = None) -> None:
        self.paths = paths or DataPaths.default()

    # -- interface -------------------------------------------------------- #

    @abstractmethod
    def fetch(self, *, force: bool = False) -> FetchResult:
        """Obtain the raw files. Must be idempotent and safe to interrupt."""

    def index(self) -> Iterator[Record]:
        """Yield normalized records for whatever is on disk. Default: nothing."""
        return iter(())

    # -- provided --------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def raw_dir(self):
        return self.paths.raw(self.info.name)

    @property
    def index_path(self):
        return self.paths.index(self.info.name)

    def write_index(self) -> int:
        """Index the source and write `data/index/<name>.jsonl`. Returns the count."""
        return write_jsonl(self.index_path, self.index())

    def manual_instructions(self) -> str:
        """Human steps for a MANUAL or GATED source. Printed instead of raising."""
        return self.info.notes or "No instructions recorded for this source."


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

_REGISTRY: dict[str, type[Source]] = {}


def register(cls: type[Source]) -> type[Source]:
    """Class decorator that registers a source under `cls.info.name`."""
    info = getattr(cls, "info", None)
    if not isinstance(info, SourceInfo):
        raise SourceError(f"{cls.__name__} must define a class-level `info: SourceInfo`")
    if info.name in _REGISTRY:
        existing = _REGISTRY[info.name]
        if existing is not cls:
            raise SourceError(
                f"duplicate source name {info.name!r}: "
                f"{existing.__module__}.{existing.__qualname__} vs "
                f"{cls.__module__}.{cls.__qualname__}"
            )
    _REGISTRY[info.name] = cls
    return cls


def get(name: str) -> type[Source]:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise SourceError(
            f"unknown source {name!r}. Known sources: {', '.join(sorted(_REGISTRY))}"
        ) from None


def all_infos() -> list[SourceInfo]:
    return [cls.info for cls in sorted(_REGISTRY.values(), key=lambda c: c.info.name)]


def select(
    *,
    names: Iterable[str] | None = None,
    commercial_only: bool = False,
    access: AccessMode | None = None,
    purpose: Purpose | None = None,
    skip_manual: bool = False,
) -> list[type[Source]]:
    """Filter registered sources. All criteria are ANDed."""
    chosen: list[type[Source]] = (
        [get(n) for n in names] if names is not None else list(_REGISTRY.values())
    )
    out = []
    for cls in chosen:
        info = cls.info
        if commercial_only and not info.commercial_ok:
            continue
        if access is not None and info.access is not access:
            continue
        if purpose is not None and purpose not in info.purposes:
            continue
        if skip_manual and not info.automatic:
            continue
        out.append(cls)
    return sorted(out, key=lambda c: c.info.name)


def commercial_blockers(names: Iterable[str] | None = None) -> list[SourceInfo]:
    """Sources that would block commercial deployment — the answer `--audit` needs."""
    infos = [cls.info for cls in select(names=names)]
    return sorted(
        (i for i in infos if not i.commercial_ok),
        key=lambda i: i.name,
    )


def clear_registry() -> None:
    """Test hook. Never call this from application code."""
    _REGISTRY.clear()
