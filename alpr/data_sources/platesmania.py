"""Platesmania — manually browsed close-ups, one attribution backlink per image.

The site's terms require an active backlink for **every image used**. That makes the
backlink a licence condition rather than a courtesy, so this module is built around
a single rule:

    An image without a recorded source URL is refused. It never enters the index.

The mechanism is a sidecar file the human fills in while browsing. Downloading the
images is not automated — the site is rate-limited and manual browsing is what its
terms contemplate — so the tool's job is to make the *provenance* impossible to
forget, not to make the collection fast.

These images are for the real test set and for glyph close-ups of the rare letters
(`Q`, `W`, `X`) that a Colombian parking lot will not supply. They are **not**
commercially usable, so any checkpoint that trains on them is research-only. In
practice they should be used as evaluation and glyph reference, not training data.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from alpr.data_sources.base import AcquisitionError, PlateType, Record, Status
from alpr.data_sources.imagesize import safe_image_size
from alpr.data_sources.registry import (
    AccessMode,
    FetchResult,
    Purpose,
    Source,
    SourceInfo,
    register,
)
from alpr.data_sources.storage import IMAGE_SUFFIXES, iter_files

SIDECAR_NAME = "sources.csv"
SIDECAR_HEADER = ("filename", "source_url", "plate", "plate_type", "notes")


class SidecarError(AcquisitionError):
    """The sidecar file is missing, malformed, or incomplete."""


@dataclass(frozen=True, slots=True)
class SidecarEntry:
    filename: str
    source_url: str
    plate: str | None = None
    plate_type: PlateType = PlateType.PARTICULAR
    notes: str = ""


def read_sidecar(path: Path | str) -> dict[str, SidecarEntry]:
    """Read `sources.csv` into `{filename: entry}`.

    Rows missing a source URL raise rather than being dropped silently: a quietly
    skipped row looks identical to an image you forgot to photograph, and the whole
    point of this file is that omissions are loud.
    """
    path = Path(path)
    if not path.exists():
        raise SidecarError(
            f"{path} does not exist.\n"
            "Platesmania images require a per-image attribution backlink, so the "
            "sidecar is mandatory. Create it with `acquire platesmania-template`."
        )
    entries: dict[str, SidecarEntry] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise SidecarError(f"{path} has no header row; expected {list(SIDECAR_HEADER)}")
        missing = {"filename", "source_url"} - {f.strip() for f in reader.fieldnames}
        if missing:
            raise SidecarError(f"{path} is missing required column(s): {sorted(missing)}")

        for lineno, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            url = (row.get("source_url") or "").strip()
            if not url:
                raise SidecarError(
                    f"{path}:{lineno}: {filename!r} has no source_url.\n"
                    "The backlink is a licence condition. Either record the page the "
                    "image came from, or delete the image — it cannot be used."
                )
            if not url.startswith(("http://", "https://")):
                raise SidecarError(f"{path}:{lineno}: source_url {url!r} is not a URL")
            plate = (row.get("plate") or "").strip().upper() or None
            raw_type = (row.get("plate_type") or "").strip().lower()
            try:
                plate_type = PlateType(raw_type) if raw_type else PlateType.PARTICULAR
            except ValueError:
                raise SidecarError(
                    f"{path}:{lineno}: unknown plate_type {raw_type!r}; "
                    f"expected one of {[t.value for t in PlateType]}"
                ) from None
            entries[filename] = SidecarEntry(
                filename=filename,
                source_url=url,
                plate=plate,
                plate_type=plate_type,
                notes=(row.get("notes") or "").strip(),
            )
    return entries


def write_sidecar_template(path: Path | str, filenames: list[str]) -> Path:
    """Write a sidecar stub listing images that have no provenance yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SIDECAR_HEADER)
        for name in filenames:
            writer.writerow([name, "", "", PlateType.PARTICULAR.value, ""])
    return path


@register
class PlatesmaniaSource(Source):
    """Manually collected Platesmania images, attribution enforced."""

    info = SourceInfo(
        name="platesmania",
        title="Platesmania (manual, attribution required)",
        license_key="platesmania-terms",
        access=AccessMode.MANUAL,
        purposes=frozenset({Purpose.REAL_TEST, Purpose.GLYPHS}),
        url="https://platesmania.com/",
        size_hint="~100 images",
        notes="Per-image backlink required by the site terms. Not commercially usable.",
    )

    last_index_skipped: int = 0
    last_index_errors: list[str] = []

    @property
    def sidecar_path(self) -> Path:
        return self.raw_dir / SIDECAR_NAME

    def fetch(self, *, force: bool = False) -> FetchResult:
        """Never downloads. Reports collection progress and sidecar completeness."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        images = [p for p in iter_files(self.raw_dir, IMAGE_SUFFIXES)]
        if not images:
            return FetchResult(
                source=self.name,
                status=Status.MANUAL_REQUIRED,
                message=self.manual_instructions(),
            )
        try:
            entries = read_sidecar(self.sidecar_path)
        except SidecarError as exc:
            missing = [p.name for p in images]
            template = self.sidecar_path.with_suffix(".template.csv")
            write_sidecar_template(template, missing)
            return FetchResult(
                source=self.name,
                status=Status.MANUAL_REQUIRED,
                message=(
                    f"{exc}\n\nA template listing all {len(missing)} image(s) has been "
                    f"written to:\n  {template}\nFill in source_url for each, then rename "
                    f"it to {SIDECAR_NAME}."
                ),
            )
        unattributed = [p.name for p in images if p.name not in entries]
        if unattributed:
            return FetchResult(
                source=self.name,
                status=Status.MANUAL_REQUIRED,
                message=(
                    f"{len(unattributed)} image(s) are missing from {SIDECAR_NAME} and "
                    f"will be REFUSED at index time:\n  "
                    + "\n  ".join(unattributed[:10])
                    + ("\n  ..." if len(unattributed) > 10 else "")
                ),
            )
        return FetchResult(
            source=self.name,
            status=Status.READY,
            message=f"{len(images)} image(s), all attributed",
            paths=[str(self.raw_dir)],
        )

    def manual_instructions(self) -> str:
        return f"""\
Platesmania images are collected by hand, one at a time.

  1. Browse https://platesmania.com/co/ and find Colombian particular plates —
     prioritise close-ups containing Q, W and X, which are rare in Colombian
     series and hard to find in a parking lot (plan, Sprint 2.2 risks).
  2. Save each image into:  {self.raw_dir}
  3. Record its page URL in:  {self.sidecar_path}
       filename,source_url,plate,plate_type,notes
       co_00123.jpg,https://platesmania.com/co/nomer12345678,ABC123,particular,

The backlink is a LICENCE CONDITION, not a courtesy. Any image without a
source_url is refused at index time and never reaches the corpus.

Rate-limit yourself: this is a small site and the plan budgets ~100 images.

LICENCE: site terms, not commercially usable. Use these for evaluation and glyph
reference; do not train a deployable model on them."""

    def index(self) -> Iterator[Record]:
        """Yield only attributed images. Unattributed ones are counted and refused."""
        images = list(iter_files(self.raw_dir, IMAGE_SUFFIXES))
        if not images:
            return
        entries = read_sidecar(self.sidecar_path)

        refused: list[str] = []
        for image_path in images:
            entry = entries.get(image_path.name)
            if entry is None:
                refused.append(image_path.name)
                continue
            yield Record(
                source=self.name,
                image_path=self.paths.relative(image_path),
                license_key=self.info.license_key,
                plate=entry.plate,
                plate_type=entry.plate_type,
                image_size=safe_image_size(image_path),
                attribution_url=entry.source_url,  # without this, Record refuses to exist
                tags=("platesmania", "non_commercial"),
                extra={"notes": entry.notes} if entry.notes else {},
            )

        self.last_index_skipped = len(refused)
        self.last_index_errors = [
            f"{name}: no source_url in {SIDECAR_NAME} — refused (attribution is a "
            "licence condition)"
            for name in refused[:20]
        ]
