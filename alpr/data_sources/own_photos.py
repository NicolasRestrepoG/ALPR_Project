"""Your own photographs — the only irreplaceable data in the project.

Everything else can be re-downloaded. These cannot: ~240 test-set images that become
**the real test set** the whole sim2real loop is measured against, and ~20 close-ups
that become the 36 extracted glyphs. If the schedule slips, this is the thing that
must not be dropped (plan §23, R9: incompressible).

Two directories, two very different jobs:

    raw/own_photos/testset/     ~240 plates, deliberately varied → THE 400
    raw/own_photos/close_ups/   ~20 sharp, square-on plates → glyph source

`labels.csv` does not exist yet. Phase 2's annotation tool (Sprint 2.1) writes it,
and this module joins it when it appears — so the same source works before and after
labelling, and `fetch` doubles as a progress report against the exit-gate targets.

The column layout is defined here because this module is the consumer; the Phase 2
tool writes to this contract.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from alpr.data_sources.base import (
    AcquisitionError,
    Corners,
    PlateType,
    Record,
    Status,
    order_corners,
)
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

LABELS_NAME = "labels.csv"

#: Written by the Sprint 2.1 annotation tool. Corners are TL, TR, BR, BL.
LABELS_HEADER = (
    "filename",
    "plate",
    "plate_type",
    "tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y",
    "tags",
    "consent",
    "session",
)  # fmt: skip

#: Sprint 2.1 exit-gate targets, checked by `fetch` so the shoot can be steered
#: while there is still time to reshoot.
TARGETS = {
    "positives": 200,
    "negatives": 40,
    "night_fraction": 0.15,
    "high_yaw_fraction": 0.25,
    "small_plate_fraction": 0.20,
}


class LabelsError(AcquisitionError):
    """`labels.csv` is malformed."""


@dataclass(frozen=True, slots=True)
class Label:
    filename: str
    plate: str | None
    plate_type: PlateType
    corners: Corners | None
    tags: tuple[str, ...]
    consent: str = ""
    session: str = ""


def _corner_fields(row: dict[str, str]) -> Corners | None:
    keys = ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y")
    values = [(row.get(k) or "").strip() for k in keys]
    if not any(values):
        return None
    if not all(values):
        raise LabelsError(f"{row.get('filename')!r}: partial corner columns {values}")
    try:
        nums = [float(v) for v in values]
    except ValueError as exc:
        raise LabelsError(f"{row.get('filename')!r}: non-numeric corner value ({exc})") from None
    pts = [(nums[i], nums[i + 1]) for i in range(0, 8, 2)]
    return order_corners(pts)


def read_labels(path: Path | str) -> dict[str, Label]:
    """Read the annotation tool's output. Missing file → empty mapping, not an error."""
    path = Path(path)
    if not path.exists():
        return {}
    labels: dict[str, Label] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise LabelsError(f"{path}: missing a 'filename' column")
        for lineno, row in enumerate(reader, start=2):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            raw_type = (row.get("plate_type") or "").strip().lower()
            try:
                plate_type = PlateType(raw_type) if raw_type else PlateType.PARTICULAR
            except ValueError:
                raise LabelsError(f"{path}:{lineno}: unknown plate_type {raw_type!r}") from None
            try:
                corners = _corner_fields(row)
            except AcquisitionError as exc:
                raise LabelsError(f"{path}:{lineno}: {exc}") from exc
            plate = (row.get("plate") or "").strip().upper() or None
            if plate_type is PlateType.PARTICULAR and plate and len(plate) != 6:
                raise LabelsError(
                    f"{path}:{lineno}: particular plate {plate!r} is not 6 characters "
                    "(format is LLLDDD)"
                )
            labels[filename] = Label(
                filename=filename,
                plate=plate,
                plate_type=plate_type,
                corners=corners,
                tags=tuple(t for t in (row.get("tags") or "").split("|") if t.strip()),
                consent=(row.get("consent") or "").strip(),
                session=(row.get("session") or "").strip(),
            )
    return labels


@register
class OwnPhotosSource(Source):
    """The real test set and the glyph close-ups. Manual by nature."""

    info = SourceInfo(
        name="own_photos",
        title="Own photographs (real test set + glyph close-ups)",
        license_key="own-work",
        access=AccessMode.MANUAL,
        purposes=frozenset({Purpose.REAL_TEST, Purpose.GLYPHS}),
        url=None,
        size_hint="~240 test images + ~20 close-ups",
        notes="Irreplaceable. Everything else can be re-downloaded; these cannot.",
    )

    @property
    def testset_dir(self) -> Path:
        return self.raw_dir / "testset"

    @property
    def close_ups_dir(self) -> Path:
        return self.raw_dir / "close_ups"

    @property
    def labels_path(self) -> Path:
        return self.raw_dir / LABELS_NAME

    def fetch(self, *, force: bool = False) -> FetchResult:
        """Reports shoot progress against the Sprint 2.1 exit gate. Downloads nothing."""
        for directory in (self.testset_dir, self.close_ups_dir):
            directory.mkdir(parents=True, exist_ok=True)

        testset = list(iter_files(self.testset_dir, IMAGE_SUFFIXES))
        close_ups = list(iter_files(self.close_ups_dir, IMAGE_SUFFIXES))
        labels = read_labels(self.labels_path)

        if not testset and not close_ups:
            return FetchResult(
                source=self.name,
                status=Status.MANUAL_REQUIRED,
                message=self.manual_instructions(),
            )
        return FetchResult(
            source=self.name,
            status=Status.READY if labels else Status.MANUAL_REQUIRED,
            message=self.progress_report(testset, close_ups, labels),
            paths=[str(self.testset_dir), str(self.close_ups_dir)],
        )

    def progress_report(
        self, testset: list[Path], close_ups: list[Path], labels: dict[str, Label]
    ) -> str:
        labelled = [lb for lb in labels.values() if lb.corners is not None]
        positives = [lb for lb in labelled if lb.plate_type is PlateType.PARTICULAR]
        negatives = [lb for lb in labelled if lb.plate_type.is_negative]

        def frac(tag: str) -> float:
            if not labelled:
                return 0.0
            return sum(1 for lb in labelled if tag in lb.tags) / len(labelled)

        def check(ok: bool) -> str:
            return "OK " if ok else "-- "

        lines = [
            f"testset images: {len(testset)}    close-ups: {len(close_ups)}    "
            f"labelled: {len(labelled)}",
            "",
            "Sprint 2.1 exit gate:",
            f"  {check(len(positives) >= TARGETS['positives'])}"
            f"positives          {len(positives):>4} / {TARGETS['positives']}",
            f"  {check(len(negatives) >= TARGETS['negatives'])}"
            f"negatives          {len(negatives):>4} / {TARGETS['negatives']}",
            f"  {check(frac('night') >= TARGETS['night_fraction'])}"
            f"night              {frac('night'):>6.0%} / {TARGETS['night_fraction']:.0%}",
            f"  {check(frac('yaw_high') >= TARGETS['high_yaw_fraction'])}"
            f"|yaw| > 10 deg     {frac('yaw_high'):>6.0%} / {TARGETS['high_yaw_fraction']:.0%}",
            f"  {check(frac('small') >= TARGETS['small_plate_fraction'])}"
            f"plate < 100 px     {frac('small'):>6.0%} / {TARGETS['small_plate_fraction']:.0%}",
        ]
        if labelled:
            missing = self.missing_characters(labels)
            lines.append(
                f"  {check(not missing)}every character >= 5x"
                + (f"   missing/rare: {' '.join(sorted(missing))}" if missing else "")
            )
        if not labels:
            lines += [
                "",
                f"No {LABELS_NAME} yet — Phase 2's annotation tool (Sprint 2.1) writes it.",
                "Images are indexed without plate strings until then.",
            ]
        return "\n".join(lines)

    @staticmethod
    def missing_characters(labels: dict[str, Label], minimum: int = 5) -> set[str]:
        """Characters appearing fewer than `minimum` times across labelled plates.

        Phase 5 ranks errors per glyph. A character seen twice produces a ranking
        that is statistical noise, so this is an exit-gate item, not a nicety.
        """
        counts: dict[str, int] = {}
        for label in labels.values():
            for char in label.plate or "":
                counts[char] = counts.get(char, 0) + 1
        alphabet = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        return {c for c in alphabet if counts.get(c, 0) < minimum}

    def manual_instructions(self) -> str:
        return f"""\
No photographs yet. This is the one source nobody else can supply.

TEST SET -> {self.testset_dir}
  ~240 photographs of parked cars' FRONT plates. Targets that decide whether the
  numbers you report later mean anything:
    - >= 200 particular plates, >= 40 negatives (publico, moto, no-plate)
    - >= 15% at night, >= 25% at |yaw| > 10 deg, >= 20% with plate width < 100 px
    - every character A-Z and 0-9 appearing >= 5 times across the set
  THREE separate sessions at different hours. One car park at one time of day
  produces a model that looks excellent and fails at night.
  Deliberately include the ugly ones: dirty, framed, bolted, faded.

CLOSE-UPS -> {self.close_ups_dir}
  ~20 sharp, square-on, full-phone-resolution shots of plates, chosen together to
  cover all 36 characters at least twice. These become the extracted glyphs.
  Q, W and X are rare in Colombian series — hunt for them specifically, and top up
  from Platesmania if the car park will not supply them.
  Do NOT download a font instead. The typeface is not published; every substitute
  produces an accuracy ceiling that looks like a model bug for weeks.

PRIVACY (docs/legal_privacy_co.md)
  Frame plates, not people. Where a vehicle is identifiable to a specific
  resident, get written permission and record it in the `consent` column.

Then: `acquire fetch own_photos` reports progress against the targets above, and
Phase 2's annotation tool writes {LABELS_NAME}."""

    def index(self) -> Iterator[Record]:
        labels = read_labels(self.labels_path)
        for kind, directory in (("testset", self.testset_dir), ("close_up", self.close_ups_dir)):
            for image_path in iter_files(directory, IMAGE_SUFFIXES):
                label = labels.get(image_path.name)
                tags = (kind,) + (label.tags if label else ())
                extra: dict[str, str] = {}
                if label and label.consent:
                    extra["consent"] = label.consent
                if label and label.session:
                    extra["session"] = label.session
                yield Record(
                    source=self.name,
                    image_path=self.paths.relative(image_path),
                    license_key=self.info.license_key,
                    plate=label.plate if label else None,
                    plate_type=label.plate_type if label else PlateType.UNKNOWN,
                    corners=label.corners if label else None,
                    image_size=safe_image_size(image_path),
                    tags=tags,
                    extra=extra,
                )
