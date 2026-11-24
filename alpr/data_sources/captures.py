"""Import field captures from the phone app into `labels.csv`.

The capture app (`docs/capture_app_spec.md`) writes an append-only
`captures.jsonl` beside the photographs. This module is the authority on that
format: every rule in §2 of the spec is enforced here, so the app can be developed
against `acquire import-captures --dry-run` rather than against prose.

Three properties of the format, and why each is worth the small cost:

**Append-only.** A killed app loses at most the line it was mid-write. Rewriting a
label file in place during a three-hour shoot is how you lose 200 labels.

**Last line wins.** Correcting a typo means appending a new line for the same
filename. The earlier line stays as an audit trail, and no in-place edit can
corrupt the file.

**Tombstones, not deletions.** `{"filename": ..., "deleted": true}` retires a
capture without touching bytes already written.

The output deliberately leaves the eight corner columns **empty**: corners are
labelled on a desktop screen with zoom and undo, by the Sprint 2.1 annotation tool.
Corner precision bounds the recognizer's accuracy (Sprint 6.2 targets mean error
≤1.5% of plate width), and a phone-tapped corner is not good enough.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from alpr.data_sources.base import AcquisitionError, PlateType
from alpr.data_sources.imagesize import safe_image_size
from alpr.data_sources.own_photos import LABELS_HEADER, LABELS_NAME
from alpr.data_sources.storage import IMAGE_SUFFIXES, atomic_write_text

CAPTURES_NAME = "captures.jsonl"
SCHEMA_VERSION = 1

#: Directories the app writes into, and the `kind` value for each.
KINDS = {"testset": "testset", "close_ups": "close_ups"}

#: Exactly the tags the app may emit (spec §2.5). A closed vocabulary means a typo
#: is rejected at import rather than silently creating a tag no slice ever matches.
ALLOWED_TAGS = frozenset({"night", "flash", "yaw_high", "frame", "bolt", "dirt", "wet"})

#: Tags describing the session rather than the plate; the app carries these forward.
STICKY_TAGS = frozenset({"night", "flash"})

ALLOWED_CONSENT = frozenset({"visitor-signage", "resident-signed", "owner-verbal", "own-vehicle"})

PARTICULAR_PLATE = re.compile(r"^[A-Z]{3}[0-9]{3}$")

#: Coverage targets. Test set is the Sprint 2.1 exit gate; close-ups are Sprint 2.2,
#: where every one of the 36 characters must be extractable.
COVERAGE_TARGETS = {"testset": 5, "close_ups": 2}

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class CaptureError(AcquisitionError):
    """A captures.jsonl line violates the contract in docs/capture_app_spec.md."""


@dataclass(slots=True)
class Capture:
    """One labelled photograph, as the app recorded it."""

    filename: str
    kind: str
    plate_type: PlateType
    session: str
    captured_at: str
    plate: str | None = None
    tags: tuple[str, ...] = ()
    consent: str = ""
    image_width: int | None = None
    image_height: int | None = None
    device: str = ""
    app_version: str = ""
    note: str = ""

    @property
    def is_particular(self) -> bool:
        return self.plate_type is PlateType.PARTICULAR


@dataclass(slots=True)
class ImportReport:
    """What the import found. `problems` being empty is the acceptance criterion."""

    captures: list[Capture] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tombstoned: int = 0
    superseded: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    def coverage(self, kind: str) -> dict[str, int]:
        """Per-character counts across particular plates of one kind."""
        counts = dict.fromkeys(ALPHABET, 0)
        for cap in self.captures:
            if cap.kind != kind or not cap.is_particular or not cap.plate:
                continue
            for char in cap.plate:
                if char in counts:
                    counts[char] += 1
        return counts

    def shortfall(self, kind: str) -> dict[str, int]:
        """How many more of each character are still needed. Empty means done."""
        target = COVERAGE_TARGETS[kind]
        return {c: target - n for c, n in self.coverage(kind).items() if n < target}


def _require(value, name: str, lineno: int) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise CaptureError(f"line {lineno}: missing required field {name!r}")
    return str(value).strip()


def parse_capture_line(payload: dict, lineno: int) -> tuple[str, Capture | None]:
    """Validate one decoded JSONL object.

    Returns `(filename, capture)`, where `capture` is None for a tombstone.
    """
    schema = payload.get("schema")
    if schema != SCHEMA_VERSION:
        raise CaptureError(
            f"line {lineno}: schema {schema!r} is not supported (expected {SCHEMA_VERSION})"
        )

    filename = _require(payload.get("filename"), "filename", lineno)
    if "/" in filename or "\\" in filename:
        raise CaptureError(f"line {lineno}: filename must be a bare name, got {filename!r}")

    if payload.get("deleted"):
        return filename, None

    kind = _require(payload.get("kind"), "kind", lineno)
    if kind not in KINDS:
        raise CaptureError(f"line {lineno}: kind {kind!r} must be one of {sorted(KINDS)}")

    raw_type = _require(payload.get("plate_type"), "plate_type", lineno)
    try:
        plate_type = PlateType(raw_type)
    except ValueError:
        raise CaptureError(
            f"line {lineno}: plate_type {raw_type!r} must be one of {[t.value for t in PlateType]}"
        ) from None

    plate = payload.get("plate")
    plate = plate.strip().upper() if isinstance(plate, str) and plate.strip() else None
    if plate_type is PlateType.PARTICULAR:
        if plate is None:
            raise CaptureError(
                f"line {lineno}: {filename} is a particular plate with no plate string. "
                "If it could not be read, record it as plate_type 'unknown' instead — "
                "an unlabelled particular plate cannot be scored."
            )
        if not PARTICULAR_PLATE.match(plate):
            raise CaptureError(
                f"line {lineno}: plate {plate!r} is not LLLDDD (three letters then "
                "three digits). Colombian particular plates are always six characters."
            )

    raw_tags = payload.get("tags", [])
    if not isinstance(raw_tags, list):
        raise CaptureError(f"line {lineno}: tags must be a list, got {type(raw_tags).__name__}")
    tags = tuple(str(t).strip() for t in raw_tags if str(t).strip())
    unknown = sorted(set(tags) - ALLOWED_TAGS)
    if unknown:
        raise CaptureError(
            f"line {lineno}: unknown tag(s) {unknown}; allowed: {sorted(ALLOWED_TAGS)}"
        )

    consent = str(payload.get("consent") or "").strip()
    if consent and consent not in ALLOWED_CONSENT:
        raise CaptureError(
            f"line {lineno}: consent {consent!r} must be one of {sorted(ALLOWED_CONSENT)}"
        )

    def _int_or_none(key: str) -> int | None:
        value = payload.get(key)
        return int(value) if isinstance(value, (int, float)) else None

    return filename, Capture(
        filename=filename,
        kind=kind,
        plate_type=plate_type,
        session=_require(payload.get("session"), "session", lineno),
        captured_at=_require(payload.get("captured_at"), "captured_at", lineno),
        plate=plate,
        tags=tags,
        consent=consent,
        image_width=_int_or_none("image_width"),
        image_height=_int_or_none("image_height"),
        device=str(payload.get("device") or ""),
        app_version=str(payload.get("app_version") or ""),
        note=str(payload.get("note") or ""),
    )


def read_captures(path: Path | str) -> ImportReport:
    """Parse captures.jsonl, resolving last-wins and tombstones."""
    path = Path(path)
    report = ImportReport()
    if not path.exists():
        report.problems.append(
            f"{path} does not exist. Copy the app's alpr_capture/ folder into "
            f"{path.parent} first (see docs/capture_app_spec.md §7)."
        )
        return report

    resolved: dict[str, Capture | None] = {}
    seen: dict[str, int] = {}

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            report.problems.append(f"line {lineno}: not valid JSON ({exc.msg})")
            continue
        try:
            filename, capture = parse_capture_line(payload, lineno)
        except CaptureError as exc:
            report.problems.append(str(exc))
            continue
        if filename in seen:
            if capture is None:
                report.tombstoned += 1
            else:
                report.superseded += 1
        resolved[filename] = capture
        seen[filename] = lineno

    report.captures = [c for c in resolved.values() if c is not None]
    report.captures.sort(key=lambda c: c.filename)
    return report


def cross_check_images(report: ImportReport, root: Path) -> None:
    """Reconcile labels against the images actually on disk."""
    labelled = {c.filename for c in report.captures}

    for capture in report.captures:
        image = root / capture.kind / capture.filename
        if not image.exists():
            report.problems.append(
                f"{capture.filename}: labelled as {capture.kind} but no such file on disk"
            )
            continue
        size = safe_image_size(image)
        if size is None:
            report.problems.append(f"{capture.filename}: not a readable image")
            continue
        recorded = (capture.image_width, capture.image_height)
        if all(recorded) and recorded != size:
            report.problems.append(
                f"{capture.filename}: app recorded {recorded[0]}x{recorded[1]} but the "
                f"file header says {size[0]}x{size[1]}. Usually EXIF orientation was "
                "not baked into the pixels — see docs/capture_app_spec.md §6."
            )

    for kind in KINDS:
        directory = root / kind
        if not directory.exists():
            continue
        for image in sorted(directory.iterdir()):
            if not (image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES):
                continue
            if image.name not in labelled:
                report.warnings.append(
                    f"{image.name}: on disk in {kind}/ but has no label — it will be "
                    "indexed without a plate string"
                )


def write_labels(report: ImportReport, path: Path | str) -> int:
    """Write `labels.csv` with the eight corner columns empty.

    Corners are the annotation tool's job (Sprint 2.1). Emitting them here — even as
    zeros — would let a bad rectification look like a good one.
    """
    path = Path(path)
    rows = []
    for capture in report.captures:
        rows.append(
            {
                "filename": capture.filename,
                "plate": capture.plate or "",
                "plate_type": capture.plate_type.value,
                **dict.fromkeys(
                    ("tl_x", "tl_y", "tr_x", "tr_y", "br_x", "br_y", "bl_x", "bl_y"), ""
                ),
                "tags": "|".join(capture.tags),
                "consent": capture.consent,
                "session": capture.session,
            }
        )

    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(LABELS_HEADER))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, stream.getvalue())
    return len(rows)


def import_captures(root: Path | str, *, dry_run: bool = False) -> ImportReport:
    """Validate `captures.jsonl` under `root` and write `labels.csv` beside it."""
    root = Path(root)
    report = read_captures(root / CAPTURES_NAME)
    if report.problems:
        return report
    cross_check_images(report, root)
    if not dry_run and report.ok:
        write_labels(report, root / LABELS_NAME)
    return report


def format_report(report: ImportReport, root: Path, *, dry_run: bool) -> str:
    """Render the import outcome, including coverage against both exit gates."""
    lines: list[str] = []
    kinds = {k: sum(1 for c in report.captures if c.kind == k) for k in KINDS}
    lines.append(
        f"{len(report.captures)} capture(s): " + ", ".join(f"{n} {k}" for k, n in kinds.items())
    )
    if report.superseded:
        lines.append(f"  {report.superseded} label(s) superseded by a later line (last wins)")
    if report.tombstoned:
        lines.append(f"  {report.tombstoned} capture(s) deleted by tombstone")

    positives = sum(1 for c in report.captures if c.is_particular and c.kind == "testset")
    negatives = sum(1 for c in report.captures if c.kind == "testset" and c.plate_type.is_negative)
    unreadable = sum(
        1 for c in report.captures if c.kind == "testset" and c.plate_type is PlateType.UNKNOWN
    )
    labelled = [c for c in report.captures if c.kind == "testset"]

    def frac(tag: str) -> float:
        return (sum(1 for c in labelled if tag in c.tags) / len(labelled)) if labelled else 0.0

    def mark(ok: bool) -> str:
        return "OK " if ok else "-- "

    lines += [
        "",
        "Sprint 2.1 exit gate (test set):",
        f"  {mark(positives >= 200)}positives      {positives:>4} / 200",
        f"  {mark(negatives >= 40)}negatives      {negatives:>4} / 40",
        f"  {mark(frac('night') >= 0.15)}night          {frac('night'):>6.0%} / 15%",
        f"  {mark(frac('yaw_high') >= 0.25)}|yaw| > 10 deg {frac('yaw_high'):>6.0%} / 25%",
    ]
    if unreadable:
        lines.append(
            f"     {unreadable} capture(s) marked 'unknown' — these count as neither "
            "positive nor negative"
        )

    for kind, target in COVERAGE_TARGETS.items():
        short = report.shortfall(kind)
        label = "test set" if kind == "testset" else "close-ups"
        lines.append(f"  {mark(not short)}{label} chars >= {target}")
        if short:
            need = " ".join(f"{c}x{n}" for c, n in sorted(short.items()))
            lines.append(f"       still needed: {need}")

    if report.warnings:
        lines += ["", f"{len(report.warnings)} warning(s):"]
        lines += [f"  - {w}" for w in report.warnings[:10]]

    if report.problems:
        lines += ["", f"{len(report.problems)} PROBLEM(S) — nothing was written:"]
        lines += [f"  - {p}" for p in report.problems[:20]]
    elif dry_run:
        lines += ["", "dry run: valid. Re-run without --dry-run to write labels.csv"]
    else:
        lines += ["", f"wrote {root / LABELS_NAME}"]
    return "\n".join(lines)


def iter_alphabet_counts(report: ImportReport, kind: str) -> Iterator[tuple[str, int]]:
    """(character, count) in alphabet order — the app's coverage grid, server-side."""
    counts = report.coverage(kind)
    yield from ((c, counts[c]) for c in ALPHABET)


def example_line(**overrides) -> str:
    """A schema-valid line, for the app's test fixtures and the spec's examples."""
    payload = {
        "schema": SCHEMA_VERSION,
        "filename": "s1_20260809_193042_0001.jpg",
        "kind": "testset",
        "plate": "ABC123",
        "plate_type": "particular",
        "tags": ["night", "yaw_high"],
        "consent": "visitor-signage",
        "session": "s1",
        "captured_at": "2026-08-09T19:30:42-05:00",
        "image_width": 4032,
        "image_height": 3024,
        "device": "iPhone 15",
        "app_version": "1.0.0",
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)
