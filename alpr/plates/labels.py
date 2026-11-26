"""Label storage and the items a source offers for labelling.

The CSV layout is owned by `alpr.data_sources.own_photos` (the consumer defines the
contract; this tool writes to it), and reading goes through its `read_labels`, so a
file this module writes is by construction one Phase 1 can read.

Where labels live:

    data/raw/own_photos/labels.csv     own photos — keyed by filename (Phase 1 contract)
    data/labels/<source>.csv           any other indexed source — keyed by the record's
                                       image path, plus "#k" when one image holds
                                       several plates

Labels are stored in **full-image pixel coordinates**, never display coordinates.

Sideways photos: a label may carry one rotation tag — `rot90`, `rot180` or `rot270`
— meaning "turn the image this many degrees clockwise to stand it upright". Its
corners are in that upright frame. Every consumer therefore reads the image with
`load_labelled_image` (or `load_image(path, rotation_of(label.tags))`) before using
the corners. The tag lives in the existing tags column, so the CSV layout Phase 1
reads is unchanged.
"""

from __future__ import annotations

import csv
import io
import re
import threading
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alpr.data_sources.base import Box, PlateType, Record, read_jsonl
from alpr.data_sources.own_photos import LABELS_HEADER, LABELS_NAME, Label, read_labels
from alpr.data_sources.storage import IMAGE_SUFFIXES, DataPaths, atomic_write_text, iter_files
from alpr.plates.rectify import ROTATIONS, check_quad, load_image

OWN_PHOTOS = "own_photos"

#: Tags the tool offers. Free text is allowed in the CSV, but a closed set in the
#: UI keeps "blurry" from also appearing as "blur" and "Blurry".
KNOWN_TAGS = (
    "glyph_source",  # sharp, frontal: characters will be cut from this plate
    "blurry",
    "worn",
    "occluded",
    "night",
    "high_yaw",
    "small",
)

PLATE_PATTERNS: dict[PlateType, re.Pattern[str]] = {
    PlateType.PARTICULAR: re.compile(r"^[A-Z]{3}[0-9]{3}$"),
    PlateType.PUBLICO: re.compile(r"^[A-Z]{3}[0-9]{3}$"),
    PlateType.MOTO: re.compile(r"^[A-Z]{3}[0-9]{2}[A-Z]?$"),
}

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

#: Degrees clockwise -> the tag that records it. 0 has no tag.
ROTATION_TAGS = {deg: f"rot{deg}" for deg in ROTATIONS if deg}


class LabelError(ValueError):
    """A label that would not survive being read back, or that cannot be a plate."""


@dataclass(frozen=True, slots=True)
class Item:
    """One plate to label: an image, and optionally the box a source already drew."""

    key: str
    image: Path
    box: Box | None = None
    record: Record | None = None  # the index record, for provenance; None for own photos

    @property
    def size_hint(self) -> float:
        return self.box.width if self.box else 0.0


def rotation_of(tags: Iterable[str]) -> int:
    """The clockwise quarter turn a label's tags ask for (0 when there is none)."""
    tags = set(tags)
    found = [deg for deg, tag in ROTATION_TAGS.items() if tag in tags]
    if len(found) > 1:
        raise LabelError(f"conflicting rotation tags: {sorted(ROTATION_TAGS[d] for d in found)}")
    return found[0] if found else 0


def with_rotation(tags: Iterable[str], degrees: int) -> list[str]:
    """`tags` with any rotation tag replaced by the one for `degrees`."""
    if degrees not in ROTATIONS:
        raise LabelError(f"rotation must be one of {ROTATIONS} degrees, got {degrees!r}")
    out = [t for t in tags if t not in ROTATION_TAGS.values()]
    return out + [ROTATION_TAGS[degrees]] if degrees else out


def without_rotation(tags: Iterable[str]) -> list[str]:
    return [t for t in tags if t not in ROTATION_TAGS.values()]


def load_labelled_image(path: Path | str, label: Label) -> np.ndarray:
    """The image in the frame the label's corners live in (upright, after rotation)."""
    return load_image(path, rotation_of(label.tags))


def labels_path(paths: DataPaths, source: str) -> Path:
    if source == OWN_PHOTOS:
        return paths.raw(OWN_PHOTOS) / LABELS_NAME
    return paths.root / "labels" / f"{source}.csv"


def list_items(paths: DataPaths, source: str, *, order: str = "size") -> list[Item]:
    """Everything `source` offers for labelling, largest plate first by default.

    Largest-first puts the best glyph candidates at the front, which is where the
    labelling effort pays off twice.
    """
    if source == OWN_PHOTOS:
        raw = paths.raw(OWN_PHOTOS)
        items = [
            Item(key=p.name, image=p)
            for sub in ("close_ups", "testset")
            for p in iter_files(raw / sub, IMAGE_SUFFIXES)
        ]
    else:
        index = paths.index(source)
        if not index.exists():
            raise LabelError(
                f"no index for {source!r} at {index} — run `acquire fetch {source}` first"
            )
        records = list(read_jsonl(index))
        per_image = Counter(r.image_path for r in records)
        seen: Counter[str] = Counter()
        items = []
        for r in sorted(records, key=lambda r: (r.image_path, r.box.x1 if r.box else 0.0)):
            k = seen[r.image_path]
            seen[r.image_path] += 1
            key = r.image_path if per_image[r.image_path] == 1 else f"{r.image_path}#{k}"
            items.append(Item(key=key, image=paths.root / r.image_path, box=r.box, record=r))

    if order == "size":
        items.sort(key=lambda it: (-it.size_hint, it.key))
    elif order == "name":
        items.sort(key=lambda it: it.key)
    else:
        raise LabelError(f"unknown order {order!r}; use 'size' or 'name'")
    return items


def validate(plate: str | None, plate_type: PlateType) -> str | None:
    """Normalise and check a plate string for its type. Returns the stored form."""
    text = re.sub(r"[\s\-·.]", "", (plate or "")).upper() or None
    if plate_type is PlateType.PARTICULAR and text is None:
        raise LabelError("a particular plate needs its text (or mark it 'unknown')")
    pattern = PLATE_PATTERNS.get(plate_type)
    if text is not None and pattern is not None and not pattern.match(text):
        expected = {"moto": "LLLDDL", "particular": "LLLDDD", "publico": "LLLDDD"}
        raise LabelError(
            f"{text!r} is not a {plate_type.value} plate (format {expected[plate_type.value]})"
        )
    return text


class LabelStore:
    """Thread-safe, rewrite-on-save label file. Small files; atomic writes."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._labels: dict[str, Label] = read_labels(self.path)

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, key: object) -> bool:
        return key in self._labels

    def get(self, key: str) -> Label | None:
        return self._labels.get(key)

    def all(self) -> list[Label]:
        return list(self._labels.values())

    def put(
        self,
        key: str,
        *,
        plate: str | None,
        plate_type: PlateType,
        corners: Sequence[Any] | None,
        tags: Iterable[str] = (),
    ) -> Label:
        text = validate(plate, plate_type)
        quad = check_quad(corners) if corners is not None else None
        if quad is None and plate_type is not PlateType.NON_PLATE:
            raise LabelError("four corners are required unless the image has no plate")
        clean_tags = tuple(sorted({t.strip() for t in tags if t.strip()}))
        bad = [t for t in clean_tags if "|" in t or "," in t]
        if bad:
            raise LabelError(f"tags may not contain '|' or ',': {bad}")
        rotation_of(clean_tags)  # at most one rotation
        with self._lock:
            old = self._labels.get(key)
            label = Label(
                filename=key,
                plate=text,
                plate_type=plate_type,
                corners=quad,
                tags=clean_tags,
                consent=old.consent if old else "",
                session=old.session if old else "",
            )
            self._labels[key] = label
            self._save()
        return label

    def delete(self, key: str) -> bool:
        with self._lock:
            existed = self._labels.pop(key, None) is not None
            if existed:
                self._save()
        return existed

    def _save(self) -> None:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(LABELS_HEADER)
        for key in sorted(self._labels):
            lb = self._labels[key]
            coords = [f"{v:.2f}" for pt in lb.corners for v in pt] if lb.corners else [""] * 8
            writer.writerow(
                [
                    key,
                    lb.plate or "",
                    lb.plate_type.value,
                    *coords,
                    "|".join(lb.tags),
                    lb.consent,
                    lb.session,
                ]
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, buf.getvalue())
        # Re-read through the Phase 1 reader: if it cannot parse what was just
        # written, fail now, at the label that caused it.
        read_labels(self.path)


@dataclass(frozen=True, slots=True)
class Coverage:
    labelled: int
    by_type: dict[str, int]
    distinct_plates: int
    char_counts: dict[str, int]  # over distinct particular plates
    min_per_char: int

    def missing(self, target: int = 5) -> dict[str, int]:
        return {c: n for c, n in self.char_counts.items() if n < target}


def coverage(labels: Iterable[Label]) -> Coverage:
    """Progress against the Sprint 2.1 gate: every character at least 5 times.

    Characters are counted over **distinct** plates, not images. The Roboflow sets
    photograph the same car many times; counting images would report coverage that
    is really one vehicle seen eight times.
    """
    labels = list(labels)
    by_type = Counter(lb.plate_type.value for lb in labels)
    plates = {lb.plate for lb in labels if lb.plate_type is PlateType.PARTICULAR and lb.plate}
    counts = Counter({c: 0 for c in CHARSET})
    for p in plates:
        counts.update(ch for ch in p if ch in counts)
    return Coverage(
        labelled=len(labels),
        by_type=dict(by_type),
        distinct_plates=len(plates),
        char_counts=dict(counts),
        min_per_char=min(counts.values()),
    )
