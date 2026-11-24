"""CCPD 2019 — Chinese City Parking Dataset (MIT licence, ~300k images).

Chinese plates, not Colombian ones, and that is fine: this source is used for
detector pretraining, for rectification transfer, and as hard negatives. The
recognizer never sees a CCPD plate as a readable string.

**All annotation lives in the filename.** There are no label files. A sample name
from the upstream README:

    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg
    │    │      │                 │                                     │           │  └ blurriness
    │    │      │                 │                                     │           └ brightness
    │    │      │                 │                                     └ plate character indices
    │    │      │                 └ four vertices, starting at the RIGHT-BOTTOM
    │    │      └ bounding box: left-up then right-bottom
    │    └ tilt: horizontal, vertical
    └ area ratio of the plate to the whole frame, per mille

The vertex convention is the trap. CCPD lists them starting from the right-bottom
vertex, so file order is (RB, LB, LT, RT). Consumed in file order they have the
*same* clockwise winding as (TL, TR, BR, BL) — it is a rotation of the same cycle,
not a reversal — so a winding check alone will not catch the mistake. Nothing
errors: the detector trains happily, the rectifier emits upside-down plates, and
the recognizer plateaus at chance. Conversion happens here, once, with a test.

**Box and corners disagree in the `ccpd_base` subset.** Measured across 69,895
indexed records on 2026-08-09: every subset except `base` has a published box that
is exactly the bounding hull of the four vertices (IoU 1.000). In `base`, all
39,847 records disagree — median IoU 0.91, minimum 0.54 — with small deltas of
inconsistent sign, i.e. the box and the quad were annotated independently rather
than one derived from the other.

Both are kept as published; nothing is silently "corrected". The consequence is for
Phase 6: when training the detector, derive the box from the corners rather than
mixing two annotation conventions, or the box head learns `base`'s looser labels
while the keypoint head learns the tighter quad. Corners are the higher-value
annotation here — corner quality is what bounds recognizer accuracy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from alpr.data_sources.base import (
    AcquisitionError,
    Box,
    Corners,
    PlateType,
    Point,
    Record,
    Status,
    order_corners,
)
from alpr.data_sources.registry import (
    AccessMode,
    FetchResult,
    Purpose,
    Source,
    SourceInfo,
    register,
)
from alpr.data_sources.storage import (
    IMAGE_SUFFIXES,
    download_gdrive,
    extract_archive,
    free_space,
    iter_files,
)

# Character tables, verbatim from the CCPD README. The last entry of each array is
# the letter O, used as a "no character" sign — there is no O in Chinese plates.
PROVINCES = (
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
)  # fmt: skip
ALPHABETS = (
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O",
)  # fmt: skip
ADS = (
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O",
)  # fmt: skip

#: CCPD lists vertices from the right-bottom, clockwise on screen: RB, LB, LT, RT.
#: TL/TR/BR/BL is therefore file indices 2, 3, 0, 1.
CCPD_VERTEX_ORDER = ("RB", "LB", "LT", "RT")
_TO_TL_TR_BR_BL = (2, 3, 0, 1)

#: Subdirectory name → condition tag, used for slicing detector metrics later.
SUBSETS = {
    "ccpd_base": "base",
    "ccpd_blur": "blur",
    "ccpd_challenge": "challenge",
    "ccpd_db": "db",
    "ccpd_fn": "fn",
    "ccpd_np": "no_plate",
    "ccpd_rotate": "rotate",
    "ccpd_tilt": "tilt",
    "ccpd_weather": "weather",
    "ccpd_green": "green",
}

CCPD_DRIVE_ID = "1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc"  # CCPD2019.tar.xz, from the README
CCPD_REPO = "https://github.com/detectRecog/CCPD"
CCPD_MIRROR = "https://opendatalab.com/OpenDataLab/CCPD"


class CCPDFilenameError(AcquisitionError):
    """A CCPD filename did not parse. Raised rather than returning garbage."""


@dataclass(frozen=True, slots=True)
class CCPDAnnotation:
    """Everything encoded in one CCPD filename, decoded and validated."""

    area_code: int
    tilt_horizontal: int
    tilt_vertical: int
    box: Box
    corners: Corners  # TL, TR, BR, BL — converted
    raw_vertices: tuple[Point, ...]  # RB, LB, LT, RT — as published
    plate: str
    plate_indices: tuple[int, ...]
    brightness: int
    blurriness: int

    @property
    def area_ratio(self) -> float:
        """Plate area as a fraction of the frame.

        The README says "area ratio" without giving the scale; the published
        example resolves it: box 232 x 90 px in a 720 x 1160 frame is 0.0250 of
        the image, and the code is 025. The integer is therefore per mille.
        """
        return self.area_code / 1000.0


def _parse_point(token: str, *, field: str, filename: str) -> Point:
    if "&" not in token:
        raise CCPDFilenameError(f"{filename}: {field} point {token!r} is missing its '&' separator")
    xs, _, ys = token.partition("&")
    try:
        return (float(int(xs)), float(int(ys)))
    except ValueError:
        raise CCPDFilenameError(
            f"{filename}: {field} point {token!r} has non-integer coordinates"
        ) from None


def _parse_int(token: str, *, field: str, filename: str) -> int:
    try:
        return int(token)
    except ValueError:
        raise CCPDFilenameError(f"{filename}: {field} {token!r} is not an integer") from None


def decode_plate(indices: tuple[int, ...], *, filename: str = "<indices>") -> str:
    """Character indices → plate string.

    Seven characters on a standard plate (province + letter + five alphanumerics);
    eight on the CCPD-Green energy-vehicle subset.
    """
    if len(indices) not in (7, 8):
        raise CCPDFilenameError(
            f"{filename}: expected 7 or 8 plate character indices, got {len(indices)}: {indices}"
        )
    tables = (PROVINCES, ALPHABETS) + (ADS,) * (len(indices) - 2)
    chars = []
    for pos, (idx, table) in enumerate(zip(indices, tables, strict=True)):
        if not 0 <= idx < len(table):
            raise CCPDFilenameError(
                f"{filename}: character index {idx} at position {pos} is outside "
                f"the table of {len(table)} entries"
            )
        chars.append(table[idx])
    return "".join(chars)


def parse_ccpd_filename(filename: str | Path, *, normalize_corners: bool = True) -> CCPDAnnotation:
    """Decode a CCPD filename. Raises `CCPDFilenameError` on anything malformed.

    Raising rather than returning a partial annotation is deliberate: a filename
    that silently yields garbage coordinates is indistinguishable from a good one
    until the detector fails to converge weeks later.

    With `normalize_corners=True` (the default) the returned `corners` are in
    TL/TR/BR/BL order. `raw_vertices` always holds the published RB/LB/LT/RT order,
    so the conversion is inspectable rather than folded away.
    """
    name = Path(filename).name
    stem = Path(name).stem

    fields = stem.split("-")
    if len(fields) != 7:
        raise CCPDFilenameError(
            f"{name}: expected 7 '-'-separated fields, got {len(fields)}. "
            "CCPD names look like "
            "'025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15'"
        )
    area_s, tilt_s, box_s, verts_s, plate_s, bright_s, blur_s = fields

    area_code = _parse_int(area_s, field="area", filename=name)
    if area_code < 0:
        raise CCPDFilenameError(f"{name}: area code {area_code} is negative")

    tilt_parts = tilt_s.split("_")
    if len(tilt_parts) != 2:
        raise CCPDFilenameError(f"{name}: tilt field {tilt_s!r} must be 'horizontal_vertical'")
    tilt_h = _parse_int(tilt_parts[0], field="horizontal tilt", filename=name)
    tilt_v = _parse_int(tilt_parts[1], field="vertical tilt", filename=name)

    box_parts = box_s.split("_")
    if len(box_parts) != 2:
        raise CCPDFilenameError(
            f"{name}: bounding box field {box_s!r} must be 'leftup_rightbottom'"
        )
    (x1, y1) = _parse_point(box_parts[0], field="box left-up", filename=name)
    (x2, y2) = _parse_point(box_parts[1], field="box right-bottom", filename=name)
    try:
        box = Box(x1, y1, x2, y2)
    except AcquisitionError as exc:
        raise CCPDFilenameError(f"{name}: {exc}") from exc

    vert_parts = verts_s.split("_")
    if len(vert_parts) != 4:
        raise CCPDFilenameError(
            f"{name}: vertices field {verts_s!r} must hold exactly 4 'x&y' points, "
            f"got {len(vert_parts)}"
        )
    raw_vertices = tuple(
        _parse_point(tok, field=f"vertex {CCPD_VERTEX_ORDER[i]}", filename=name)
        for i, tok in enumerate(vert_parts)
    )

    rotated = tuple(raw_vertices[i] for i in _TO_TL_TR_BR_BL)
    if normalize_corners:
        # `order_corners` is geometric, so it is idempotent on well-behaved samples
        # and self-correcting on the Rotate/Tilt subsets where the published
        # starting vertex is not where the fixed permutation assumes.
        try:
            corners = order_corners(rotated)
        except AcquisitionError as exc:
            raise CCPDFilenameError(f"{name}: {exc}") from exc
    else:
        corners = rotated  # type: ignore[assignment]

    plate_parts = plate_s.split("_")
    indices = tuple(
        _parse_int(tok, field=f"plate character index {i}", filename=name)
        for i, tok in enumerate(plate_parts)
    )
    plate = decode_plate(indices, filename=name)

    return CCPDAnnotation(
        area_code=area_code,
        tilt_horizontal=tilt_h,
        tilt_vertical=tilt_v,
        box=box,
        corners=corners,
        raw_vertices=raw_vertices,
        plate=plate,
        plate_indices=indices,
        brightness=_parse_int(bright_s, field="brightness", filename=name),
        blurriness=_parse_int(blur_s, field="blurriness", filename=name),
    )


@register
class CCPDSource(Source):
    """CCPD 2019 — MIT licensed, commercially clean, ~12 GB."""

    info = SourceInfo(
        name="ccpd",
        title="CCPD 2019 (Chinese City Parking Dataset)",
        license_key="mit",
        access=AccessMode.AUTOMATIC,
        purposes=frozenset(
            {
                Purpose.DETECTOR_PRETRAIN,
                Purpose.RECTIFICATION,
                Purpose.NEGATIVES,
                Purpose.BACKGROUNDS,
            }
        ),
        url=CCPD_REPO,
        size_hint="~300k images, ~12 GB",
        notes=(
            "Chinese plates. Used for detector pretraining, rectification transfer "
            "and hard negatives — never as readable strings."
        ),
    )

    #: Populated once `index()` has been fully consumed.
    last_index_errors: list[str] = []
    last_index_skipped: int = 0

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / "CCPD2019.tar.xz"

    #: Measured 2026-08-09 from the Drive object.
    ARCHIVE_BYTES = 13_164_924_944

    #: Measured, not assumed: a 20% extraction produced 4.8 GB, so the full tree is
    #: ~24 GB — nearly twice the archive. The obvious guess ("JPEG payload, so the
    #: archive is about the same size as the tree") is wrong, because xz compresses
    #: tar's per-member padding across ~350k small files very effectively. Guessing
    #: here is what fills a disk at 90% of a one-hour extraction.
    EXTRACTED_BYTES = 24_000_000_000

    def preflight(self, *, extract: bool = True, keep_fraction: float = 1.0) -> str | None:
        """Return a problem description, or None if the fetch can proceed.

        Checked up front because the alternative is a one-hour download that dies
        at 90% with `No space left on device`, having already filled the disk.
        """
        needed = self.EXTRACTED_BYTES * keep_fraction if extract else 0.0
        if not self.archive_path.exists():
            needed += self.ARCHIVE_BYTES
        available = free_space(self.raw_dir)
        if available < needed:
            return (
                f"not enough disk space for CCPD: need ~{needed / 1e9:.0f} GB, "
                f"but {available / 1e9:.0f} GB is free at {self.raw_dir}.\n"
                "Options, cheapest first:\n"
                "  - extract a sample: --keep-fraction 0.2 keeps ~60k images (~2.6 GB);\n"
                "  - --subsets ccpd_base ccpd_np to take only what a stage needs;\n"
                "  - --delete-archive to reclaim 13 GB once extraction succeeds;\n"
                "  - point ALPR_DATA_ROOT at an external drive;\n"
                "  - CCPD is only needed for detector pretraining (Phase 6). Phases 1-5 "
                "reach the V0 gate without it, so deferring this is a legitimate choice."
            )
        return None

    def _selector(
        self, subsets: Sequence[str] | None, keep_fraction: float
    ) -> Callable[[str], bool] | None:
        """Build a member filter for selective extraction.

        Sampling is deterministic — a stable hash of the filename, not a random
        draw — so the same fraction always yields the same images. Re-extracting
        after an interruption therefore converges on the same corpus rather than
        an overlapping-but-different one.
        """
        if subsets is None and keep_fraction >= 1.0:
            return None
        wanted = {s.lower() for s in subsets} if subsets else None

        def select(name: str) -> bool:
            parts = Path(name).parts
            if wanted is not None and not any(p.lower() in wanted for p in parts):
                return False
            if keep_fraction >= 1.0:
                return True
            if not name.lower().endswith(tuple(IMAGE_SUFFIXES)):
                return True  # keep split files, READMEs and directory entries
            digest = hashlib.sha1(Path(name).name.encode()).digest()
            return int.from_bytes(digest[:4], "big") / 2**32 < keep_fraction

        return select

    def fetch(
        self,
        *,
        force: bool = False,
        extract: bool = True,
        subsets: Sequence[str] | None = None,
        keep_fraction: float = 1.0,
        delete_archive: bool = False,
    ) -> FetchResult:
        self.paths.ensure()
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if (problem := self.preflight(extract=extract, keep_fraction=keep_fraction)) is not None:
            return FetchResult(source=self.name, status=Status.FAILED, message=problem)

        download_gdrive(CCPD_DRIVE_ID, self.archive_path, force=force)
        message = f"archive at {self.archive_path}"
        archive_bytes = self.archive_path.stat().st_size
        if extract:
            extract_archive(
                self.archive_path,
                self.raw_dir,
                force=force,
                select=self._selector(subsets, keep_fraction),
                delete_archive=delete_archive,
                progress_every=20_000,
            )
            message = f"extracted into {self.raw_dir}"
            if delete_archive:
                message += " (archive deleted)"
        return FetchResult(
            source=self.name,
            status=Status.DOWNLOADED,
            message=message,
            paths=[str(self.archive_path)],
            bytes_written=archive_bytes,
        )

    def index(self) -> Iterator[Record]:
        """Walk the extracted tree and yield one Record per parseable image.

        Unparseable filenames are counted and reported through `last_index_errors`
        rather than aborting the walk — one bad file in 300k should not cost the
        other 299,999 — but a filename handed directly to `parse_ccpd_filename`
        still raises, which is what the unit tests exercise.
        """
        errors: list[str] = []
        skipped = 0
        for path in iter_files(self.raw_dir, IMAGE_SUFFIXES):
            subset = self._subset_for(path)
            if subset == "no_plate":
                # ccpd_np holds frames with no plate at all: no annotation to parse.
                yield Record(
                    source=self.name,
                    image_path=self.paths.relative(path),
                    license_key=self.info.license_key,
                    plate_type=PlateType.NON_PLATE,
                    tags=("ccpd", subset),
                )
                continue
            try:
                ann = parse_ccpd_filename(path.name)
            except CCPDFilenameError as exc:
                skipped += 1
                if len(errors) < 20:
                    errors.append(str(exc))
                continue
            yield Record(
                source=self.name,
                image_path=self.paths.relative(path),
                license_key=self.info.license_key,
                plate=ann.plate,
                plate_type=PlateType.FOREIGN,
                box=ann.box,
                corners=ann.corners,
                attribution_url=None,
                tags=("ccpd",) + ((subset,) if subset else ()),
                extra={
                    "area_ratio": round(ann.area_ratio, 5),
                    "tilt_h": ann.tilt_horizontal,
                    "tilt_v": ann.tilt_vertical,
                    "brightness": ann.brightness,
                    "blurriness": ann.blurriness,
                },
            )
        self.last_index_errors = errors
        self.last_index_skipped = skipped

    @staticmethod
    def _subset_for(path: Path) -> str:
        for part in path.parts:
            if part.lower() in SUBSETS:
                return SUBSETS[part.lower()]
        return ""
