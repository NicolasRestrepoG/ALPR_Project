"""Roboflow Universe — three Colombian license-plate projects.

These are the only Colombian *image* data the project can obtain automatically, and
they carry the two complications the plan warns about.

**Box-only annotations.** The exports have no keypoint labels. Corners are left
`None`, and Phase 6 masks the keypoint loss on these records. Fabricating corners
from a bounding box would teach the detector that every plate is axis-aligned
(plan §7), so there is no code path here that can produce them.

**Two of the three have unverified licences.** They default to the `unknown`
licence, which is not commercially usable, so `--commercial-only` excludes them by
construction rather than by anyone remembering to.

The `roboflow` PyPI package is deliberately *not* used: it depends on numpy,
opencv-python and Pillow, which would end Phase 1's property of being testable with
no image or ML libraries installed. The public REST API is two endpoints.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from alpr.config import get_secret, secret_source
from alpr.data_sources.base import (
    AcquisitionError,
    Box,
    PlateType,
    Record,
    Status,
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
from alpr.data_sources.storage import IMAGE_SUFFIXES, download, extract_archive, iter_files

API_ROOT = "https://api.roboflow.com"
API_KEY_ENV = "ROBOFLOW_API_KEY"
EXPORT_FORMAT = "yolov8"

API_KEY_INSTRUCTIONS = f"""\
No Roboflow API key found.

  1. Create a free account at https://app.roboflow.com
  2. Settings -> API Keys -> copy the Private API Key
  3. Put it in the project's .env file:

       {API_KEY_ENV}=your_key_here

     .env is gitignored, so the key cannot be committed. Alternatively export
     {API_KEY_ENV} in your shell — a real environment variable takes precedence
     over the file.

The key is free and rate-limited rather than metered. It is never written to disk
by this tool, never logged, and never recorded in data/manifest.json.
"""


class RoboflowError(AcquisitionError):
    """The Roboflow API refused, or returned something unusable."""


@dataclass(frozen=True, slots=True)
class RoboflowProject:
    workspace: str
    project: str
    version: int | None = None  # None -> latest

    @property
    def slug(self) -> str:
        return f"{self.workspace}/{self.project}"

    @property
    def url(self) -> str:
        return f"https://universe.roboflow.com/{self.workspace}/{self.project}"


def api_key() -> str | None:
    """The Roboflow key from the environment or `.env`. None if unset."""
    return get_secret(API_KEY_ENV)


def api_key_source() -> str:
    """Where the key came from, for diagnostics. Never returns the key itself."""
    return secret_source(API_KEY_ENV)


KEY_REJECTED_HELP = """\
Roboflow rejected the API key (401: "does not exist or has been revoked").

Roboflow issues two different keys and only one of them works here:

  Private API Key   <- THIS ONE. Roboflow dashboard -> Settings -> API Keys.
                       Grants dataset export via the REST API.
  Publishable Key   <- not this one. For client-side inference in a browser;
                       rejected by the dataset endpoints.

Check you copied the private key, that the workspace still exists, and that the
key has not been rotated. Verify a key without fetching anything:

    acquire keys --check

That calls Roboflow's own validation endpoint and reports valid/invalid without
printing the key."""


def check_api_key(key: str, *, timeout: float = 25.0) -> tuple[bool, str]:
    """Validate a key against Roboflow's root endpoint. Returns (ok, message).

    Never includes the key in the returned message.
    """
    import requests  # noqa: PLC0415 — lazy

    try:
        resp = requests.get(f"{API_ROOT}/?api_key={key}", timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — a network problem is not an invalid key
        return False, f"could not reach {API_ROOT}: {type(exc).__name__}: {exc}"
    if resp.status_code == 200:
        workspace = (resp.json() or {}).get("workspace")
        return True, f"valid (workspace {workspace!r})" if workspace else "valid"
    if resp.status_code == 401:
        return False, "rejected: key does not exist or has been revoked"
    return False, f"unexpected status {resp.status_code}: {resp.text[:120]}"


def _get_json(url: str, *, timeout: float = 30.0) -> dict:
    import requests  # noqa: PLC0415 — lazy, keeps the base layer dependency-free

    resp = requests.get(url, timeout=timeout)
    if resp.status_code == 401:
        raise RoboflowError(KEY_REJECTED_HELP)
    if resp.status_code == 404:
        raise RoboflowError(
            f"Roboflow project not found (404): {url.split('?')[0]}. "
            "Universe projects are occasionally renamed or made private."
        )
    resp.raise_for_status()
    return resp.json()


def resolve_version(project: RoboflowProject, key: str) -> int:
    """Return the requested version, or the highest published one."""
    if project.version is not None:
        return project.version
    info = _get_json(f"{API_ROOT}/{project.slug}?api_key={key}")
    versions = info.get("versions") or []
    if not versions:
        raise RoboflowError(
            f"{project.slug} has no published versions, so there is nothing to export.\n"
            "A Roboflow project holds images, but only a *generated version* is "
            "downloadable, and only the project's owner can generate one.\n"
            "Workaround: open the project on Universe, fork it into your own workspace, "
            "generate a version there with NO augmentation (never horizontal flip), then "
            "point RoboflowProject at your workspace slug."
        )
    numbers = []
    for v in versions:
        vid = str(v.get("id", ""))
        tail = vid.rsplit("/", 1)[-1]
        if tail.isdigit():
            numbers.append(int(tail))
    if not numbers:
        raise RoboflowError(f"{project.slug}: could not parse a version number from {versions!r}")
    return max(numbers)


def version_details(project: RoboflowProject, version: int, key: str) -> dict:
    """Metadata for one version: image count, splits, preprocessing, augmentation."""
    payload = _get_json(f"{API_ROOT}/{project.slug}/{version}?api_key={key}")
    return payload.get("version") or {}


def horizontal_flip_enabled(augmentation: dict) -> bool:
    """True if a Roboflow version bakes horizontal mirroring into its images.

    This is the one augmentation the project forbids outright (plan §2): a front
    plate is never seen mirrored, and flipping destroys the identity of 2/5, E/3
    and the asymmetric MinTransporte logo. Roboflow applies its augmentation at
    *export* time, so a flipped version cannot be undone by anything downstream —
    it has to be refused here.
    """
    for field in ("flip", "bbflip"):
        settings = augmentation.get(field) or {}
        if isinstance(settings, dict) and settings.get("horizontal"):
            return True
    return False


def export_link(project: RoboflowProject, version: int, key: str) -> str:
    payload = _get_json(f"{API_ROOT}/{project.slug}/{version}/{EXPORT_FORMAT}?api_key={key}")
    link = (payload.get("export") or {}).get("link")
    if not link:
        raise RoboflowError(
            f"{project.slug} v{version}: no export link in the API response. "
            f"Roboflow may still be generating the {EXPORT_FORMAT} export — retry shortly."
        )
    return link


# --------------------------------------------------------------------------- #
# YOLO annotation parsing
# --------------------------------------------------------------------------- #


def parse_yolo_label(line: str) -> tuple[int, float, float, float, float]:
    """Parse one `class cx cy w h` line, all box values normalized to [0, 1]."""
    parts = line.split()
    if len(parts) < 5:
        raise RoboflowError(f"YOLO label needs 5 fields, got {len(parts)}: {line!r}")
    try:
        cls = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:5])
    except ValueError as exc:
        raise RoboflowError(f"non-numeric YOLO label {line!r}: {exc}") from exc
    if cls < 0:
        raise RoboflowError(f"negative class index in {line!r}")
    for name, value in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
        if not 0.0 <= value <= 1.0:
            raise RoboflowError(
                f"YOLO {name}={value} is outside [0, 1] in {line!r} — "
                "this export may already be in pixel units"
            )
    if w <= 0 or h <= 0:
        raise RoboflowError(f"degenerate YOLO box in {line!r}")
    return cls, cx, cy, w, h


def yolo_to_box(cx: float, cy: float, w: float, h: float, width: int, height: int) -> Box:
    """Normalized centre/size → absolute pixel corners, clipped to the image."""
    half_w = w * width / 2.0
    half_h = h * height / 2.0
    x_center = cx * width
    y_center = cy * height
    x1 = max(0.0, x_center - half_w)
    y1 = max(0.0, y_center - half_h)
    x2 = min(float(width), x_center + half_w)
    y2 = min(float(height), y_center + half_h)
    if x2 <= x1 or y2 <= y1:
        raise RoboflowError(
            f"box collapsed after clipping to {width}x{height}: cx={cx} cy={cy} w={w} h={h}"
        )
    return Box(x1, y1, x2, y2)


def read_class_names(dataset_dir: Path) -> list[str]:
    """Pull `names:` out of data.yaml without a YAML dependency.

    The generated file is machine-written and uses one of two shapes:
    `names: ['a', 'b']` on one line, or a block list. Both are handled; an
    unparseable file yields an empty list rather than an exception, since class
    names are informational here.
    """
    for candidate in ("data.yaml", "data.yml"):
        path = dataset_dir / candidate
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, raw in enumerate(text.splitlines()):
            line = raw.strip()
            if not line.startswith("names:"):
                continue
            inline = line[len("names:") :].strip()
            if inline.startswith("["):
                try:
                    return [str(n) for n in json.loads(inline.replace("'", '"'))]
                except json.JSONDecodeError:
                    return []
            names = []
            for follow in text.splitlines()[i + 1 :]:
                stripped = follow.strip()
                if not stripped.startswith("-"):
                    break
                names.append(stripped.lstrip("-").strip().strip("'\""))
            return names
    return []


#: Class-name substrings that genuinely identify a plate type. Anything else stays
#: UNKNOWN — a detection box does not tell you whether a plate is particular.
_CLASS_HINTS = (
    ("moto", PlateType.MOTO),
    ("publico", PlateType.PUBLICO),
    ("público", PlateType.PUBLICO),
    ("taxi", PlateType.PUBLICO),
)


def plate_type_for_class(name: str) -> PlateType:
    lowered = name.lower()
    for needle, plate_type in _CLASS_HINTS:
        if needle in lowered:
            return plate_type
    return PlateType.UNKNOWN


# --------------------------------------------------------------------------- #
# Source
# --------------------------------------------------------------------------- #


class RoboflowSource(Source):
    """Shared implementation; each project subclasses with its own `info`."""

    project: RoboflowProject

    last_index_errors: list[str] = []
    last_index_skipped: int = 0

    @property
    def archive_path(self) -> Path:
        return self.raw_dir / f"{self.project.project}.zip"

    @property
    def dataset_dir(self) -> Path:
        return self.raw_dir / "dataset"

    def fetch(self, *, force: bool = False) -> FetchResult:
        key = api_key()
        if key is None:
            return FetchResult(
                source=self.name,
                status=Status.UNAVAILABLE,
                message=API_KEY_INSTRUCTIONS,
            )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        version = resolve_version(self.project, key)

        details = version_details(self.project, version, key)
        augmentation = details.get("augmentation") or {}
        if horizontal_flip_enabled(augmentation):
            return FetchResult(
                source=self.name,
                status=Status.FAILED,
                message=(
                    f"{self.project.slug} v{version} bakes HORIZONTAL FLIP into its "
                    "exported images.\n"
                    "The project never horizontal-flips training data (plan §2): a front "
                    "plate is never seen mirrored, and flipping destroys the identity of "
                    "2/5, E/3 and the asymmetric MinTransporte logo. Roboflow applies "
                    "augmentation at export time, so this cannot be undone downstream.\n"
                    "Pin a version without flip augmentation via RoboflowProject(version=N)."
                ),
            )

        link = export_link(self.project, version, key)
        download(link, self.archive_path, force=force, progress=False)
        if not zipfile.is_zipfile(self.archive_path):
            self.archive_path.unlink(missing_ok=True)
            return FetchResult(
                source=self.name,
                status=Status.FAILED,
                message=f"{self.project.slug} v{version}: export was not a zip archive",
            )
        extract_archive(self.archive_path, self.dataset_dir, force=force)

        note = ""
        if details.get("preprocessing", {}).get("resize"):
            resize = details["preprocessing"]["resize"]
            note = (
                f"; NOTE images are {resize.get('format')} {resize.get('width')}x"
                f"{resize.get('height')} — aspect ratio is distorted upstream"
            )
        if augmentation:
            note += f"; augmentation baked in: {sorted(augmentation)}"

        return FetchResult(
            source=self.name,
            status=Status.DOWNLOADED,
            message=f"{self.project.slug} v{version} → {self.dataset_dir}{note}",
            paths=[str(self.dataset_dir)],
            bytes_written=self.archive_path.stat().st_size,
        )

    def index(self) -> Iterator[Record]:
        errors: list[str] = []
        skipped = 0
        class_names = read_class_names(self.dataset_dir)

        for image_path in iter_files(self.dataset_dir, IMAGE_SUFFIXES):
            label_path = self._label_for(image_path)
            if label_path is None or not label_path.exists():
                skipped += 1
                continue
            size = safe_image_size(image_path)
            if size is None:
                skipped += 1
                if len(errors) < 20:
                    errors.append(f"{image_path.name}: could not read image dimensions")
                continue
            width, height = size
            split = self._split_for(image_path)

            for lineno, raw in enumerate(
                label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                line = raw.strip()
                if not line:
                    continue
                try:
                    cls, cx, cy, w, h = parse_yolo_label(line)
                    box = yolo_to_box(cx, cy, w, h, width, height)
                except RoboflowError as exc:
                    skipped += 1
                    if len(errors) < 20:
                        errors.append(f"{label_path.name}:{lineno}: {exc}")
                    continue
                class_name = class_names[cls] if cls < len(class_names) else str(cls)
                yield Record(
                    source=self.name,
                    image_path=self.paths.relative(image_path),
                    license_key=self.info.license_key,
                    plate=None,  # detection-only: no transcription in these exports
                    plate_type=plate_type_for_class(class_name),
                    box=box,
                    corners=None,  # box-only source: never fabricate corners
                    image_size=(width, height),
                    tags=("roboflow", self.project.project) + ((split,) if split else ()),
                    extra={"class": class_name},
                )

        self.last_index_errors = errors
        self.last_index_skipped = skipped

    @staticmethod
    def _label_for(image_path: Path) -> Path | None:
        """YOLO layout puts labels in a sibling `labels/` directory."""
        parts = list(image_path.parts)
        if "images" not in parts:
            return None
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return Path(*parts).with_suffix(".txt")

    @staticmethod
    def _split_for(image_path: Path) -> str:
        for part in image_path.parts:
            if part.lower() in ("train", "valid", "val", "test"):
                return part.lower()
        return ""


@register
class PlacasColombianasSource(RoboflowSource):
    """The one Colombian project with a verified, commercially usable licence."""

    project = RoboflowProject(workspace="licenseplates-gk27i", project="placas-colombianas")
    info = SourceInfo(
        name="roboflow_placas_colombianas",
        title="Roboflow placas-colombianas",
        license_key="cc-by-4.0",
        access=AccessMode.API_KEY,
        purposes=frozenset({Purpose.DETECTOR_FINETUNE, Purpose.BACKGROUNDS}),
        url=project.url,
        size_hint="~1,770 images",
        notes="CC BY 4.0 — attribution required in the writeup. Box-only annotations.",
    )


@register
class PlacasColombiaItmSource(RoboflowSource):
    """ITM project. Licence verified as MIT via the Roboflow API on 2026-08-09.

    The plan (§8) listed this as "unverified — check". The check has now been done:
    the API reports `license: "MIT"`, so it moves off the `unknown` key and onto the
    commercially clean path.
    """

    project = RoboflowProject(workspace="itm-mprof", project="placas-colombia")
    info = SourceInfo(
        name="roboflow_placas_colombia_itm",
        title="Roboflow placas-colombia (ITM)",
        license_key="mit",
        access=AccessMode.API_KEY,
        # Glyphs and recognizer evaluation stand in for the own photos until Phase 5
        # (provisional V0; see alpr/plates/testset.py). Backgrounds: vehicle fronts
        # with the plate painted out (alpr/synth/backgrounds.py).
        purposes=frozenset(
            {
                Purpose.DETECTOR_FINETUNE,
                Purpose.GLYPHS,
                Purpose.RECOGNIZER_EVAL,
                Purpose.BACKGROUNDS,
            }
        ),
        url=project.url,
        size_hint="~100 images",
        notes=(
            "Licence MIT, read from the Roboflow API 2026-08-09. Images are stretched "
            "to 640x640 upstream, so their aspect ratio is distorted."
        ),
    )


@register
class ProyectoPlacasSource(RoboflowSource):
    """Third project. Licence verified as CC BY 4.0 via the Roboflow API on 2026-08-09.

    Pinned to **version 1**: it is the only un-augmented version. Version 2 inflates
    274 real images to 664 by shearing and blurring each one three times, which adds
    no information the Phase 3 degradation pipeline does not already produce under
    our own control — and makes the dataset look 2.4x larger than it is.
    """

    project = RoboflowProject(
        workspace="placas-colombia", project="proyecto-placas-8arfj", version=1
    )
    info = SourceInfo(
        name="roboflow_proyecto_placas",
        title="Roboflow proyecto-placas",
        license_key="cc-by-4.0",
        access=AccessMode.API_KEY,
        # Glyphs and recognizer evaluation stand in for the own photos until Phase 5
        # (provisional V0; see alpr/plates/testset.py). Backgrounds: vehicle fronts
        # with the plate painted out (alpr/synth/backgrounds.py).
        purposes=frozenset(
            {
                Purpose.DETECTOR_FINETUNE,
                Purpose.GLYPHS,
                Purpose.RECOGNIZER_EVAL,
                Purpose.BACKGROUNDS,
            }
        ),
        url=project.url,
        size_hint="274 images (v1, un-augmented)",
        notes=(
            "Licence CC BY 4.0, read from the Roboflow API 2026-08-09 — attribution "
            "owed in the writeup. Pinned to v1; v2 bakes in shear and blur."
        ),
    )
