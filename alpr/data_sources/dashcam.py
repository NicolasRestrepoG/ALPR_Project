"""YouTube dashcam footage — backgrounds and unlabelled real frames.

Two uses, both secondary: vehicle-front backgrounds for the synthetic compositor
(Sprint 3.1), and unlabelled frames for the hard-negative mining pass in Sprint 5.2.
Neither needs many videos.

**Licensing.** YouTube's Terms of Service govern, per video. `yt-dlp` exposes the
uploader's licence choice, so this module can prefer Creative Commons videos and
refuse everything else with `--cc-only`. Even so the source is marked
non-commercial: "Creative Commons on YouTube" is a claim by an uploader who may not
hold the rights, and a gate system is not the place to rely on that. Frames are
research-grade input, not shippable training data.

Provenance is recorded per frame, not per video: every extracted frame carries the
video URL it came from, and the `youtube-tos` licence requires that URL at the type
level, so a frame without provenance cannot become a Record.

**External tools.** `yt-dlp` (a dependency) and `ffmpeg` (a system binary) are both
invoked as subprocesses. Neither is imported, so this module still costs nothing at
import time.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import asdict, dataclass
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
from alpr.data_sources.storage import IMAGE_SUFFIXES, atomic_write_text, iter_files

VIDEO_LIST = "videos.txt"
PROVENANCE_NAME = "provenance.json"

#: Frames differing from the previous one by more than this are treated as a new
#: scene. 0.3 is loose enough to catch a change of road and tight enough not to
#: fire on every lane marking.
SCENE_THRESHOLD = 0.3

CC_LICENSE_MARKER = "creative commons"


class DashcamError(AcquisitionError):
    """A video could not be fetched, or frames could not be extracted."""


@dataclass(frozen=True, slots=True)
class VideoProvenance:
    """Everything needed to attribute a frame, recorded at download time."""

    video_id: str
    url: str
    title: str
    uploader: str
    license: str
    duration_s: float
    frames: int

    @property
    def creative_commons(self) -> bool:
        return CC_LICENSE_MARKER in (self.license or "").lower()


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def have_yt_dlp() -> bool:
    return shutil.which("yt-dlp") is not None


def read_video_list(path: Path | str) -> list[str]:
    """One URL per line; `#` comments and blank lines ignored."""
    path = Path(path)
    if not path.exists():
        return []
    urls = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            urls.append(line)
    return urls


def video_metadata(url: str, *, timeout: float = 120.0) -> VideoProvenance:
    """Ask yt-dlp for metadata without downloading the video."""
    if not have_yt_dlp():
        raise DashcamError(
            "yt-dlp is not on PATH. Install the Phase 1 dependencies:\n  pip install -e '.[dev]'"
        )
    proc = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-warnings", "--skip-download", url],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise DashcamError(f"yt-dlp could not read {url}: {proc.stderr.strip()[:400]}")
    try:
        meta = json.loads(proc.stdout.splitlines()[0])
    except (json.JSONDecodeError, IndexError) as exc:
        raise DashcamError(f"yt-dlp returned unparseable metadata for {url}") from exc
    return VideoProvenance(
        video_id=meta.get("id", ""),
        url=meta.get("webpage_url", url),
        title=meta.get("title", ""),
        uploader=meta.get("uploader", ""),
        license=meta.get("license") or "",
        duration_s=float(meta.get("duration") or 0.0),
        frames=0,
    )


def download_video(url: str, dest_dir: Path, *, max_height: int = 720) -> Path:
    """Download one video at a capped resolution. Returns the file path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    template = str(dest_dir / "%(id)s.%(ext)s")
    proc = subprocess.run(
        [
            "yt-dlp",
            "-f",
            f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]",
            "--no-playlist",
            "--no-warnings",
            "-o",
            template,
            url,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DashcamError(f"yt-dlp failed on {url}: {proc.stderr.strip()[:400]}")
    candidates = sorted(dest_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    videos = [p for p in candidates if p.suffix.lower() in (".mp4", ".mkv", ".webm")]
    if not videos:
        raise DashcamError(f"yt-dlp reported success but produced no video file for {url}")
    return videos[0]


def extract_scene_frames(
    video: Path,
    dest_dir: Path,
    *,
    threshold: float = SCENE_THRESHOLD,
    max_frames: int = 200,
    prefix: str = "",
) -> list[Path]:
    """Extract one JPEG per detected scene change.

    Scene-cut extraction rather than fixed-interval sampling: a dashcam sitting at a
    red light produces hundreds of near-identical frames, and those add nothing but
    a skewed background distribution.
    """
    if not have_ffmpeg():
        raise DashcamError(
            "ffmpeg is not on PATH. Install it with:  brew install ffmpeg\n"
            "(it is a system binary, not a Python package)"
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(dest_dir / f"{prefix}%05d.jpg")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select='gt(scene,{threshold})'",
            "-vsync",
            "vfr",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "2",
            pattern,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise DashcamError(f"ffmpeg failed on {video.name}: {proc.stderr.strip()[:400]}")
    return sorted(dest_dir.glob(f"{prefix}*.jpg"))


@register
class DashcamSource(Source):
    """Frames from Creative-Commons-preferred YouTube dashcam videos."""

    info = SourceInfo(
        name="dashcam",
        title="YouTube dashcam frames",
        license_key="youtube-tos",
        access=AccessMode.MANUAL,
        purposes=frozenset({Purpose.BACKGROUNDS, Purpose.REAL_TEST}),
        url="https://github.com/yt-dlp/yt-dlp",
        size_hint="unbounded; a handful of videos is plenty",
        notes=(
            "Per-video terms. Prefer Creative Commons. Backgrounds and hard-negative "
            "mining only — not commercially usable."
        ),
    )

    last_index_skipped: int = 0
    last_index_errors: list[str] = []

    @property
    def video_list_path(self) -> Path:
        return self.raw_dir / VIDEO_LIST

    @property
    def videos_dir(self) -> Path:
        return self.raw_dir / "videos"

    @property
    def frames_dir(self) -> Path:
        return self.raw_dir / "frames"

    @property
    def provenance_path(self) -> Path:
        return self.raw_dir / PROVENANCE_NAME

    def load_provenance(self) -> dict[str, VideoProvenance]:
        if not self.provenance_path.exists():
            return {}
        raw = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        return {k: VideoProvenance(**v) for k, v in raw.items()}

    def save_provenance(self, records: dict[str, VideoProvenance]) -> None:
        atomic_write_text(
            self.provenance_path,
            json.dumps({k: asdict(v) for k, v in records.items()}, indent=2, sort_keys=True) + "\n",
        )

    def fetch(
        self,
        *,
        force: bool = False,
        cc_only: bool = True,
        max_frames: int = 200,
        keep_videos: bool = False,
    ) -> FetchResult:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        urls = read_video_list(self.video_list_path)
        if not urls:
            self._write_video_list_template()
            return FetchResult(
                source=self.name,
                status=Status.MANUAL_REQUIRED,
                message=self.manual_instructions(),
            )
        missing_tools = [
            name
            for name, present in (("yt-dlp", have_yt_dlp()), ("ffmpeg", have_ffmpeg()))
            if not present
        ]
        if missing_tools:
            return FetchResult(
                source=self.name,
                status=Status.UNAVAILABLE,
                message=(
                    f"missing external tool(s): {', '.join(missing_tools)}\n"
                    "  yt-dlp:  pip install -e '.[dev]'\n"
                    "  ffmpeg:  brew install ffmpeg"
                ),
            )

        provenance = self.load_provenance()
        skipped_non_cc: list[str] = []
        total_frames = 0

        for url in urls:
            meta = video_metadata(url)
            if meta.video_id in provenance and not force:
                continue
            if cc_only and not meta.creative_commons:
                skipped_non_cc.append(f"{meta.video_id} ({meta.license or 'standard licence'})")
                continue
            video = download_video(url, self.videos_dir)
            frames = extract_scene_frames(
                video,
                self.frames_dir / meta.video_id,
                max_frames=max_frames,
                prefix=f"{meta.video_id}_",
            )
            total_frames += len(frames)
            provenance[meta.video_id] = VideoProvenance(**{**asdict(meta), "frames": len(frames)})
            self.save_provenance(provenance)
            if not keep_videos:
                video.unlink(missing_ok=True)  # frames are the artifact; video is 10x the size

        message = f"{total_frames} frame(s) from {len(provenance)} video(s)"
        if skipped_non_cc:
            message += f"; skipped {len(skipped_non_cc)} non-CC video(s): " + ", ".join(
                skipped_non_cc[:5]
            )
        return FetchResult(
            source=self.name,
            status=Status.DOWNLOADED if total_frames else Status.READY,
            message=message,
            paths=[str(self.frames_dir)],
        )

    def _write_video_list_template(self) -> None:
        if self.video_list_path.exists():
            return
        atomic_write_text(
            self.video_list_path,
            "# One YouTube URL per line. Lines starting with # are ignored.\n"
            "# Prefer Creative Commons videos; --cc-only refuses the rest.\n"
            "# Search YouTube for: dashcam colombia bogota medellin 4k\n"
            "#\n"
            "# https://www.youtube.com/watch?v=EXAMPLE\n",
        )

    def manual_instructions(self) -> str:
        return f"""\
No dashcam videos listed yet.

  1. Find Colombian dashcam footage on YouTube. Filter to Creative Commons:
     add "&sp=EgIwAQ%253D%253D" to a search URL, or use the Features filter.
  2. Put one URL per line in:  {self.video_list_path}
  3. Run `acquire fetch dashcam`. Each video is downloaded at <=720p, one frame
     per scene change is extracted, and the video file is then deleted.

Per-frame provenance is written to {self.provenance_path} and every frame Record
carries its video URL — the `youtube-tos` licence requires it at the type level, so
a frame without provenance cannot be indexed.

Needs `ffmpeg` on PATH ({"found" if have_ffmpeg() else "NOT FOUND — brew install ffmpeg"}).

LICENCE: per-video YouTube terms. Backgrounds and hard-negative mining only; not
commercially usable, and `acquire audit` will say so."""

    def index(self) -> Iterator[Record]:
        provenance = self.load_provenance()
        skipped: list[str] = []
        for frame in iter_files(self.frames_dir, IMAGE_SUFFIXES):
            video_id = self._video_id_for(frame)
            meta = provenance.get(video_id)
            if meta is None:
                # No provenance means no attribution, which means it cannot be used.
                skipped.append(frame.name)
                continue
            yield Record(
                source=self.name,
                image_path=self.paths.relative(frame),
                license_key=self.info.license_key,
                plate=None,
                plate_type=PlateType.UNKNOWN,
                image_size=safe_image_size(frame),
                attribution_url=meta.url,
                tags=("dashcam", "unlabelled", "non_commercial"),
                extra={"video_id": video_id, "uploader": meta.uploader, "license": meta.license},
            )
        self.last_index_skipped = len(skipped)
        self.last_index_errors = [
            f"{name}: no entry in {PROVENANCE_NAME} — refused (attribution required)"
            for name in skipped[:20]
        ]

    @staticmethod
    def _video_id_for(frame: Path) -> str:
        """Frames live in `frames/<video_id>/<video_id>_00001.jpg`."""
        if frame.parent.name and frame.parent.name != "frames":
            return frame.parent.name
        match = re.match(r"^(.+)_\d+$", frame.stem)
        return match.group(1) if match else ""
