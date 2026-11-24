"""Read image dimensions from file headers, using only the standard library.

YOLO annotations are normalized to [0, 1], so converting them to the pixel boxes a
`Record` stores requires the image's width and height. The obvious way to get that
is Pillow or OpenCV — but Phase 1 is deliberately free of image and ML libraries so
its tests run in milliseconds on any machine, and the `roboflow` SDK alone would
pull in numpy, opencv-python and more.

Every format here stores its dimensions in the first few hundred bytes, so reading
them costs one short seek rather than a decode.
"""

from __future__ import annotations

import struct
from pathlib import Path

from alpr.data_sources.base import AcquisitionError


class ImageSizeError(AcquisitionError):
    """The file is not a recognised image, or its header is truncated."""


def _jpeg_size(fh) -> tuple[int, int]:
    """Walk the JPEG marker segments to the frame header."""
    fh.seek(2)  # past SOI
    while True:
        byte = fh.read(1)
        while byte and byte != b"\xff":  # resync on fill bytes
            byte = fh.read(1)
        if not byte:
            raise ImageSizeError("truncated JPEG: no frame header found")
        marker = fh.read(1)
        while marker == b"\xff":  # padding
            marker = fh.read(1)
        if not marker:
            raise ImageSizeError("truncated JPEG: no frame header found")
        code = marker[0]
        # SOF0..SOF15 carry the dimensions; DHT/JPG/DAC are not frame headers.
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            fh.read(3)  # segment length (2) + sample precision (1)
            data = fh.read(4)
            if len(data) != 4:
                raise ImageSizeError("truncated JPEG frame header")
            height, width = struct.unpack(">HH", data)
            return width, height
        if code in (0xD8, 0xD9) or 0xD0 <= code <= 0xD7:
            continue  # standalone markers carry no length
        length_bytes = fh.read(2)
        if len(length_bytes) != 2:
            raise ImageSizeError("truncated JPEG segment")
        (length,) = struct.unpack(">H", length_bytes)
        fh.seek(length - 2, 1)


def image_size(path: Path | str) -> tuple[int, int]:
    """Return `(width, height)` in pixels. Raises `ImageSizeError` if unreadable."""
    path = Path(path)
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
            if len(head) < 16:
                raise ImageSizeError(f"{path}: file is too short to be an image")

            if head[:2] == b"\xff\xd8":
                return _jpeg_size(fh)

            if head[:8] == b"\x89PNG\r\n\x1a\n":
                width, height = struct.unpack(">II", head[16:24])
                return width, height

            if head[:6] in (b"GIF87a", b"GIF89a"):
                width, height = struct.unpack("<HH", head[6:10])
                return width, height

            if head[:2] == b"BM":
                fh.seek(18)
                width, height = struct.unpack("<ii", fh.read(8))
                return abs(width), abs(height)

            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                fourcc = head[12:16]
                if fourcc == b"VP8 ":
                    fh.seek(26)
                    width, height = struct.unpack("<HH", fh.read(4))
                    return width & 0x3FFF, height & 0x3FFF
                if fourcc == b"VP8L":
                    fh.seek(21)
                    (bits,) = struct.unpack("<I", fh.read(4))
                    return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
                if fourcc == b"VP8X":
                    fh.seek(24)
                    raw = fh.read(6)
                    width = raw[0] | raw[1] << 8 | raw[2] << 16
                    height = raw[3] | raw[4] << 8 | raw[5] << 16
                    return width + 1, height + 1
                raise ImageSizeError(f"{path}: unsupported WebP variant {fourcc!r}")
    except OSError as exc:
        raise ImageSizeError(f"{path}: {exc}") from exc

    raise ImageSizeError(f"{path}: unrecognised image format")


def safe_image_size(path: Path | str) -> tuple[int, int] | None:
    """`image_size` but returns None instead of raising, for bulk indexing."""
    try:
        return image_size(path)
    except ImageSizeError:
        return None
