"""Byte-level image factories shared across the acquisition tests.

Phase 1 has no image library, so tests that need a real JPEG or PNG on disk build
one from its header bytes. These are structurally valid files — `image_size` walks
them the same way it walks a photograph.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def make_png(path: Path, width: int, height: int) -> Path:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b""))
    return path


def make_jpeg(path: Path, width: int, height: int, *, with_extra_segments: bool = True) -> Path:
    """A minimal but structurally honest JPEG: SOI, optional segments, SOF0, EOI."""
    out = bytearray(b"\xff\xd8")
    if with_extra_segments:
        app0 = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        out += b"\xff\xe0" + struct.pack(">H", len(app0) + 2) + app0
        dqt = bytes(65)
        out += b"\xff\xdb" + struct.pack(">H", len(dqt) + 2) + dqt
    sof = struct.pack(">BHHB", 8, height, width, 3) + bytes(9)
    out += b"\xff\xc0" + struct.pack(">H", len(sof) + 2) + sof
    out += b"\xff\xd9"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
    return path


def make_gif(path: Path, width: int, height: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GIF89a" + struct.pack("<HH", width, height) + bytes(20))
    return path


def make_bmp(path: Path, width: int, height: int) -> Path:
    header = bytearray(b"BM" + bytes(16))
    header += struct.pack("<ii", width, height)
    header += bytes(20)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(header))
    return path
