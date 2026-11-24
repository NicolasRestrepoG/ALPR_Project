"""Header-only image dimension reading, with no image library installed."""

from __future__ import annotations

import pytest

from alpr.data_sources.imagesize import ImageSizeError, image_size, safe_image_size

from .factories import make_bmp, make_gif, make_jpeg, make_png


def test_png(tmp_path):
    assert image_size(make_png(tmp_path / "a.png", 1920, 1080)) == (1920, 1080)


@pytest.mark.parametrize("extra", [True, False])
def test_jpeg(tmp_path, extra):
    """The marker walk must skip APP0/DQT segments to reach the frame header."""
    path = make_jpeg(tmp_path / "a.jpg", 4032, 3024, with_extra_segments=extra)
    assert image_size(path) == (4032, 3024)


def test_gif(tmp_path):
    assert image_size(make_gif(tmp_path / "a.gif", 320, 240)) == (320, 240)


def test_bmp_with_negative_height(tmp_path):
    """Bottom-up BMPs store a negative height; the magnitude is what matters."""
    assert image_size(make_bmp(tmp_path / "a.bmp", 100, -50)) == (100, 50)


def test_ccpd_frame_dimensions(tmp_path):
    """CCPD frames are 720x1160 — the size the area-ratio test depends on."""
    assert image_size(make_jpeg(tmp_path / "ccpd.jpg", 720, 1160)) == (720, 1160)


def test_unrecognised_format_raises(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"this is definitely not an image file at all, no")
    with pytest.raises(ImageSizeError, match="unrecognised"):
        image_size(path)


def test_short_file_raises(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"\xff\xd8")
    with pytest.raises(ImageSizeError, match="too short"):
        image_size(path)


def test_truncated_jpeg_raises(tmp_path):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"\xff\xd8" + b"\x00" * 40)
    with pytest.raises(ImageSizeError):
        image_size(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageSizeError):
        image_size(tmp_path / "nope.jpg")


def test_safe_image_size_returns_none(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"x" * 64)
    assert safe_image_size(path) is None
    assert safe_image_size(make_png(tmp_path / "b.png", 8, 8)) == (8, 8)
