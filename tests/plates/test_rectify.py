"""Rectification: upright, unmirrored, precise, and loud about impossible corners."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from alpr.data_sources import base
from alpr.plates import rectify as rectify_module
from alpr.plates.geometry import CANONICAL_SIZE
from alpr.plates.rectify import (
    ROTATIONS,
    RectifyError,
    check_quad,
    jitter_corners,
    load_image,
    rectify,
    rotate_image,
    rotate_points,
    unrotate_points,
)

from .conftest import SCENE_QUAD, place, quadrant_plate


def _quadrant_colours(flat: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = flat.shape[:2]
    spots = [(h // 4, w // 4), (h // 4, 3 * w // 4), (3 * h // 4, 3 * w // 4), (3 * h // 4, w // 4)]
    return [tuple(int(v) for v in flat[y, x]) for y, x in spots]


EXPECTED = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]  # TL, TR, BR, BL


def test_output_shape_and_dtype(scene_and_quad):
    scene, quad = scene_and_quad
    flat = rectify(scene, quad)
    assert flat.shape == (CANONICAL_SIZE[1], CANONICAL_SIZE[0], 3)
    assert flat.dtype == np.uint8


def test_plate_comes_out_upright_and_unmirrored(scene_and_quad):
    scene, quad = scene_and_quad
    got = _quadrant_colours(rectify(scene, quad))
    for g, e in zip(got, EXPECTED, strict=True):
        assert np.abs(np.array(g) - np.array(e)).max() < 10, (got, EXPECTED)


@pytest.mark.parametrize("perm", list(itertools.permutations(range(4))))
def test_any_corner_order_gives_the_same_crop(scene_and_quad, perm):
    """Corners are normalised by the shared helper, so input order cannot matter —
    including CCPD's right-bottom-first order, which has the *same* winding."""
    scene, quad = scene_and_quad
    reference = rectify(scene, quad)
    shuffled = [quad[k] for k in perm]
    assert np.array_equal(rectify(scene, shuffled), reference)


def test_round_trip_is_precise():
    """Warp a known plate into a scene, rectify it back: the result matches the
    plate downsampled directly, to within resampling noise."""
    plate = quadrant_plate()
    scene = place(plate)
    flat = rectify(scene, SCENE_QUAD).astype(int)
    direct = cv2.resize(plate, CANONICAL_SIZE, interpolation=cv2.INTER_AREA).astype(int)
    diff = np.abs(flat - direct)
    # Away from the colour boundaries the two must agree almost exactly.
    interior = np.ones(diff.shape[:2], bool)
    h, w = interior.shape
    interior[h // 2 - 2 : h // 2 + 2, :] = False
    interior[:, w // 2 - 2 : w // 2 + 2] = False
    interior[:2, :] = interior[-2:, :] = False
    interior[:, :2] = interior[:, -2:] = False
    assert diff[interior].mean() < 2.0
    assert diff[interior].max() < 40


def test_uses_the_shared_corner_helper():
    """One function orders corners for the sources, the tool and the rectifier."""
    assert rectify_module.order_corners is base.order_corners


@pytest.mark.parametrize(
    "corners, match",
    [
        ([(0, 0), (10, 10), (20, 20), (30, 30)], "degenerate"),  # collinear
        ([(0, 0), (0, 0), (100, 50), (0, 50)], "distinct"),  # coincident
        ([(0, 0), (5, 0), (5, 3), (0, 3)], "shorter than"),  # too small to mean anything
        ([(0, 0), (900, 0), (900, 20), (0, 20)], "aspect"),  # sliver
        ([(0, 0), (450, 0), (275, 200), (175, 200)], "differ"),  # extreme perspective
        ([(0, 0), (200, 0), (600, 100), (400, 100)], "angle"),  # sheared flat
    ],
)
def test_degenerate_quads_are_refused(corners, match):
    with pytest.raises(RectifyError, match=match):
        check_quad(corners)


def test_wrong_point_count_is_refused():
    with pytest.raises(RectifyError):
        check_quad([(0, 0), (10, 0), (10, 5)])


def test_check_quad_returns_canonical_order():
    tl, tr, br, bl = (10, 10), (200, 12), (198, 110), (12, 108)
    assert check_quad([br, tl, bl, tr]) == (tl, tr, br, bl)


def test_jitter_zero_is_identity_and_seeded_jitter_is_repeatable(scene_and_quad):
    _, quad = scene_and_quad
    assert jitter_corners(quad, 0.0) == check_quad(quad)
    a = jitter_corners(quad, 0.02, np.random.default_rng(7))
    b = jitter_corners(quad, 0.02, np.random.default_rng(7))
    assert a == b
    assert a != check_quad(quad)


def test_jitter_is_bounded_relative_to_plate_width(scene_and_quad):
    _, quad = scene_and_quad
    width = (
        np.hypot(*np.subtract(quad[1], quad[0])) + np.hypot(*np.subtract(quad[2], quad[3]))
    ) / 2
    rng = np.random.default_rng(0)
    for _ in range(200):
        moved = jitter_corners(quad, 0.02, rng)
        offsets = np.abs(np.array(moved) - np.array(check_quad(quad)))
        assert offsets.max() <= 3 * 0.02 * width + 1e-6


def test_rectify_with_jitter_still_upright(scene_and_quad):
    scene, quad = scene_and_quad
    flat = rectify(scene, quad, jitter=0.02, rng=np.random.default_rng(3))
    got = _quadrant_colours(flat)
    for g, e in zip(got, EXPECTED, strict=True):
        assert np.abs(np.array(g) - np.array(e)).max() < 10


def test_custom_size(scene_and_quad):
    scene, quad = scene_and_quad
    assert rectify(scene, quad, (1320, 640)).shape == (640, 1320, 3)


def test_load_image_returns_rgb(tmp_path):
    path = tmp_path / "red.png"
    bgr = np.zeros((4, 6, 3), np.uint8)
    bgr[..., 2] = 255  # red, in OpenCV's BGR order
    cv2.imwrite(str(path), bgr)
    rgb = load_image(path)
    assert rgb.shape == (4, 6, 3)
    assert tuple(rgb[0, 0]) == (255, 0, 0)


def test_load_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.jpg")
    junk = tmp_path / "junk.jpg"
    junk.write_bytes(b"not an image")
    with pytest.raises(RectifyError, match="decode"):
        load_image(junk)


def test_empty_image_is_refused():
    with pytest.raises(RectifyError):
        rectify(np.zeros((0, 0, 3), np.uint8), SCENE_QUAD)


@pytest.mark.parametrize("deg", ROTATIONS)
def test_rotate_points_follows_the_pixels(deg):
    """A marked pixel's centre, mapped by rotate_points, lands on the same pixel in
    the rotated image — and unrotate_points brings it back."""
    w, h = 7, 4
    img = np.zeros((h, w, 3), np.uint8)
    img[1, 5] = 255  # row 1, column 5
    out = rotate_image(img, deg)
    ((x, y),) = rotate_points([(5.5, 1.5)], deg, (w, h))
    assert out[int(y), int(x)].tolist() == [255, 255, 255]
    assert unrotate_points([(x, y)], deg, (w, h)) == pytest.approx([(5.5, 1.5)])


def test_why_turned_photos_must_be_stood_upright(scene_and_quad):
    """Corners are ordered by image position, so in a turned photo they cannot name the
    plate's own corners: a sideways plate is refused, an upside-down one comes out
    upside down. Hence the labeller's rotation, applied before any corner is used."""
    scene, quad = scene_and_quad
    size = (scene.shape[1], scene.shape[0])
    for deg in (90, 270):
        with pytest.raises(RectifyError, match="aspect"):
            rectify(rotate_image(scene, deg), rotate_points(quad, deg, size), CANONICAL_SIZE)
    flat = rectify(rotate_image(scene, 180), rotate_points(quad, 180, size), CANONICAL_SIZE)
    assert flat[20, 20].tolist() == pytest.approx([0, 0, 255], abs=12)  # blue where red belongs


def test_bad_rotation_is_refused(tmp_path):
    with pytest.raises(RectifyError, match="rotation"):
        rotate_image(np.zeros((2, 2, 3), np.uint8), 45)
    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((4, 6, 3), np.uint8))
    assert load_image(path, 90).shape == (6, 4, 3)
