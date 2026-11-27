"""Each degradation stage alone: shape, dtype, range, and the physics it claims."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from alpr.synth.stages import ORDER
from alpr.synth.stages.blur import blur, defocus_kernel, motion_kernel
from alpr.synth.stages.composite import composite, place_background, plain_background
from alpr.synth.stages.downsample import downsample, plate_width
from alpr.synth.stages.jpeg import jpeg
from alpr.synth.stages.noise import noise
from alpr.synth.stages.photometric import linear_to_srgb, photometric, srgb_to_linear
from alpr.synth.stages.warp import project_corners, warp

from .conftest import SQUARE


def test_order_is_the_plan():
    assert ORDER == ("warp", "composite", "downsample", "photometric", "blur", "noise", "jpeg")


# -- 1 warp ------------------------------------------------------------------- #


def test_frontal_warp_keeps_the_plate_a_rectangle_at_render_scale():
    plate = np.full((640, 1320, 3), 200, np.uint8)
    warped, alpha, c = warp(plate, 0, 0, 0, 5000, 4.0)
    assert warped.dtype == np.uint8 and alpha.dtype == np.float32
    assert plate_width(c) == pytest.approx(1320, abs=1)
    assert c[0][1] == pytest.approx(c[1][1]) and c[0][0] == pytest.approx(c[3][0])
    cx, cy = c.mean(axis=0).astype(int)
    assert alpha[cy, cx] == pytest.approx(1.0) and alpha[2, 2] == 0.0
    assert (warped[cy, cx] == 200).all()


def test_yaw_shortens_the_far_side_and_corners_stay_ordered():
    c = project_corners((1320, 640), 25, 0, 0, 3000, 4.0)
    left, right = np.linalg.norm(c[3] - c[0]), np.linalg.norm(c[2] - c[1])
    assert left != pytest.approx(right, rel=0.01)
    assert c[0][0] < c[1][0] and c[3][0] < c[2][0]  # TL left of TR, BL left of BR
    assert c[0][1] < c[3][1] and c[1][1] < c[2][1]  # tops above bottoms


def test_roll_tilts_the_top_edge():
    c = project_corners((1320, 640), 0, 0, 8, 5000, 4.0)
    assert c[1][1] - c[0][1] == pytest.approx(1320 * np.sin(np.radians(8)), rel=0.02)


# -- 2 composite -------------------------------------------------------------- #


def test_composite_obeys_alpha():
    fg = np.full((10, 10, 3), 255, np.uint8)
    bg = np.zeros((10, 10, 3), np.uint8)
    a = np.zeros((10, 10), np.float32)
    a[:, 5:] = 1.0
    a[:, 4] = 0.5
    out = composite(fg, a, bg)
    assert out.dtype == np.uint8
    assert out[0, 0, 0] == 0 and out[0, 9, 0] == 255 and out[0, 4, 0] == pytest.approx(128, abs=1)
    with pytest.raises(ValueError):
        composite(fg, a[:5], bg)


def test_background_plate_lands_under_the_new_plate():
    bg = np.zeros((100, 200, 3), np.uint8)
    quad = np.array([(50.0, 40.0), (150.0, 40.0), (150.0, 60.0), (50.0, 60.0)])
    cv2.rectangle(bg, (50, 40), (149, 59), (255, 0, 0), -1)
    out = place_background(bg, quad, quad * 2, (400, 200))
    assert tuple(out[100, 200]) == (255, 0, 0)  # centre of the doubled quad
    assert plain_background((40, 30), np.random.default_rng(0)).shape == (30, 40, 3)


# -- 3 downsample ------------------------------------------------------------- #


def test_downsample_hits_the_target_width_and_preserves_mean(scene):
    out, c = downsample(scene, SQUARE, 50)
    assert plate_width(c) == pytest.approx(50, rel=0.02)
    assert out.mean() == pytest.approx(scene.mean(), abs=0.01)
    assert out.dtype == np.float32


def test_downsample_never_enlarges(scene):
    out, c = downsample(scene, SQUARE, 10_000)
    assert out.shape == scene.shape and np.allclose(c, SQUARE)


# -- 4 photometric ------------------------------------------------------------ #


def test_srgb_round_trip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_neutral_photometric_is_identity(scene):
    mask = np.zeros(scene.shape[:2], np.float32)
    out = photometric(scene, SQUARE, mask)
    assert np.allclose(out, scene, atol=1e-4)


def test_exposure_white_balance_and_gamma(scene):
    mask = np.zeros(scene.shape[:2], np.float32)
    brighter = photometric(scene, SQUARE, mask, exposure_log2=1.0)
    assert brighter.mean() > scene.mean()
    warm = photometric(scene, SQUARE, mask, temp_gains=(1.2, 1.0, 0.8))
    assert warm[..., 0].mean() > scene[..., 0].mean() and warm[..., 2].mean() < scene[..., 2].mean()
    assert photometric(scene, SQUARE, mask, gamma=1.3).mean() < scene.mean()
    for out in (brighter, warm):
        assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1


def test_shadow_darkens_one_side(scene):
    mask = np.zeros(scene.shape[:2], np.float32)
    spec = {"strength": 0.6, "angle_deg": 0.0, "offset": 0.0, "softness": 0.05}
    out = photometric(scene, SQUARE, mask, shadow=spec)
    assert out[:, 250:].mean() < scene[:, 250:].mean() - 0.05  # right of centre
    assert out[:, :70].mean() == pytest.approx(scene[:, :70].mean(), abs=0.01)


def test_specular_brightens_only_the_plate(scene):
    mask = np.zeros(scene.shape[:2], np.float32)
    mask[100:140, 60:260] = 1
    spec = {"strength": 0.4, "angle_deg": 90.0, "offset": 0.0, "width": 0.3}
    out = photometric(scene, SQUARE, mask, specular=spec)
    assert out[110:130, 130:250].mean() > scene[110:130, 130:250].mean()
    assert np.allclose(out[:90], scene[:90], atol=1e-4)


def test_bloom_and_infrared(scene):
    mask = np.zeros(scene.shape[:2], np.float32)
    mask[100:140, 60:260] = 1
    bloom = {"strength": 1.0, "x": 0.0, "y": -1.5, "radius": 0.5, "ghosts": 2}
    out = photometric(scene, SQUARE, mask, bloom=bloom)
    assert out[60:90, 140:180].mean() > scene[60:90, 140:180].mean() + 0.05  # above plate
    ir = photometric(scene, SQUARE, mask, ir=True)
    chroma = np.abs(ir[..., 0] - ir[..., 1]).mean()
    assert chroma < 0.02  # colour is gone
    assert ir[105:110, 130:250].mean() > ir[:50].mean()  # the sheet glows, the scene falls off


# -- 5 blur ------------------------------------------------------------------- #


@pytest.mark.parametrize("k", [motion_kernel(6, 30), defocus_kernel(1.5)])
def test_kernels_are_normalised(k):
    assert k.sum() == pytest.approx(1.0, abs=1e-5) and (k >= 0).all()


def test_blur_identity_mean_and_sharpness(scene):
    assert np.array_equal(blur(scene), scene)
    out = blur(scene, defocus_px=1.5, motion_px=5, motion_angle_deg=0)
    assert out.mean() == pytest.approx(scene.mean(), abs=0.005)
    edge = lambda im: np.abs(np.diff(im, axis=1)).mean()  # noqa: E731
    assert edge(out) < edge(scene)


# -- 6 noise ------------------------------------------------------------------ #


def test_noise_identity_mean_and_range(scene):
    rng = np.random.default_rng(0)
    assert np.array_equal(noise(scene, rng), scene)
    out = noise(scene, rng, shot_peak=200, read_sigma=0.02)
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1
    assert out.mean() == pytest.approx(scene.mean(), abs=0.01)


def test_noise_is_independent_per_pixel_and_scales_with_light():
    flat = np.full((200, 200, 3), 0.5, np.float32)
    rng = np.random.default_rng(1)
    n = noise(flat, rng, shot_peak=100, read_sigma=0.01)[..., 0] - 0.5
    corr = np.corrcoef(n[:, :-1].ravel(), n[:, 1:].ravel())[0, 1]
    assert abs(corr) < 0.05  # no correlation between neighbours
    quiet = noise(flat, rng, shot_peak=10_000)[..., 0].std()
    assert n.std() > 3 * quiet  # fewer photons, more noise


# -- 7 jpeg ------------------------------------------------------------------- #


def test_jpeg_quality_controls_error(scene):
    hi, lo = jpeg(scene, 95), jpeg(scene, 20)
    assert hi.dtype == np.uint8 and hi.shape == scene.shape
    ref = scene * 255
    assert np.abs(lo - ref).mean() > np.abs(hi - ref).mean()
    assert np.abs(jpeg(scene, 100).astype(float) - ref).max() <= 0.5 + 1e-3
