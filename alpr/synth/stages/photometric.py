"""Stage 4 — photometric: what scene radiance reaches the lens, then the tone curve.

Exposure, white balance, cast shadows, specular bands, headlight bloom and the
infrared mode act on *linear* radiance, so they come before blur and noise; the
gamma (tone curve) closes the stage, as in the plan's order — applying it after
noise would stretch the noise non-linearly, which no sensor does.

Float32 RGB in [0, 1] (sRGB-encoded) in and out. `plate_mask` is the plate's alpha
at this resolution; `corners` place the plate-relative effects.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from alpr.synth import augment


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def _plate_frame(corners: np.ndarray) -> tuple[np.ndarray, float, float]:
    c = np.asarray(corners, float)
    centre = c.mean(axis=0)
    w = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    h = (np.linalg.norm(c[3] - c[0]) + np.linalg.norm(c[2] - c[1])) / 2
    return centre, float(w), float(h)


def band_coordinate(
    shape: tuple[int, int], corners: np.ndarray, angle_deg: float, offset: float
) -> np.ndarray:
    """Signed distance (in plate widths) from a line through the plate at `angle_deg`,
    shifted by `offset` plate widths from the centre."""
    (cx, cy), w, _ = _plate_frame(corners)
    h_img, w_img = shape
    ys, xs = np.mgrid[0:h_img, 0:w_img].astype(np.float32)
    a = math.radians(angle_deg)
    return ((xs + 0.5 - cx) * math.cos(a) + (ys + 0.5 - cy) * math.sin(a)) / w - offset


def apply_shadow(lin: np.ndarray, corners: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    d = band_coordinate(lin.shape[:2], corners, spec["angle_deg"], spec["offset"])
    t = np.clip(0.5 + d / max(spec["softness"], 1e-3), 0.0, 1.0)
    return lin * (1.0 - spec["strength"] * t)[..., None]


def apply_specular(
    lin: np.ndarray, corners: np.ndarray, plate_mask: np.ndarray, spec: dict[str, Any]
) -> np.ndarray:
    d = band_coordinate(lin.shape[:2], corners, spec["angle_deg"], spec["offset"])
    band = np.exp(-0.5 * (d / max(spec["width"], 1e-3)) ** 2) * spec["strength"]
    return lin + (band * plate_mask)[..., None]


def photometric(
    image: np.ndarray,
    corners: np.ndarray,
    plate_mask: np.ndarray,
    *,
    exposure_log2: float = 0.0,
    temp_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    gamma: float = 1.0,
    shadow: dict[str, Any] | None = None,
    specular: dict[str, Any] | None = None,
    bloom: dict[str, Any] | None = None,
    ir: bool = False,
) -> np.ndarray:
    lin = srgb_to_linear(image)
    # IR replaces colour entirely, so white balance only applies to visible light.
    lin = augment.infrared(lin, plate_mask) if ir else lin * np.asarray(temp_gains, np.float32)
    lin = lin * np.float32(2.0**exposure_log2)
    if shadow:
        lin = apply_shadow(lin, corners, shadow)
    if specular:
        lin = apply_specular(lin, corners, plate_mask, specular)
    if bloom:
        lin = augment.bloom(lin, corners, bloom)
    out = linear_to_srgb(lin)
    if gamma != 1.0:
        out = out ** np.float32(gamma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
