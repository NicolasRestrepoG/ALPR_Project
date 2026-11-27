"""Front-view augmentations — the occluders and light sources a gate camera meets.

Each is applied by the pipeline at the point its physics belongs:

    on the flat plate, before warp    bolts, decorative frame, dirt   (coplanar objects)
    on the scene, before downsample   grille / bumper bars            (in front of it)
    in the photometric stage          headlight bloom, infrared mode  (radiance)

Probabilities live in `params.Ranges` (the plan's numbers); strengths scale with
severity there. Plate-level functions work on the uint8 render at 4 px/mm and take
positions in millimetres.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from alpr.plates.render import SLOT_CENTRES_MM

PX = 4.0

BOLT_TONES = {"chrome": (196, 196, 202), "black": (38, 38, 40), "rusty": (122, 84, 52)}
FRAME_TONES = {"chrome": (205, 205, 212), "black": (28, 28, 30), "white": (236, 236, 236)}
FRAME_TEXTS = ("AUTOMOTORES", "CONCESIONARIO", "AUTOS", "MOTORS", "TALLER")
DIRT_COLOR = np.array([92, 78, 58], np.float32)


# --------------------------------------------------------------------------- #
# On the flat plate
# --------------------------------------------------------------------------- #


def bolts(
    plate: np.ndarray, specs: tuple[dict[str, Any], ...] | list[dict[str, Any]]
) -> np.ndarray:
    """Domed bolt heads at the mounting slots, lit from the top-left, with a slot."""
    out = plate.copy()
    h, w = out.shape[:2]
    for b in specs:
        sx, sy = SLOT_CENTRES_MM[b["slot"]]
        cx, cy = (sx + b["dx_mm"]) * PX, (sy + b["dy_mm"]) * PX
        r = b["radius_mm"] * PX
        x0, y0 = max(0, int(cx - r - 2)), max(0, int(cy - r - 2))
        x1, y1 = min(w, int(cx + r + 3)), min(h, int(cy + r + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy) / r
        inside = np.clip((1.0 - d) * r, 0.0, 1.0)  # 1 px anti-aliased edge
        # Dome shading: brighter towards the top-left, darker rim.
        lit = 1.05 - 0.35 * d**2 - 0.25 * ((xs - cx) + (ys - cy)) / (2 * r)
        tone = np.asarray(BOLT_TONES[b["tone"]], np.float32) * lit[..., None]
        # A screwdriver slot across the head.
        slot = (np.abs(ys - cy) < max(1.0, 0.12 * r)) & (d < 0.7)
        tone[slot] *= 0.35
        patch = out[y0:y1, x0:x1].astype(np.float32)
        a = inside[..., None]
        out[y0:y1, x0:x1] = np.clip(tone * a + patch * (1 - a), 0, 255).astype(np.uint8)
    return out


def _rounded_rect_mask(
    w: int, h: int, x0: float, y0: float, x1: float, y1: float, r: float
) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    r = max(1, int(min(r, (x1 - x0) / 2, (y1 - y0) / 2)))
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 1, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 1, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, 1, -1)
    return m


def frame(
    plate: np.ndarray, spec: dict[str, Any], rng: np.random.Generator | None = None
) -> np.ndarray:
    """A decorative frame: a rounded-rect annulus over the plate's edge, often with a
    thicker bottom bar carrying dealer text over the municipality strip."""
    out = plate.copy()
    h, w = out.shape[:2]
    t = spec["thickness_mm"] * PX
    bottom = t + spec.get("bottom_extra_mm", 0.0) * PX
    inner = _rounded_rect_mask(w, h, t, t, w - t, h - bottom, spec["radius_mm"] * PX)
    ring = (1 - inner).astype(np.float32)
    ring = cv2.GaussianBlur(ring, (3, 3), 0.7)[..., None]
    base = np.asarray(FRAME_TONES[spec["material"]], np.float32)
    shade = np.linspace(1.08, 0.82, h, dtype=np.float32)[:, None, None]
    tone = np.broadcast_to(base, out.shape) * shade
    res = tone * ring + out.astype(np.float32) * (1 - ring)
    out = np.clip(res, 0, 255).astype(np.uint8)
    if spec.get("text") and bottom >= 9 * PX:
        rng = rng or np.random.default_rng(0)
        text = str(rng.choice(FRAME_TEXTS))
        ink = (20, 20, 20) if spec["material"] != "black" else (225, 225, 225)
        scale = (bottom - 2 * PX) / 30.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, int(scale * 2)))
        org = (int((w - tw) / 2), int(h - (bottom - th) / 2))
        cv2.putText(
            out,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            ink,
            max(1, int(scale * 2)),
            cv2.LINE_AA,
        )
    return out


def value_noise(
    shape: tuple[int, int], scale_px: float, rng: np.random.Generator, octaves: int = 3
) -> np.ndarray:
    """Smooth fractal noise in [0, 1] (a cheap stand-in for Perlin)."""
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        s = max(2.0, scale_px / (2**o))
        grid = rng.random((int(h / s) + 3, int(w / s) + 3)).astype(np.float32)
        acc += amp * cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.5
    acc /= total
    return (acc - acc.min()) / max(float(acc.max() - acc.min()), 1e-6)


def dirt(plate: np.ndarray, spec: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Grime (broad, soft, partly transparent) or spatter (fine specks), covering
    ~`coverage` of the plate. Grime darkens and browns the sheet in smudges; it is
    not the hard-edged spots a single threshold gives."""
    cov = float(spec["coverage"])
    if cov <= 0:
        return plate
    n = value_noise(plate.shape[:2], spec["scale_mm"] * PX, rng)
    thresh = float(np.quantile(n, 1.0 - cov))
    soft = 0.25 if spec.get("kind", "grime") == "grime" else 0.03
    a = np.clip((n - thresh) / soft, 0.0, 1.0) * float(spec.get("opacity", 0.85))
    color = DIRT_COLOR * (float(spec.get("tone", 0.5)) / 0.5)
    out = plate.astype(np.float32) * (1 - a[..., None]) + color * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def wear(
    plate: np.ndarray, spec: dict[str, Any], rng: np.random.Generator, sheet_rgb: tuple[int, ...]
) -> np.ndarray:
    """Worn paint: the ink fades towards the sheet colour in patches, as on old plates."""
    amount = float(spec["amount"])
    if amount <= 0:
        return plate
    f = plate.astype(np.float32)
    lum = f @ np.array([0.3, 0.6, 0.1], np.float32)
    ink = np.clip((140.0 - lum) / 80.0, 0.0, 1.0)  # how ink-like each pixel is
    n = value_noise(plate.shape[:2], spec["scale_mm"] * PX, rng)
    a = (ink * np.clip(n * 1.6 - 0.4, 0.0, 1.0) * amount)[..., None]
    sheet = np.asarray(sheet_rgb, np.float32) * 0.8  # faded paint, not clean sheet
    return np.clip(f * (1 - a) + sheet * a, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# On the scene, before the downsample
# --------------------------------------------------------------------------- #


def grille(
    image: np.ndarray, background: np.ndarray, corners: np.ndarray, spec: dict[str, Any]
) -> np.ndarray:
    """Horizontal bars across the plate, their pixels taken from the background's own
    rows above the plate, so they look like that vehicle's grille or bumper."""
    out = image.copy()
    h, w = out.shape[:2]
    c = np.asarray(corners, float)
    top, bottom = c[:, 1].min(), c[:, 1].max()
    ph = bottom - top
    for k in range(spec["bars"]):
        y = top + spec["positions"][k] * ph
        t = max(2, int(spec["thickness"] * ph))
        y0, y1 = int(max(0, y - t / 2)), int(min(h, y + t / 2))
        src0 = int(max(0, top - 1.5 * t - (y1 - y0)))
        rows = background[src0 : src0 + (y1 - y0)]
        if y1 <= y0 or rows.shape[0] != (y1 - y0):
            continue
        shade = np.linspace(1.1, 0.75, y1 - y0, dtype=np.float32)[:, None, None]
        out[y0:y1] = np.clip(rows * shade, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# Radiance (called from the photometric stage, on linear light)
# --------------------------------------------------------------------------- #


def bloom(lin: np.ndarray, corners: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """An oncoming headlight above the plate, plus ghosts mirrored through the image
    centre — the path of internal lens reflections."""
    h, w = lin.shape[:2]
    c = np.asarray(corners, float)
    centre = c.mean(axis=0)
    pw = (np.linalg.norm(c[1] - c[0]) + np.linalg.norm(c[2] - c[3])) / 2
    ph = pw / 2.06
    p0 = centre + np.array([spec["x"] * pw, spec["y"] * ph])
    sigma = spec["radius"] * ph
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    def lobe(p: np.ndarray, s: float, k: float, tint: tuple[float, float, float]) -> np.ndarray:
        d2 = (xs + 0.5 - p[0]) ** 2 + (ys + 0.5 - p[1]) ** 2
        return (k * np.exp(-d2 / (2 * s * s)))[..., None] * np.asarray(tint, np.float32)

    out = lin + lobe(p0, sigma, spec["strength"], (1.0, 0.96, 0.85))
    frame_c = np.array([w / 2, h / 2])
    for g in range(spec["ghosts"]):
        t = (0.5, 1.1, 0.25)[g % 3]
        p = frame_c - t * (p0 - frame_c)
        out = out + lobe(p, sigma * 0.45, spec["strength"] * 0.18, (0.7, 0.85, 1.0))
    return out.astype(np.float32)


def infrared(lin: np.ndarray, plate_mask: np.ndarray) -> np.ndarray:
    """Near-IR with an illuminator: the retroreflective sheet saturates, the ink stays
    black, colour vanishes, and the unlit scene falls off."""
    lum = lin @ np.array([0.3, 0.6, 0.1], np.float32)
    plate = np.clip(lum * 3.0, 0.0, 1.0)
    scene = lum * 0.35
    m = plate_mask.astype(np.float32)
    g = plate * m + scene * (1 - m)
    return np.repeat(g[..., None], 3, axis=2) * np.array([1.0, 1.0, 1.02], np.float32)
