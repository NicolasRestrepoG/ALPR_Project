"""Every random decision for one synthetic sample, drawn up front.

`sample_params(rng, severity)` returns a frozen `SampleParams`; the pipeline then
applies it deterministically. That split is what makes a sample reproducible from
its parameters alone, lets each stage be tested with hand-written parameters, and
gives Sprint 3.2's corpus hash something exact to hash.

**Severity.** One scalar in [0, 1] scales every magnitude: a magnitude is drawn as
`lerp(easy, hard, severity * u)` with `u ~ U(0, 1)`. Severity 0 is a pristine,
frontal, well-lit, large plate; severity 1 lets every magnitude reach its harsh end
(it does not force it there). Augmentation *probabilities* are the plan's and do not
change with severity; their *strength* does.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from alpr.plates.labels import CHARSET

LETTERS = CHARSET[:26]
DIGITS = CHARSET[26:]


@dataclass(frozen=True, slots=True)
class Ranges:
    """The envelope, easy end first. Sprint 3.2 versions and hashes this."""

    plate_px: tuple[float, float] = (260.0, 70.0)  # final plate width, log-scale
    #: True: width shrinks with severity (demo sheets). False: log-uniform over the
    #: whole range regardless of severity — the corpus sampling policy (plan, 3.2).
    plate_px_follows_severity: bool = True
    yaw_deg: float = 25.0  # truncated-normal limits (plan: front-plate envelope)
    pitch_deg: float = 15.0
    roll_deg: float = 8.0
    distance_mm: tuple[float, float] = (8000.0, 3000.0)  # nearer = more perspective
    relief_smear_mm: tuple[float, float] = (0.0, 5.0)  # emboss parallax (photos: ~4.5)
    light_elev_deg: tuple[float, float] = (60.0, 20.0)
    exposure: tuple[float, float] = (0.0, 1.2)  # |log2 gain| reach
    color_temp: tuple[float, float] = (0.0, 0.25)  # blue/red gain spread
    gamma: tuple[float, float] = (0.0, 0.25)  # |log gamma| reach
    shadow: tuple[float, float] = (0.0, 0.65)
    specular: tuple[float, float] = (0.0, 0.45)
    defocus_px: tuple[float, float] = (0.0, 1.8)  # at final resolution
    motion_px: tuple[float, float] = (0.0, 7.0)
    shot_peak: tuple[float, float] = (4000.0, 120.0)  # photons at white, log-scale
    read_sigma: tuple[float, float] = (0.0, 0.03)
    jpeg_quality: tuple[float, float] = (95.0, 30.0)
    corner_jitter: tuple[float, float] = (0.0, 0.02)  # fraction of plate width
    night_p: float = 0.2
    # Front-view augmentation probabilities (plan, Sprint 3.1).
    bloom_p_day: float = 0.35
    bloom_p_night: float = 0.75
    frame_p: float = 0.30
    grille_p: float = 0.12
    bolts_p: float = 0.55
    dirt_p: float = 0.20
    ir_p: float = 0.08
    wear_p: float = 0.35
    # Sheeting and ink colour (albedo), from 169 labelled plates outside the test set:
    # hue (OpenCV 0-180) 15-25 around 20 — amber, not lemon; ink a warm dark grey.
    # Lighting darkens further in the photometric stage, so value stays bright here.
    plate_hue: tuple[float, float] = (15.0, 25.0)
    plate_sat: tuple[float, float] = (175.0, 250.0)
    plate_val: tuple[float, float] = (195.0, 250.0)
    ink_level: tuple[float, float] = (14.0, 62.0)
    publico_val: tuple[float, float] = (215.0, 248.0)  # white sheet, slightly warm


DEFAULT_RANGES = Ranges()


@dataclass(frozen=True, slots=True)
class SampleParams:
    severity: float
    seed: int  # for the stages that draw per-pixel randomness (noise, dirt)
    text: str
    municipality: str
    style: str  # "particular" (Phase 3.2 adds negatives)
    plate_rgb: tuple[int, int, int]  # sheeting albedo
    ink_rgb: tuple[int, int, int]
    night: bool
    # renderer
    light_dir: tuple[float, float, float]
    relief_smear_mm: float
    # geometry
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    distance_mm: float
    plate_px: float
    background: int  # index into the background store, -1 for a plain one
    bg_offset: tuple[float, float]  # plate centre shift against the background's plate
    corner_jitter: tuple[tuple[float, float], ...]  # 4 corners, fraction of plate width
    # photometric
    exposure_log2: float
    temp_gains: tuple[float, float, float]
    gamma: float
    shadow: dict[str, Any] | None
    specular: dict[str, Any] | None
    bloom: dict[str, Any] | None
    ir: bool
    # optics, sensor, codec
    defocus_px: float
    motion_px: float
    motion_angle_deg: float
    shot_peak: float
    read_sigma: float
    jpeg_quality: int
    # occluders
    bolts: tuple[dict[str, Any], ...] = ()
    frame: dict[str, Any] | None = None
    dirt: dict[str, Any] | None = None
    grille: dict[str, Any] | None = None
    wear: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _mag(rng: np.random.Generator, rng_pair: tuple[float, float], s: float) -> float:
    return _lerp(rng_pair[0], rng_pair[1], s * rng.random())


def _log_mag(rng: np.random.Generator, rng_pair: tuple[float, float], s: float) -> float:
    lo, hi = math.log(rng_pair[0]), math.log(rng_pair[1])
    return math.exp(_lerp(lo, hi, s * rng.random()))


def _angle(rng: np.random.Generator, limit: float, s: float) -> float:
    """Truncated normal (sd = limit / 2, clipped at the limit), scaled by severity."""
    return float(np.clip(rng.normal(0.0, limit / 2), -limit, limit)) * s


def random_text(rng: np.random.Generator, moto: bool = False) -> str:
    """Uniform letters and digits: no issuance-series prior for the model to learn.
    Motorbike plates end in a letter: LLLDDL."""
    letters = "".join(rng.choice(list(LETTERS), 3))
    if moto:
        return letters + "".join(rng.choice(list(DIGITS), 2)) + str(rng.choice(list(LETTERS)))
    return letters + "".join(rng.choice(list(DIGITS), 3))


def sample_params(
    rng: np.random.Generator,
    severity: float,
    *,
    municipalities: list[str],
    n_backgrounds: int = 0,
    ranges: Ranges = DEFAULT_RANGES,
    text: str | None = None,
    style: str = "particular",
) -> SampleParams:
    """`style`: "particular" (a readable, valid plate), "publico" (white sheet, same
    layout) or "moto" (LLLDDL with COLOMBIA in the strip — the single-line motorbike
    plates seen in the labelled photos)."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    s = severity
    r = ranges
    night = bool(rng.random() < r.night_p * s)

    # Light: from above-front by default, swinging with severity.
    elev = math.radians(_mag(rng, r.light_elev_deg, s))
    azim = math.radians(rng.uniform(-180, 180) * s - 90.0)  # -90: from above
    light = (
        math.cos(elev) * math.cos(azim),
        math.cos(elev) * math.sin(azim),
        math.sin(elev),
    )

    def maybe(p: float) -> bool:
        return bool(rng.random() < p) and s > 0

    shadow = None
    if maybe(0.4):
        shadow = {
            "strength": _mag(rng, r.shadow, s),
            "angle_deg": float(rng.uniform(0, 180)),
            "offset": float(rng.uniform(-0.5, 0.5)),  # fraction of plate width from centre
            "softness": float(rng.uniform(0.02, 0.2)),
        }
    specular = None
    if maybe(0.3):
        specular = {
            "strength": _mag(rng, r.specular, s),
            "angle_deg": float(rng.uniform(20, 160)),
            "offset": float(rng.uniform(-0.6, 0.6)),
            "width": float(rng.uniform(0.05, 0.3)),
        }
    bloom = None
    if maybe(r.bloom_p_night if night else r.bloom_p_day):
        bloom = {
            "strength": _mag(rng, (0.0, 1.4), s),
            "x": float(rng.uniform(-0.8, 0.8)),  # fraction of plate width from centre
            "y": float(rng.uniform(-1.6, -0.7)),  # above the plate, plate heights
            "radius": float(rng.uniform(0.3, 0.9)),  # plate heights
            "ghosts": int(rng.integers(1, 4)),
        }

    bolts: list[dict[str, Any]] = []
    if maybe(r.bolts_p):
        positions = [0, 1] if rng.random() < 0.6 else [0, 1, 2, 3]
        tone = str(rng.choice(["chrome", "black", "rusty"]))
        for k in positions:
            bolts.append(
                {
                    "slot": k,
                    "dx_mm": float(rng.normal(0, 3.0 * s)),
                    "dy_mm": float(rng.normal(0, 3.0 * s)),
                    "radius_mm": float(rng.uniform(5.0, 5.0 + 4.0 * s)),
                    "tone": tone,
                }
            )
    frame = None
    if maybe(r.frame_p):
        frame = {
            "thickness_mm": float(rng.uniform(4.0, 4.0 + 10.0 * s)),
            "bottom_extra_mm": float(rng.uniform(0, 12.0 * s)) if rng.random() < 0.5 else 0.0,
            "material": str(rng.choice(["chrome", "black", "white"])),
            "text": bool(rng.random() < 0.5),
            "radius_mm": float(rng.uniform(3.0, 12.0)),
        }
    dirt = None
    if maybe(r.dirt_p):
        if rng.random() < 0.75:  # grime: soft, partial, broad
            dirt = {
                "kind": "grime",
                "coverage": _mag(rng, (0.0, 0.5), s),
                "scale_mm": float(rng.uniform(25, 90)),
                "opacity": float(rng.uniform(0.15, 0.55)),
                "tone": float(rng.uniform(0.35, 0.65)),
            }
        else:  # spatter: fine specks
            dirt = {
                "kind": "spatter",
                "coverage": _mag(rng, (0.0, 0.08), s),
                "scale_mm": float(rng.uniform(1.5, 4.0)),
                "opacity": float(rng.uniform(0.5, 0.85)),
                "tone": float(rng.uniform(0.3, 0.5)),
            }
    wear = None
    if maybe(r.wear_p):
        wear = {"amount": _mag(rng, (0.0, 0.7), s), "scale_mm": float(rng.uniform(6, 30))}
    grille = None
    if maybe(r.grille_p):
        grille = {
            "bars": int(rng.integers(1, 3)),
            "thickness": float(rng.uniform(0.04, 0.04 + 0.12 * s)),  # plate heights
            "positions": [float(rng.uniform(-0.1, 1.0)) for _ in range(2)],
        }

    jitter_mag = _mag(rng, r.corner_jitter, s)
    return SampleParams(
        severity=float(s),
        seed=int(rng.integers(0, 2**31 - 1)),
        text=text or random_text(rng, moto=style == "moto"),
        municipality="COLOMBIA"
        if style == "moto"
        else (str(rng.choice(municipalities)) if municipalities else "BOGOTA D.C."),
        style=style,
        plate_rgb=_publico_rgb(rng, r) if style == "publico" else _plate_rgb(rng, r),
        ink_rgb=_ink_rgb(rng, r),
        night=night,
        light_dir=light,
        relief_smear_mm=_mag(rng, r.relief_smear_mm, s),
        yaw_deg=_angle(rng, r.yaw_deg, s),
        pitch_deg=_angle(rng, r.pitch_deg, s),
        roll_deg=_angle(rng, r.roll_deg, s),
        distance_mm=_mag(rng, r.distance_mm, s),
        plate_px=_log_mag(rng, r.plate_px, s)
        if r.plate_px_follows_severity
        else _log_mag(rng, r.plate_px, 1.0),
        background=int(rng.integers(0, n_backgrounds)) if n_backgrounds else -1,
        bg_offset=(float(rng.normal(0, 0.05)), float(rng.normal(0, 0.05))),
        corner_jitter=tuple(
            (float(rng.normal(0, jitter_mag)), float(rng.normal(0, jitter_mag))) for _ in range(4)
        ),
        exposure_log2=float(rng.uniform(-1, 1)) * _mag(rng, r.exposure, s)
        - (1.0 if night else 0.0),
        temp_gains=_temp(rng, _mag(rng, r.color_temp, s)),
        gamma=math.exp(float(rng.uniform(-1, 1)) * _mag(rng, r.gamma, s)),
        shadow=shadow,
        specular=specular,
        bloom=bloom,
        ir=maybe(r.ir_p),
        defocus_px=_mag(rng, r.defocus_px, s),
        motion_px=_mag(rng, r.motion_px, s),
        motion_angle_deg=float(rng.uniform(0, 180)),
        shot_peak=_log_mag(rng, r.shot_peak, s) / (4.0 if night else 1.0),
        read_sigma=_mag(rng, r.read_sigma, s) * (2.0 if night else 1.0),
        jpeg_quality=round(_mag(rng, r.jpeg_quality, s)),
        bolts=tuple(bolts),
        frame=frame,
        dirt=dirt,
        wear=wear,
        grille=grille,
    )


def _plate_rgb(rng: np.random.Generator, r: Ranges) -> tuple[int, int, int]:
    """Sheeting colour: an appearance, not a degradation, so it varies at every severity."""
    import cv2

    hsv = np.array(
        [[[rng.uniform(*r.plate_hue), rng.uniform(*r.plate_sat), rng.uniform(*r.plate_val)]]],
        np.uint8,
    )
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _publico_rgb(rng: np.random.Generator, r: Ranges) -> tuple[int, int, int]:
    v = float(rng.uniform(*r.publico_val))
    return (round(v), round(v * 0.99), round(v * 0.95))


def _ink_rgb(rng: np.random.Generator, r: Ranges) -> tuple[int, int, int]:
    g = float(rng.uniform(*r.ink_level))
    return (min(255, round(g * 1.25)), round(g), round(g * 0.75))  # warm, as measured


def _temp(rng: np.random.Generator, spread: float) -> tuple[float, float, float]:
    """White-balance error: warm (red up, blue down) or cool, green fixed."""
    t = float(rng.uniform(-1, 1)) * spread
    return (1.0 + t, 1.0, 1.0 - t)
