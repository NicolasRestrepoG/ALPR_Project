"""One synthetic sample: render → occluders → the seven stages, in order → crop.

    render (Phase 2)          clean embossed plate, lit, relief parallax
    dirt, frame, bolts        on the flat plate (they are coplanar with it)
    1 warp                    perspective at render resolution
    2 composite               onto a vehicle-front background
      grille bars             in front of the plate, before any band-limiting
    3 downsample              area filter to the sampled plate width
    4 photometric             exposure, white balance, shadow, specular, bloom, IR, gamma
    5 blur                    motion, defocus
    6 noise                   shot + read, per pixel
    7 jpeg
    rectify                   with jittered corners → the 192 x 96 crop the model reads

`Synthesizer.run(params)` is deterministic: same parameters, same pixels.

`order` exists for one purpose: the Sprint 3.2 ablation, which trains on corpora
built with each of three deliberately wrong orders to show the right one matters.

    correct                     as above
    composite_after_downsample  plate and background shrunk separately, then joined
                                with a hard alpha edge
    noise_before_blur           stages 5 and 6 swapped
    jpeg_before_noise           stages 6 and 7 swapped
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alpr.plates.geometry import CANONICAL_SIZE, MT001
from alpr.plates.rectify import rectify
from alpr.plates.render import Lighting, PlateRenderer, PlateSpec, PlateStyle
from alpr.synth import augment
from alpr.synth.backgrounds import BackgroundStore
from alpr.synth.params import DEFAULT_RANGES, Ranges, SampleParams, sample_params
from alpr.synth.stages import blur as _blur
from alpr.synth.stages import composite as _composite
from alpr.synth.stages import downsample as _downsample
from alpr.synth.stages import jpeg as _jpeg
from alpr.synth.stages import noise as _noise
from alpr.synth.stages import photometric as _photometric
from alpr.synth.stages import warp as _warp

EMBOSS_MM = 1.5
ORDERS = ("correct", "composite_after_downsample", "noise_before_blur", "jpeg_before_noise")

#: Render resolutions available (px/mm), and the oversampling the warp must keep
#: over the final plate width. The plan's rule is "warp at high resolution, then one
#: band-limiting downsample" — never warp a small image. Rendering every plate at
#: 4 px/mm (1320 px) to end at 70 px is 19x oversampling; 4x keeps the rule and the
#: staircase-free edges, and makes small plates 4-16x cheaper.
RENDER_SCALES = (1.0, 2.0, 4.0)
OVERSAMPLE = 4.0


def render_scale(plate_px: float) -> float:
    need = OVERSAMPLE * plate_px / MT001.plate_w
    return next((s for s in RENDER_SCALES if s >= need), RENDER_SCALES[-1])


@dataclass
class Sample:
    image: np.ndarray  # the degraded "photograph", uint8 RGB
    corners: np.ndarray  # true plate corners in it, TL/TR/BR/BL
    crop: np.ndarray  # rectified 192 x 96 uint8, from the jittered corners
    params: SampleParams


class Synthesizer:
    def __init__(
        self,
        renderer: PlateRenderer,
        backgrounds: BackgroundStore | None = None,
        municipalities: list[str] | None = None,
        ranges: Ranges = DEFAULT_RANGES,
    ) -> None:
        self.renderer = renderer
        self.backgrounds = backgrounds
        self.municipalities = municipalities or ["BOGOTA D.C."]
        self.ranges = ranges

    def params(
        self,
        rng: np.random.Generator,
        severity: float,
        text: str | None = None,
        style: str = "particular",
    ) -> SampleParams:
        return sample_params(
            rng,
            severity,
            municipalities=self.municipalities,
            n_backgrounds=len(self.backgrounds) if self.backgrounds else 0,
            ranges=self.ranges,
            text=text,
            style=style,
        )

    def run(self, p: SampleParams, order: str = "correct") -> Sample:
        if order not in ORDERS:
            raise ValueError(f"unknown order {order!r}; one of {ORDERS}")
        rng = np.random.default_rng(p.seed)
        spec = PlateSpec(
            p.text,
            p.municipality,
            style=PlateStyle(background=p.plate_rgb, ink=p.ink_rgb, name=p.style),
            lighting=Lighting(direction=p.light_dir),
            view=(0.0, p.relief_smear_mm / EMBOSS_MM),
            emboss_mm=EMBOSS_MM,
        )
        px = render_scale(p.plate_px)
        plate = self.renderer.at(px).render(spec)
        if p.wear:
            plate = augment.wear(plate, p.wear, rng, p.plate_rgb)
        if p.dirt:
            plate = augment.dirt(plate, p.dirt, rng)
        if p.frame:
            plate = augment.frame(plate, p.frame, rng)
        if p.bolts:
            plate = augment.bolts(plate, p.bolts)

        # 1 — warp at render resolution
        warped, alpha, corners = _warp.warp(
            plate, p.yaw_deg, p.pitch_deg, p.roll_deg, p.distance_mm, px
        )
        size = (warped.shape[1], warped.shape[0])
        # 2 — composite, still at render resolution
        if self.backgrounds is not None and p.background >= 0:
            bg_img, bg_quad = self.backgrounds.get(p.background)
            bg = _composite.place_background(bg_img, bg_quad, corners, size, p.bg_offset)
        else:
            bg = _composite.plain_background(size, rng)
        if order == "composite_after_downsample":
            # Wrong on purpose: each layer shrunk alone, joined with a hard edge.
            small_plate, small_corners = _downsample.downsample(warped, corners, p.plate_px)
            small_bg, _ = _downsample.downsample(bg, corners, p.plate_px)
            hard = _downsample.plate_mask(small_plate.shape[:2], small_corners) > 0.5
            img = _composite.composite(small_plate, hard.astype(np.float32), small_bg)
            corners = small_corners
        else:
            img = _composite.composite(warped, alpha, bg)
            if p.grille:
                img = augment.grille(img, bg, corners, p.grille)
            # 3 — downsample: the one band-limit
            img, corners = _downsample.downsample(img, corners, p.plate_px)
        img = img.astype(np.float32) / 255.0
        mask = _downsample.plate_mask(img.shape[:2], corners)
        # 4 — photometric
        img = _photometric.photometric(
            img,
            corners,
            mask,
            exposure_log2=p.exposure_log2,
            temp_gains=p.temp_gains,
            gamma=p.gamma,
            shadow=p.shadow,
            specular=p.specular,
            bloom=p.bloom,
            ir=p.ir,
        )
        # 5 — blur, 6 — noise, 7 — jpeg
        if order == "noise_before_blur":
            img = _noise.noise(img, rng, p.shot_peak, p.read_sigma)
            img = _blur.blur(img, p.defocus_px, p.motion_px, p.motion_angle_deg)
            final = _jpeg.jpeg(img, p.jpeg_quality)
        elif order == "jpeg_before_noise":
            img = _blur.blur(img, p.defocus_px, p.motion_px, p.motion_angle_deg)
            coded = _jpeg.jpeg(img, p.jpeg_quality).astype(np.float32) / 255.0
            final = _jpeg.jpeg(_noise.noise(coded, rng, p.shot_peak, p.read_sigma), 100)
        else:
            img = _blur.blur(img, p.defocus_px, p.motion_px, p.motion_angle_deg)
            img = _noise.noise(img, rng, p.shot_peak, p.read_sigma)
            final = _jpeg.jpeg(img, p.jpeg_quality)

        width = _downsample.plate_width(corners)
        jittered = corners + np.asarray(p.corner_jitter) * width
        crop = rectify(final, jittered, CANONICAL_SIZE)
        return Sample(final, corners, crop, p)

    def sample(
        self,
        rng: np.random.Generator,
        severity: float,
        text: str | None = None,
        style: str = "particular",
        order: str = "correct",
    ) -> Sample:
        return self.run(self.params(rng, severity, text, style), order)
