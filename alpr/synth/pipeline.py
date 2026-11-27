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
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from alpr.plates.geometry import CANONICAL_SIZE, RENDER_PX_PER_MM
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
        self, rng: np.random.Generator, severity: float, text: str | None = None
    ) -> SampleParams:
        return sample_params(
            rng,
            severity,
            municipalities=self.municipalities,
            n_backgrounds=len(self.backgrounds) if self.backgrounds else 0,
            ranges=self.ranges,
            text=text,
        )

    def run(self, p: SampleParams) -> Sample:
        rng = np.random.default_rng(p.seed)
        spec = PlateSpec(
            p.text,
            p.municipality,
            style=PlateStyle(background=p.plate_rgb, ink=p.ink_rgb, name=p.style),
            lighting=Lighting(direction=p.light_dir),
            view=(0.0, p.relief_smear_mm / EMBOSS_MM),
            emboss_mm=EMBOSS_MM,
        )
        plate = self.renderer.render(spec)
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
            plate, p.yaw_deg, p.pitch_deg, p.roll_deg, p.distance_mm, float(RENDER_PX_PER_MM)
        )
        size = (warped.shape[1], warped.shape[0])
        # 2 — composite, still at render resolution
        if self.backgrounds is not None and p.background >= 0:
            bg_img, bg_quad = self.backgrounds.get(p.background)
            bg = _composite.place_background(bg_img, bg_quad, corners, size, p.bg_offset)
        else:
            bg = _composite.plain_background(size, rng)
        img = _composite.composite(warped, alpha, bg)
        if p.grille:
            img = augment.grille(img, bg.astype(np.float32) / 255.0, corners, p.grille)
        # 3 — downsample: the one band-limit
        img, corners = _downsample.downsample(img, corners, p.plate_px)
        mask = cv2.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
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
        img = _blur.blur(img, p.defocus_px, p.motion_px, p.motion_angle_deg)
        img = _noise.noise(img, rng, p.shot_peak, p.read_sigma)
        final = _jpeg.jpeg(img, p.jpeg_quality)

        width = _downsample.plate_width(corners)
        jittered = corners + np.asarray(p.corner_jitter) * width
        crop = rectify(final, jittered, CANONICAL_SIZE)
        return Sample(final, corners, crop, p)

    def sample(self, rng: np.random.Generator, severity: float, text: str | None = None) -> Sample:
        return self.run(self.params(rng, severity, text))
