"""Stage 6 — sensor noise: Poisson shot plus Gaussian read, independent per pixel.

After blur, so neighbouring pixels stay uncorrelated: correlated noise is a strong,
learnable synthetic signature. `shot_peak` is the photon count a white pixel
collects; fewer photons (dark scenes, small apertures) mean relatively more noise.
"""

from __future__ import annotations

import numpy as np


def noise(
    image: np.ndarray, rng: np.random.Generator, shot_peak: float = 1e9, read_sigma: float = 0.0
) -> np.ndarray:
    out = image.astype(np.float32)
    if shot_peak < 1e8:
        photons = rng.poisson(np.clip(out, 0, None) * shot_peak).astype(np.float32)
        out = photons / np.float32(shot_peak)
    if read_sigma > 0:
        out = out + rng.normal(0.0, read_sigma, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0)
