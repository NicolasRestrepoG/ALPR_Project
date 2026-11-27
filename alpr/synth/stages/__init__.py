"""The seven degradation stages, in pipeline order. One module each.

    1 warp         perspective at full render resolution
    2 composite    onto a background, still at render resolution
    3 downsample   area filter to the sampled plate width: the one band-limit
    4 photometric  exposure, white balance, shadows, specular, bloom, IR, gamma
    5 blur         motion and defocus, on the radiance field
    6 noise        Poisson shot + Gaussian read, per-pixel independent, after blur
    7 jpeg         last, as in a real ISP

Images between stages are float32 RGB in [0, 1] from composite onward; warp takes
the uint8 render and jpeg returns uint8.
"""

ORDER = ("warp", "composite", "downsample", "photometric", "blur", "noise", "jpeg")
