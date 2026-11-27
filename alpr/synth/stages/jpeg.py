"""Stage 7 — JPEG, last, as in a real ISP.

Anything applied after it (blur, noise) would produce DCT-block statistics unlike
any real JPEG. Float32 [0, 1] in, uint8 out.
"""

from __future__ import annotations

import cv2
import numpy as np


def jpeg(image: np.ndarray, quality: int = 95) -> np.ndarray:
    u8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if quality >= 100:
        return u8
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
