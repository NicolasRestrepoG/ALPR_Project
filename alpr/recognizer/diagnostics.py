"""Attention-drift diagnostic (plan §9.2).

Each epoch, average the head's attention over a fixed batch and take each query's
argmax column. On a healthy model the six peaks increase left to right and sit near
columns 2, 6, 10, 14, 18, 22. Non-monotonic peaks mean a query has collapsed onto the
wrong glyph — visible here an epoch before the metric shows it. The Stage A → B gate
in training stops the run if the peaks are not ordered by the end of epoch 4.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from alpr.recognizer.model import COLUMNS, N_SLOTS, expected_column

CELL = 18  # figure pixels per attention cell


def mean_attention(attention: object) -> np.ndarray:
    """(B, 6, 24) or (6, 24), tensor or array → (6, 24) float array."""
    if hasattr(attention, "detach"):
        attention = attention.detach().float().cpu().numpy()
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=0)
    if a.shape != (N_SLOTS, COLUMNS):
        raise ValueError(f"expected (6, 24) attention, got {a.shape}")
    return a


def peaks(attention: object) -> list[int]:
    return [int(c) for c in mean_attention(attention).argmax(axis=1)]


def is_monotonic(cols: list[int]) -> bool:
    return all(b > a for a, b in zip(cols, cols[1:], strict=False))


def spacing_error(cols: list[int]) -> int:
    """Largest distance of a peak from its expected column."""
    return max(abs(c - expected_column(k)) for k, c in enumerate(cols))


@dataclass
class AttentionLog:
    epochs: list[int] = field(default_factory=list)
    maps: list[np.ndarray] = field(default_factory=list)

    def add(self, epoch: int, attention: object) -> list[int]:
        m = mean_attention(attention)
        self.epochs.append(int(epoch))
        self.maps.append(m)
        return peaks(m)

    def summary(self) -> list[dict[str, object]]:
        out = []
        for e, m in zip(self.epochs, self.maps, strict=True):
            cols = peaks(m)
            out.append(
                {
                    "epoch": e,
                    "peaks": cols,
                    "monotonic": is_monotonic(cols),
                    "spacing_error": spacing_error(cols),
                }
            )
        return out

    def line(self) -> str:
        """One log line for the latest epoch."""
        s = self.summary()[-1]
        ok = "ok" if s["monotonic"] else "NOT MONOTONIC"
        return f"attention peaks {s['peaks']} ({ok}, max offset {s['spacing_error']})"

    def save(self, path: Path | str) -> None:
        data = {"summary": self.summary(), "maps": [m.round(5).tolist() for m in self.maps]}
        Path(path).write_text(json.dumps(data) + "\n")

    def figure(self) -> np.ndarray:
        """One band per epoch: the 6 x 24 mean attention as a heatmap, the peak of
        each row ringed, the expected columns ticked, and a verdict per epoch."""
        if not self.maps:
            raise ValueError("no epochs logged")
        label_w, band_h = 150, N_SLOTS * CELL + 10
        width = label_w + COLUMNS * CELL + 10
        top = 26
        img = np.full((top + band_h * len(self.maps), width, 3), 255, np.uint8)
        for k in range(N_SLOTS):  # expected-column ticks
            x = label_w + expected_column(k) * CELL + CELL // 2
            cv2.line(img, (x, 4), (x, top - 6), (60, 60, 60), 2)
        cv2.putText(img, "expected", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
        for i, (e, m) in enumerate(zip(self.epochs, self.maps, strict=True)):
            y0 = top + i * band_h
            norm = m / max(float(m.max()), 1e-9)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            heat = cv2.resize(
                heat, (COLUMNS * CELL, N_SLOTS * CELL), interpolation=cv2.INTER_NEAREST
            )
            img[y0 : y0 + N_SLOTS * CELL, label_w : label_w + COLUMNS * CELL] = heat[..., ::-1]
            cols = peaks(m)
            for k, c in enumerate(cols):
                center = (label_w + c * CELL + CELL // 2, y0 + k * CELL + CELL // 2)
                cv2.circle(img, center, CELL // 2 - 2, (255, 60, 60), 2)
            ok = is_monotonic(cols)
            color = (20, 140, 20) if ok else (200, 20, 20)
            cv2.putText(
                img, f"epoch {e}", (8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1
            )
            verdict = "ordered" if ok else "NOT ordered"
            cv2.putText(img, verdict, (8, y0 + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img

    def save_figure(self, path: Path | str) -> Path:
        path = Path(path)
        cv2.imwrite(str(path), cv2.cvtColor(self.figure(), cv2.COLOR_RGB2BGR))
        return path
