"""The recognizer: a convolutional trunk and two interchangeable heads (plan §9).

    crop 3 x 96 x 192  →  Trunk  →  24 x 256 column sequence
                                    ├─ PositionalAttentionHead (v0): 6 queries →
                                    │    3 letter logits (26), 3 digit logits (10),
                                    │    validity logit, attention (6 x 24)
                                    └─ CTCHead (v1, Sprint 5.3 ablation): BiGRU → 37-way

Parameter counts are contracts, tested exactly: trunk 2,353,888; attention head
183,533; CTC head 305,957. If a count is off, a layer is wrong.

**Why the trunk keeps 24 columns.** Six glyphs span 192 input pixels; 24 output
columns give each character ~4 columns whose receptive fields cover the glyph
without the neighbours dominating. Height carries little (a glyph's identity is in
the horizontal placement of its strokes), so it is pooled to 1 with (2,1) and (3,1)
pools.

**Why letters and digits have separate classifiers.** The confusions that dominate
plate OCR — 0/O, 1/I, 8/B, 5/S, 2/Z, 6/G — are each one letter and one digit. With
positions 1-3 able to output only letters and 4-6 only digits, those confusions are
not discouraged; they are impossible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
CTC_ALPHABET = DIGITS + LETTERS  # index 0 is the CTC blank; symbols start at 1

INPUT_SIZE = (96, 192)  # H, W
COLUMNS = 24
WIDTH = 256
N_SLOTS = 6

TRUNK_PARAMS = 2_353_888
ATTENTION_HEAD_PARAMS = 183_533
CTC_HEAD_PARAMS = 305_957


def _conv_bn(c_in: int, c_out: int) -> list[nn.Module]:
    return [
        nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]


class Trunk(nn.Module):
    """3 x 96 x 192 → 24 x 256. Five stages of two 3x3 conv+BN+ReLU each; square
    pooling until width reaches 24, then height-only pooling to 1."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            *_conv_bn(3, 32), *_conv_bn(32, 32), nn.MaxPool2d(2),        # 32 x 48 x 96
            *_conv_bn(32, 64), *_conv_bn(64, 64), nn.MaxPool2d(2),       # 64 x 24 x 48
            *_conv_bn(64, 128), *_conv_bn(128, 128), nn.MaxPool2d(2),    # 128 x 12 x 24
            *_conv_bn(128, 256), *_conv_bn(256, 256), nn.MaxPool2d((2, 1)),  # 256 x 6 x 24
            *_conv_bn(256, 256), *_conv_bn(256, 256), nn.MaxPool2d((2, 1)),  # 256 x 3 x 24
            nn.MaxPool2d((3, 1)),                                        # 256 x 1 x 24
        )  # fmt: skip

    def forward(self, x: Tensor) -> Tensor:
        """(B, 3, 96, 192) → (B, 24, 256)."""
        return self.body(x).squeeze(2).transpose(1, 2)


@dataclass
class HeadOutput:
    letters: list[Tensor]  # 3 x (B, 26)
    digits: list[Tensor]  # 3 x (B, 10)
    validity: Tensor  # (B,) logit
    attention: Tensor  # (B, 6, 24), rows sum to 1

    @property
    def logits(self) -> list[Tensor]:
        """The six per-position logit tensors, in plate order."""
        return [*self.letters, *self.digits]


class PositionalAttentionHead(nn.Module):
    """Six learned queries attend over the 24 columns (single head, scaled
    dot-product). Keys carry the column position; values carry glyph features only.
    Positions 1-3 are classified over 26 letters, 4-6 over 10 digits. A separate
    validity head (mean over columns → 256 → 64 → 1) says whether this is a readable
    particular plate at all."""

    def __init__(self) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(COLUMNS, WIDTH))
        self.queries = nn.Parameter(torch.zeros(N_SLOTS, WIDTH))
        self.key = nn.Linear(WIDTH, WIDTH)
        self.value = nn.Linear(WIDTH, WIDTH)
        self.letter_heads = nn.ModuleList(nn.Linear(WIDTH, len(LETTERS)) for _ in range(3))
        self.digit_heads = nn.ModuleList(nn.Linear(WIDTH, len(DIGITS)) for _ in range(3))
        self.validity = nn.Sequential(nn.Linear(WIDTH, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self._init_positional()

    def _init_positional(self) -> None:
        """Sinusoidal column embeddings, and each query initialised to the embedding
        of its expected column (2, 6, 10, ...): attention starts aligned with the
        slots instead of having to discover them."""
        cols = torch.arange(COLUMNS, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, WIDTH, 2, dtype=torch.float32) * (-math.log(1e4) / WIDTH))
        pe = torch.zeros(COLUMNS, WIDTH)
        pe[:, 0::2] = torch.sin(cols * freq)
        pe[:, 1::2] = torch.cos(cols * freq)
        with torch.no_grad():
            self.pos.copy_(pe)
            for k in range(N_SLOTS):
                self.queries[k].copy_(pe[expected_column(k)] * 4.0)
            # Identity keys at start, so the positional term reaches the scores intact.
            self.key.weight.copy_(torch.eye(WIDTH))
            self.key.bias.zero_()

    def forward(self, seq: Tensor) -> HeadOutput:
        """seq (B, 24, 256) → HeadOutput."""
        keys = self.key(seq + self.pos)  # (B, 24, 256)
        values = self.value(seq)  # (B, 24, 256)
        scores = torch.einsum("qd,bcd->bqc", self.queries, keys) / math.sqrt(WIDTH)
        attention = scores.softmax(dim=-1)  # (B, 6, 24)
        slots = torch.einsum("bqc,bcd->bqd", attention, values)  # (B, 6, 256)
        letters = [head(slots[:, k]) for k, head in enumerate(self.letter_heads)]
        digits = [head(slots[:, 3 + k]) for k, head in enumerate(self.digit_heads)]
        validity = self.validity(seq.mean(dim=1)).squeeze(-1)
        return HeadOutput(letters, digits, validity, attention)


class CTCHead(nn.Module):
    """v1, for the Sprint 5.3 ablation: BiGRU (256 → 128 per direction) and a
    37-way classifier (blank + 0-9 + A-Z) per column, trained with CTC."""

    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(WIDTH, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, len(CTC_ALPHABET) + 1)

    def forward(self, seq: Tensor) -> Tensor:
        """seq (B, 24, 256) → log-probabilities (24, B, 37), the layout CTCLoss wants."""
        out, _ = self.rnn(seq)
        return self.classifier(out).log_softmax(-1).transpose(0, 1)


def expected_column(slot: int) -> int:
    """Where slot `slot` (0-5) should attend on a healthy model: 2, 6, 10, 14, 18, 22."""
    return 2 + 4 * slot


#: Input normalisation: uint8 RGB → (x / 255 - MEAN) / STD, every channel.
MEAN, STD = 0.5, 0.25


@dataclass(frozen=True)
class Reading:
    """The Phase 4 contract: what the recognizer says about one rectified crop."""

    plate: str
    per_position_confidence: tuple[float, ...]  # softmax max, one per slot
    validity: float  # probability this is a readable particular plate


def preprocess(crops: object) -> Tensor:
    """A (96, 192, 3) uint8 RGB crop, or a sequence/array of them → (B, 3, 96, 192)."""
    import numpy as np

    arr = np.asarray(crops)
    if arr.ndim == 3:
        arr = arr[None]
    if arr.shape[1:] != (*INPUT_SIZE, 3):
        raise ValueError(f"expected (*, 96, 192, 3) RGB crops, got {arr.shape}")
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float()
    return (x / 255.0 - MEAN) / STD


class Recognizer(nn.Module):
    """Model v0: trunk + positional attention head (2,537,421 parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = Trunk()
        self.head = PositionalAttentionHead()

    def forward(self, x: Tensor) -> HeadOutput:
        return self.head(self.trunk(x))

    @torch.no_grad()
    def read_batch(self, crops: object) -> list[Reading]:
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        out = self(preprocess(crops).to(device))
        if was_training:
            self.train()
        probs = [t.softmax(-1) for t in out.logits]
        conf = torch.stack([p.max(-1).values for p in probs], dim=1).cpu()
        idx = [p.argmax(-1).cpu() for p in probs]
        valid = out.validity.sigmoid().cpu()
        readings = []
        for b in range(conf.shape[0]):
            text = "".join(LETTERS[int(idx[k][b])] for k in range(3))
            text += "".join(DIGITS[int(idx[3 + k][b])] for k in range(3))
            readings.append(
                Reading(text, tuple(round(float(c), 4) for c in conf[b]), float(valid[b]))
            )
        return readings

    def read(self, crop: object) -> Reading:
        """One rectified 96 x 192 RGB crop → Reading."""
        return self.read_batch(crop)[0]


class CTCRecognizer(nn.Module):
    """Model v1: the same trunk + the CTC head."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = Trunk()
        self.head = CTCHead()

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.trunk(x))


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def ctc_greedy_decode(log_probs: Tensor) -> list[str]:
    """(T, B, 37) → strings: best path, repeats collapsed, blanks removed."""
    best = log_probs.argmax(-1).transpose(0, 1)  # (B, T)
    out = []
    for row in best.tolist():
        chars, prev = [], 0
        for idx in row:
            if idx != prev and idx != 0:
                chars.append(CTC_ALPHABET[idx - 1])
            prev = idx
        out.append("".join(chars))
    return out
