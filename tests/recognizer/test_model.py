"""The recognizer architecture: parameter counts and shapes are contracts (plan §9)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from alpr.recognizer.model import (
    ATTENTION_HEAD_PARAMS,
    CTC_ALPHABET,
    CTC_HEAD_PARAMS,
    TRUNK_PARAMS,
    CTCHead,
    CTCRecognizer,
    PositionalAttentionHead,
    Reading,
    Recognizer,
    Trunk,
    count_params,
    ctc_greedy_decode,
    expected_column,
    preprocess,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Recognizer().eval()


def test_parameter_counts_are_exact():
    assert count_params(Trunk()) == TRUNK_PARAMS == 2_353_888
    assert count_params(PositionalAttentionHead()) == ATTENTION_HEAD_PARAMS == 183_533
    assert count_params(CTCHead()) == CTC_HEAD_PARAMS == 305_957
    assert count_params(Recognizer()) == 2_537_421


def test_trunk_layers_follow_the_spec():
    convs = [m for m in Trunk().modules() if isinstance(m, nn.Conv2d)]
    assert [(c.in_channels, c.out_channels) for c in convs] == [
        (3, 32), (32, 32), (32, 64), (64, 64), (64, 128),
        (128, 128), (128, 256), (256, 256), (256, 256), (256, 256),
    ]  # fmt: skip
    for c in convs:
        assert c.kernel_size == (3, 3) and c.stride == (1, 1) and c.padding == (1, 1)
        assert c.bias is None  # BatchNorm follows
    pools = [m.kernel_size for m in Trunk().modules() if isinstance(m, nn.MaxPool2d)]
    assert pools == [2, 2, 2, (2, 1), (2, 1), (3, 1)]


def test_shapes(model):
    x = torch.randn(4, 3, 96, 192)
    seq = model.trunk(x)
    assert seq.shape == (4, 24, 256)
    out = model(x)
    assert [tuple(t.shape) for t in out.logits] == [(4, 26)] * 3 + [(4, 10)] * 3
    assert out.validity.shape == (4,)
    assert out.attention.shape == (4, 6, 24)
    assert torch.allclose(out.attention.sum(-1), torch.ones(4, 6), atol=1e-5)


def test_attention_starts_on_the_expected_columns(model):
    att = model(torch.randn(8, 3, 96, 192)).attention.mean(0)
    assert (
        att.argmax(-1).tolist() == [expected_column(k) for k in range(6)] == [2, 6, 10, 14, 18, 22]
    )


def test_every_parameter_gets_a_gradient():
    torch.manual_seed(0)
    m = Recognizer().train()
    out = m(torch.randn(2, 3, 96, 192))
    loss = sum(t.logsumexp(-1).mean() for t in out.logits) + out.validity.mean()
    loss.backward()
    missing = [
        n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert missing == []


def test_read_returns_a_formatted_reading(model):
    crop = np.random.default_rng(0).integers(0, 255, (96, 192, 3), dtype=np.uint8)
    r = model.read(crop)
    assert isinstance(r, Reading)
    assert len(r.plate) == 6 and r.plate[:3].isalpha() and r.plate[3:].isdigit()
    assert len(r.per_position_confidence) == 6
    assert all(0 < c <= 1 for c in r.per_position_confidence) and 0 < r.validity < 1
    assert len(model.read_batch(np.stack([crop, crop]))) == 2
    assert not model.training


def test_preprocess_checks_the_crop_shape():
    assert preprocess(np.zeros((96, 192, 3), np.uint8)).shape == (1, 3, 96, 192)
    with pytest.raises(ValueError, match="96, 192, 3"):
        preprocess(np.zeros((64, 128, 3), np.uint8))


def test_ctc_head_and_decoding():
    m = CTCRecognizer().eval()
    lp = m(torch.randn(3, 3, 96, 192))
    assert lp.shape == (24, 3, 37)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(24, 3), atol=1e-4)
    targets = torch.tensor([[CTC_ALPHABET.index(c) + 1 for c in "ABC123"]] * 3)
    loss = nn.CTCLoss(blank=0)(lp, targets, torch.full((3,), 24), torch.full((3,), 6))
    assert torch.isfinite(loss)
    # A hand-made best path: A A blank B C C 1 2 blank 2 3 → "ABC1223"
    path = [11, 11, 0, 12, 13, 13, 2, 3, 0, 3, 4]
    fake = torch.full((len(path), 1, 37), -10.0)
    for t, k in enumerate(path):
        fake[t, 0, k] = 0.0
    assert ctc_greedy_decode(fake) == ["ABC1223"]


def test_a_few_steps_reduce_the_loss():
    """Sanity, not training: the model can fit four fixed samples."""
    torch.manual_seed(0)
    m = Recognizer().train()
    x = torch.randn(4, 3, 96, 192)
    targets = torch.randint(0, 26, (4, 3)), torch.randint(0, 10, (4, 3))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

    def loss_fn():
        out = m(x)
        ce = nn.functional.cross_entropy
        return sum(
            ce(out.letters[k], targets[0][:, k]) + ce(out.digits[k], targets[1][:, k])
            for k in range(3)
        )

    first = loss_fn().item()
    for _ in range(12):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert loss_fn().item() < 0.5 * first


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_runs_on_mps_with_the_same_shapes(model):
    m = Recognizer().to("mps").eval()
    out = m(torch.randn(2, 3, 96, 192, device="mps"))
    assert [tuple(t.shape) for t in out.logits] == [(2, 26)] * 3 + [(2, 10)] * 3
