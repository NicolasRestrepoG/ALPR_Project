"""The attention-drift diagnostic."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from alpr.recognizer.diagnostics import (
    AttentionLog,
    is_monotonic,
    mean_attention,
    peaks,
    spacing_error,
)


def _attention(cols, sharp=8.0):
    a = np.full((6, 24), 1e-3)
    for k, c in enumerate(cols):
        a[k, c] = sharp
    return a / a.sum(axis=1, keepdims=True)


def test_peaks_and_checks():
    good = _attention([2, 6, 10, 14, 18, 22])
    assert peaks(good) == [2, 6, 10, 14, 18, 22]
    assert is_monotonic(peaks(good)) and spacing_error(peaks(good)) == 0
    collapsed = _attention([2, 6, 6, 14, 18, 22])
    assert not is_monotonic(peaks(collapsed))
    assert spacing_error([3, 6, 10, 14, 18, 20]) == 2


def test_mean_attention_accepts_batches_and_tensors():
    batch = torch.tensor(np.stack([_attention([2, 6, 10, 14, 18, 22])] * 3))
    assert mean_attention(batch).shape == (6, 24)
    with pytest.raises(ValueError):
        mean_attention(np.zeros((5, 24)))


def test_log_summary_line_save_and_figure(tmp_path):
    log = AttentionLog()
    log.add(1, _attention([3, 6, 9, 14, 18, 21]))
    log.add(2, _attention([2, 2, 10, 14, 18, 22]))
    s = log.summary()
    assert s[0]["monotonic"] and not s[1]["monotonic"]
    assert "NOT MONOTONIC" in log.line()
    log.save(tmp_path / "att.json")
    data = json.loads((tmp_path / "att.json").read_text())
    assert data["summary"][1]["peaks"] == [2, 2, 10, 14, 18, 22]
    fig = log.figure()
    assert fig.dtype == np.uint8 and fig.shape[0] > 2 * 6 * 18 and fig.shape[2] == 3
    assert log.save_figure(tmp_path / "att.png").exists()
    with pytest.raises(ValueError):
        AttentionLog().figure()
