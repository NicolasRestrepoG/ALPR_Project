"""Phase 4 — the recognizer.

Contract (plan §7): `Recognizer.read(crop) → Reading(plate, per_position_confidence,
validity)`, where `crop` is a rectified 96 x 192 RGB crop from Phase 2's `rectify`.

    model        trunk, positional attention head, CTC head — parameter counts tested
    diagnostics  attention drift: per-epoch argmax columns, monotonicity, a figure

This is the only package that imports PyTorch.
"""
