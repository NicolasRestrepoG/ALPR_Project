"""Phase 3 — the synthetic corpus.

Rendered plates (Phase 2) pass through a physically ordered degradation pipeline —
warp, composite, downsample, photometric, blur, sensor noise, JPEG — and come out
as rectified 192 x 96 crops, the same thing the recognizer sees at inference.

The order of the stages is the contract (plan, Sprint 3.1): each sits where a real
camera does that operation, and each reordering leaves an artifact the model can
learn instead of the glyphs.
"""
