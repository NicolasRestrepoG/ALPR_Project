"""The plate domain model — everything that knows what a Colombian plate *is*.

Phase 2 contract (plan §7):

    rectify(image, corners) -> 192 x 96 RGB array
    render(PlateSpec)       -> image array               [Sprint 2.2]
    data/assets/glyphs/     -> 36 character images        [Sprint 2.2]

`geometry` is the single source of truth for millimetre layout; every other module
in this package reads its numbers from there.
"""
