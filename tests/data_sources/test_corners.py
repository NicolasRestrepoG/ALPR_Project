"""Corner ordering — the single highest-value unit test in the project.

Getting the order wrong produces no error anywhere: the detector trains happily,
the rectifier emits upside-down plates, and the recognizer plateaus at chance.
So the order is verified by winding direction and by starting vertex, not by eye.
"""

from __future__ import annotations

import itertools
import math

import pytest

from alpr.data_sources.base import (
    CornerOrderError,
    Winding,
    assert_corner_order,
    corners_to_box,
    order_corners,
    signed_area,
    winding,
)

# An axis-aligned plate, 200 x 100 px at (100, 50). TL, TR, BR, BL.
RECT = ((100.0, 50.0), (300.0, 50.0), (300.0, 150.0), (100.0, 150.0))


def test_canonical_order_is_clockwise_on_screen():
    assert winding(RECT) is Winding.CLOCKWISE
    assert signed_area(RECT) > 0


def test_order_corners_is_idempotent():
    assert order_corners(RECT) == RECT
    assert order_corners(order_corners(RECT)) == RECT


@pytest.mark.parametrize("permutation", list(itertools.permutations(range(4))))
def test_every_input_permutation_yields_the_same_order(permutation):
    """All 24 orderings of the same four points must normalise identically.

    This is the property that makes `order_corners` safe to apply to any source,
    whatever convention it publishes.
    """
    shuffled = [RECT[i] for i in permutation]
    assert order_corners(shuffled) == RECT


def test_counterclockwise_input_is_reversed():
    ccw = (RECT[0], RECT[3], RECT[2], RECT[1])  # TL, BL, BR, TR
    assert winding(ccw) is Winding.COUNTERCLOCKWISE
    assert order_corners(ccw) == RECT


def test_perspective_quad_orders_correctly():
    """A plate seen at yaw: the far edge is shorter and vertically inset."""
    quad = ((120.0, 60.0), (300.0, 48.0), (300.0, 152.0), (120.0, 140.0))
    assert order_corners(quad) == quad
    tl, tr, br, bl = order_corners(quad)
    assert tl[0] < tr[0] and bl[0] < br[0]
    assert tl[1] < bl[1] and tr[1] < br[1]


@pytest.mark.parametrize("degrees", [-8, -5, 0, 5, 8])
def test_rolled_plate_orders_correctly(degrees):
    """Roll across the full trained envelope (±8°) must not confuse the ordering."""
    theta = math.radians(degrees)
    cx, cy = 200.0, 100.0

    def rot(p):
        x, y = p[0] - cx, p[1] - cy
        return (
            cx + x * math.cos(theta) - y * math.sin(theta),
            cy + x * math.sin(theta) + y * math.cos(theta),
        )

    rolled = tuple(rot(p) for p in RECT)
    ordered = order_corners(rolled)
    # Same cyclic identity: each ordered corner is the rotation of the RECT corner
    # at the same index.
    assert ordered == rolled
    assert_corner_order(ordered, strict=True)


def test_assert_corner_order_rejects_a_rotated_cycle():
    """The CCPD failure mode: right-bottom first, same winding, wrong start.

    Winding alone cannot catch this — it is a rotation of the cycle, not a
    reversal — which is exactly why `assert_corner_order` pins the starting vertex.
    """
    rb_first = (RECT[2], RECT[3], RECT[0], RECT[1])  # BR, BL, TL, TR
    assert winding(rb_first) is winding(RECT)  # identical winding!
    with pytest.raises(CornerOrderError, match="TL/TR/BR/BL"):
        assert_corner_order(rb_first)


def test_assert_corner_order_accepts_canonical():
    assert assert_corner_order(RECT) == RECT


def test_strict_rejects_a_quad_that_is_canonical_but_not_upright():
    """`strict` is the extra guard for a plate-shaped quad.

    This skewed quad is a perfectly valid TL/TR/BR/BL cycle — it passes the
    non-strict check — but no plate inside the ±25°/±15°/±8° envelope looks like
    this, so `strict=True` rejects it. That is why the strict check is opt-in:
    CCPD's Rotate and Tilt subsets are legitimately extreme.
    """
    skewed = ((61.8, 25.1), (90.8, 50.5), (28.2, 75.6), (47.7, 58.3))
    assert order_corners(skewed) == skewed
    assert assert_corner_order(skewed) == skewed  # non-strict: fine
    with pytest.raises(CornerOrderError, match="not upright"):
        assert_corner_order(skewed, strict=True)


@pytest.mark.parametrize(
    "bad, match",
    [
        ([(0, 0), (1, 0), (1, 1)], "exactly 4"),
        ([(0, 0), (1, 0), (1, 1), (0, 1), (2, 2)], "exactly 4"),
        ([(0, 0), (0, 0), (1, 1), (0, 1)], "distinct"),
        ([(0, 0), (1, 1), (2, 2), (3, 3)], "degenerate"),
        ([(0, 0), (1, 0), (2, 0), (3, 0)], "degenerate"),
        ([(0, 0), (1, 0), (1, 1), (0, float("nan"))], "finite"),
        ([(0, 0), (1, 0), (1, 1), (0, "x")], "numeric"),
        ([(0, 0), (1, 0), (1, 1), (0,)], "pair"),
    ],
)
def test_malformed_corners_raise(bad, match):
    with pytest.raises(CornerOrderError, match=match):
        order_corners(bad)


def test_corners_to_box_is_tight():
    box = corners_to_box(RECT)
    assert box.as_list() == [100.0, 50.0, 300.0, 150.0]
    assert box.width == 200.0
    assert box.height == 100.0


def test_no_box_to_corners_helper_exists():
    """Box-only sources must set corners=None rather than fabricating them.

    Fabricated corners teach the detector that every plate is axis-aligned
    (plan §7). The absence of this helper is a design decision, so it is asserted.
    """
    from alpr.data_sources import base

    assert not hasattr(base, "box_to_corners")
