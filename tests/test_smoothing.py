"""Tests for utils.smoothing, plus app.state's CursorMapper/ScrollMapper
which are built directly on top of it (no camera or gesture engine
involved — pure math on synthetic points)."""

from __future__ import annotations

import pytest

from app.state import CursorMapper, ScrollMapper
from utils.smoothing import ExponentialSmoother, Point2DSmoother


def test_exponential_smoother_first_value_passthrough():
    smoother = ExponentialSmoother(factor=0.5)
    assert smoother.update(10.0) == 10.0


def test_exponential_smoother_converges_toward_constant_input():
    smoother = ExponentialSmoother(factor=0.5)
    value = 0.0
    for _ in range(50):
        value = smoother.update(10.0)
    assert value == pytest.approx(10.0, abs=0.01)


def test_exponential_smoother_dampens_a_single_spike():
    smoother = ExponentialSmoother(factor=0.9)
    for _ in range(5):
        smoother.update(0.0)
    spiked = smoother.update(100.0)
    assert spiked < 100.0
    assert spiked > 0.0


def test_exponential_smoother_rejects_invalid_factor():
    with pytest.raises(ValueError):
        ExponentialSmoother(factor=1.0)
    with pytest.raises(ValueError):
        ExponentialSmoother(factor=-0.1)


def test_exponential_smoother_reset():
    smoother = ExponentialSmoother(factor=0.5)
    smoother.update(10.0)
    smoother.reset()
    assert smoother.update(3.0) == 3.0


def test_point2d_smoother_first_value_passthrough():
    smoother = Point2DSmoother(factor=0.5)
    assert smoother.update((1.0, 2.0)) == (1.0, 2.0)


def test_point2d_smoother_converges():
    smoother = Point2DSmoother(factor=0.5)
    point = (0.0, 0.0)
    for _ in range(50):
        point = smoother.update((5.0, -5.0))
    assert point[0] == pytest.approx(5.0, abs=0.01)
    assert point[1] == pytest.approx(-5.0, abs=0.01)


def test_cursor_mapper_maps_center_to_screen_center():
    mapper = CursorMapper(
        screen_width=1000, screen_height=500, sensitivity=1.0, deadzone=0.0, smoothing=0.0
    )
    x, y = mapper.update((0.5, 0.5))
    assert x == pytest.approx(500.0)
    assert y == pytest.approx(250.0)


def test_cursor_mapper_sensitivity_expands_range():
    mapper = CursorMapper(
        screen_width=1000, screen_height=1000, sensitivity=2.0, deadzone=0.0, smoothing=0.0
    )
    # A small movement off-center should be amplified and clamped at the edge.
    x, _ = mapper.update((0.75, 0.5))
    assert x == pytest.approx(1000.0)  # clamped: 0.5 + (0.25)*2 = 1.0 -> full width


def test_cursor_mapper_deadzone_suppresses_tiny_jitter():
    mapper = CursorMapper(
        screen_width=1000, screen_height=1000, sensitivity=1.0, deadzone=0.05, smoothing=0.0
    )
    first = mapper.update((0.5, 0.5))
    # Movement smaller than the deadzone should be ignored (output unchanged).
    second = mapper.update((0.51, 0.5))
    assert second == first


def test_cursor_mapper_reset_clears_history():
    mapper = CursorMapper(
        screen_width=1000, screen_height=1000, sensitivity=1.0, deadzone=0.05, smoothing=0.0
    )
    mapper.update((0.5, 0.5))
    mapper.reset()
    # After reset, a tiny movement should not be compared against stale history.
    x, y = mapper.update((0.5, 0.5))
    assert x == pytest.approx(500.0)
    assert y == pytest.approx(500.0)


def test_scroll_mapper_returns_zero_below_deadzone():
    mapper = ScrollMapper(sensitivity=10.0, deadzone=0.1, invert=False)
    assert mapper.update(0.01) == 0


def test_scroll_mapper_moving_up_scrolls_positive():
    mapper = ScrollMapper(sensitivity=10.0, deadzone=0.01, invert=False)
    # Normalized y decreasing (negative velocity) == moving up.
    amount = 0
    for _ in range(5):
        amount = mapper.update(-1.0)
    assert amount > 0


def test_scroll_mapper_moving_down_scrolls_negative():
    mapper = ScrollMapper(sensitivity=10.0, deadzone=0.01, invert=False)
    amount = 0
    for _ in range(5):
        amount = mapper.update(1.0)
    assert amount < 0


def test_scroll_mapper_invert_flips_direction():
    mapper = ScrollMapper(sensitivity=10.0, deadzone=0.01, invert=True)
    amount = 0
    for _ in range(5):
        amount = mapper.update(-1.0)
    assert amount < 0
