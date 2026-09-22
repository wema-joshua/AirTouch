"""Signal smoothing helpers used by the cursor and scroll controllers.

Kept generic (operate on floats/tuples) so they can be tested without
any vision or OS dependency.
"""

from __future__ import annotations


class ExponentialSmoother:
    """One-Euro-style exponential moving average smoother for a scalar.

    ``factor`` in [0, 1): 0 means no smoothing (output = input),
    values closer to 1 mean heavier smoothing (slower to react).
    """

    def __init__(self, factor: float = 0.5) -> None:
        if not 0.0 <= factor < 1.0:
            raise ValueError("smoothing factor must be in [0, 1)")
        self.factor = factor
        self._value: float | None = None

    def reset(self) -> None:
        self._value = None

    def update(self, new_value: float) -> float:
        if self._value is None:
            self._value = new_value
        else:
            self._value = self.factor * self._value + (1 - self.factor) * new_value
        return self._value


class Point2DSmoother:
    """Smooths an (x, y) point using two independent exponential smoothers."""

    def __init__(self, factor: float = 0.5) -> None:
        self._x = ExponentialSmoother(factor)
        self._y = ExponentialSmoother(factor)

    def reset(self) -> None:
        self._x.reset()
        self._y.reset()

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        return (self._x.update(point[0]), self._y.update(point[1]))
