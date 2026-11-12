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
    
def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Smooths one scalar signal. Feed it ``(timestamp, value)`` pairs.
 
    ``min_cutoff``: lower = smoother at rest (more lag when slow).
    ``beta``: higher = less lag during fast movement.
    ``d_cutoff``: smoothing of the speed estimate; rarely needs changing.
    """
 
    def __init__(
        self,
        min_cutoff: float = 1.5,
        beta: float = 8.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._t_prev: float | None = None
        self._x_raw_prev = 0.0
        self._x_hat_prev = 0.0
        self._dx_hat_prev = 0.0
 
    def reset(self) -> None:
        self._t_prev = None
        self._x_raw_prev = 0.0
        self._x_hat_prev = 0.0
        self._dx_hat_prev = 0.0
 
    def __call__(self, timestamp: float, value: float) -> float:
        if self._t_prev is None:
            self._t_prev = timestamp
            self._x_raw_prev = value
            self._x_hat_prev = value
            self._dx_hat_prev = 0.0
            return value
 
        dt = max(timestamp - self._t_prev, 1e-6)
 
        dx = (value - self._x_raw_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_hat_prev
 
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)
        x_hat = a * value + (1.0 - a) * self._x_hat_prev
 
        self._t_prev = timestamp
        self._x_raw_prev = value
        self._x_hat_prev = x_hat
        self._dx_hat_prev = dx_hat
        return x_hat