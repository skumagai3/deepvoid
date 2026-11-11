"""Early-stopping bookkeeping, kept free of TensorFlow so it can be tested.

Two thresholds are tracked separately:

- the **best** score so far, which decides when ``best.keras`` is saved
  (any improvement counts);
- the **reference** score, which decides when training stops. The patience
  counter only resets when the score beats the reference by at least
  ``min_delta``. Without it, a score creeping up by 1e-4 per epoch resets
  the counter every epoch and a run goes on long after it has stopped
  getting meaningfully better.
"""

from __future__ import annotations

import math


class Stopper:
    def __init__(self, mode: str = "max", patience: int = 15, min_delta: float = 0.0,
                 state: dict | None = None):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        state = state or {}
        self.mode, self.patience, self.min_delta = mode, patience, min_delta
        worst = -math.inf if mode == "max" else math.inf
        self.best = state.get("best", worst)
        self.best_epoch = state.get("best_epoch", -1)
        self.ref = state.get("ref", self.best)
        self.wait = state.get("wait", 0)

    def _better(self, a: float, b: float, margin: float = 0.0) -> bool:
        return a > b + margin if self.mode == "max" else a < b - margin

    def update(self, epoch: int, value: float) -> bool:
        """Record one epoch's score; return True if it is the best so far."""
        value = float(value)
        improved = self._better(value, self.best)
        if improved:
            self.best, self.best_epoch = value, epoch
        if self._better(value, self.ref, self.min_delta):
            self.ref, self.wait = value, 0
        else:
            self.wait += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return self.wait >= self.patience

    def state(self) -> dict:
        return {"best": self.best, "best_epoch": self.best_epoch, "ref": self.ref,
                "wait": self.wait}
