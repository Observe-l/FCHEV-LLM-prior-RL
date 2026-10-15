from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FCSDegradationResult:
    increment: float
    cumulative: float
    soh: float
    startup_percent: float
    low_power_percent: float
    high_power_percent: float
    load_change_percent: float


class FCSDegradation:
    """Song/Peng four-mechanism degradation; internal health values are fractions."""

    def __init__(self, config: dict):
        self.cfg = config
        self.cumulative = 0.0
        self.previous_power_kw = 0.0

    def reset(self):
        self.cumulative = 0.0
        self.previous_power_kw = 0.0

    def step(self, power_kw: float, dt_s: float) -> FCSDegradationResult:
        p = max(0.0, float(power_kw))
        c = self.cfg
        startup_pct = c["startup_degradation_percent"] if p > 0 and self.previous_power_kw <= 0 else 0.0
        low_pct = (
            c["low_power_degradation_percent_per_hour"] * dt_s / 3600.0
            if 0 < p < c["low_power_threshold_kw"] else 0.0
        )
        high_pct = (
            c["high_power_degradation_percent_per_hour"] * dt_s / 3600.0
            if p > c["high_power_threshold_kw"] else 0.0
        )
        load_pct = (
            c["load_change_degradation_percent"]
            * abs(p - self.previous_power_kw)
            / max(c["high_power_threshold_kw"] - c["low_power_threshold_kw"], 1e-12)
        )
        increment = (startup_pct + low_pct + high_pct + load_pct) / 100.0
        self.cumulative = min(1.0, self.cumulative + increment)
        self.previous_power_kw = p
        return FCSDegradationResult(
            increment, self.cumulative, 1.0 - self.cumulative,
            startup_pct, low_pct, high_pct, load_pct,
        )


@dataclass
class EDUDegradationResult:
    increment: float
    cumulative: float
    soh: float


class EDUDegradation:
    def __init__(self, reference_loss_w: float, lifetime_hours: float):
        if not np.isfinite(reference_loss_w) or reference_loss_w <= 0:
            raise ValueError("EDU reference loss must be positive and finite")
        self.reference_loss_w = float(reference_loss_w)
        self.lifetime_s = float(lifetime_hours) * 3600.0
        self.cumulative = 0.0

    def reset(self):
        self.cumulative = 0.0

    def step(self, loss_w: float, dt_s: float) -> EDUDegradationResult:
        loss = max(0.0, float(loss_w))
        inc = loss * float(dt_s) / (self.reference_loss_w * self.lifetime_s)
        self.cumulative = min(1.0, self.cumulative + inc)
        return EDUDegradationResult(inc, self.cumulative, 1.0 - self.cumulative)
