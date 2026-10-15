from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FCSPoint:
    power_kw: float
    hydrogen_rate_g_s: float
    efficiency_paper: float
    efficiency_effective: float


class FCSModel:
    def __init__(self, table: np.ndarray, maximum_power_kw: float):
        self.table = np.asarray(table, dtype=np.float64)
        self.maximum_power_kw = float(maximum_power_kw)
        if self.table.ndim != 2 or self.table.shape[1] != 4:
            raise ValueError("fcs_parameters.txt must have four numeric columns")
        if not np.all(np.diff(self.table[:, 0]) > 0):
            raise ValueError("FCS power samples must be strictly increasing")

    def lookup(self, power_kw: float) -> FCSPoint:
        power = float(np.clip(power_kw, 0.0, self.maximum_power_kw))
        if power <= 0.0:
            return FCSPoint(0.0, 0.0, 0.0, 0.0)
        hydrogen = float(np.interp(power, self.table[:, 0], self.table[:, 1]))
        paper_eta = float(np.interp(power, self.table[:, 0], self.table[:, 2]))
        effective_eta = float(np.interp(power, self.table[:, 0], self.table[:, 3]))
        return FCSPoint(power, hydrogen, paper_eta, effective_eta)
