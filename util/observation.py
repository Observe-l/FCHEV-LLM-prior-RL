from __future__ import annotations

import numpy as np


def normalize_observation(observation, vehicle_cfg):
    """Fixed, invertible feature scaling; the environment itself exposes SI units."""
    x = np.asarray(observation, dtype=np.float32).copy()
    x[4] /= max(vehicle_cfg["emergency_deceleration_m_s2"], 1e-6)
    x[5] /= max(vehicle_cfg["max_speed_mps"], 1e-6)
    x[6] /= max(vehicle_cfg["emergency_deceleration_m_s2"], 1e-6)
    x[7] /= max(vehicle_cfg["max_speed_mps"], 1e-6)
    x[8] /= 300.0
    return x
