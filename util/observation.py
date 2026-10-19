from __future__ import annotations

import numpy as np


def normalize_observation(observation, vehicle_cfg):
    """Keep physical observations in SI units; scale only the signal category."""
    x = np.asarray(observation, dtype=np.float32).copy()
    # Preserve acceleration, speed, gap, signal timing/distance, speed limits,
    # and speed-region distance in their physical units. Only map the
    # categorical {red, green, yellow} encoding {0, 1, 2} into [0, 1].
    x[9] /= 2.0
    return x
