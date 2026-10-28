from __future__ import annotations

import numpy as np


def normalize_observation(observation, vehicle_cfg=None, enabled=True):
    """Scale the 15 policy inputs, or preserve legacy scaling when disabled.

    ``vehicle_cfg`` is retained in the signature for call-site compatibility.
    Environment observations and reward calculations remain in physical units;
    this transform is only for actor/critic inputs.
    """
    x = np.asarray(observation, dtype=np.float32).copy()
    if x.shape[-1] != 15:
        raise ValueError(
            f"Expected a 15-element FCHEV observation, received shape {x.shape}"
        )

    # Keep the legacy traffic-state encoding when normalization is disabled.
    # Raw states are red=0, green=1, yellow=2; policy states are 0, 0.5, 1.
    if not enabled:
        x[..., 9] /= 2.0
        return x

    # [SOC, battery SOH, FCS SOH, EDU SOH, ego acc/speed, lead acc/speed,
    #  gap, traffic state, TLS distance/time, current/next limits,
    #  distance to next speed region]. SOH fields become remaining-life loss.
    x[..., 1:4] = (1.0 - x[..., 1:4]) * 1_000.0
    x[..., 4] /= 1.5
    x[..., 5] /= 20.0
    x[..., 6] /= 1.5
    x[..., 7] /= 20.0
    x[..., 8] /= 1_000.0
    x[..., 9] /= 2.0
    x[..., 10] /= 1_000.0
    x[..., 11] /= 100.0
    x[..., 12] /= 20.0
    x[..., 13] /= 20.0
    x[..., 14] /= 1_000.0
    return x
