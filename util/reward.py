from __future__ import annotations

import numpy as np


def safety_distance(speed_m_s: float):
    v = max(0.0, float(speed_m_s))
    # Screenshot equations use ego speed v_h directly in m/s.
    dmin = 2.0 + 0.5 * v + 0.0625 * v * v
    dmax = 10.0 + v + 0.0825 * v * v
    return dmin, dmax


def speeding_reward(speed_m_s: float, speed_limit_m_s: float, beta: float = 1.0):
    """Return a linear overspeed penalty and excess speed, both in SI units."""
    speed = max(0.0, float(speed_m_s))
    speed_limit = max(0.0, float(speed_limit_m_s))
    delta_v = max(0.0, speed - speed_limit)
    return -float(beta) * delta_v, delta_v


def predicted_speed_at_limit_entry(
    speed_m_s: float,
    speed_limit_m_s: float,
    distance_to_entry_m: float,
    deceleration_m_s2: float = 0.75,
):
    """Estimate entry speed under constant braking and its relative overspeed.

    ``deceleration_m_s2`` is a positive braking magnitude. An invalid or
    nonpositive limit has no speed-limit error and returns ``(speed, 0)``.
    """
    speed = max(0.0, float(speed_m_s))
    limit = float(speed_limit_m_s)
    distance = max(0.0, float(distance_to_entry_m))
    deceleration = max(0.0, float(deceleration_m_s2))
    if not np.isfinite(limit) or limit <= 0.0:
        return speed, 0.0
    entry_speed = float(np.sqrt(max(0.0, speed * speed - 2.0 * deceleration * distance)))
    relative_overspeed = max(0.0, (entry_speed - limit) / limit)
    return entry_speed, float(relative_overspeed)


def following_reward(speed_m_s: float, gap_m: float, d_ref_m: float,
                     previous_gap_m: float, c0: float, c1: float,
                     c2: float, c3: float, *, gap_valid: bool = True):
    """Shape following reward using the current gap and per-step gap change."""
    # A SUMO sentinel may indicate that the designated lead is no longer in
    # front (or is absent). It is not a physical gap and must not enter the
    # delta-gap branch, where it could create an enormous artificial reward.
    if not gap_valid:
        return 0.0
    v = max(0.0, float(speed_m_s))
    d = float(gap_m)
    d_safe, d_max = safety_distance(v)
    # Keep the close-gap branch continuous at zero for a collision terminal step.
    d = max(0.0, d)
    if d < d_safe:
        normalized_deficit = float(np.clip(1.0 - d / d_safe, 0.0, 1.0))
        return float(c0) - float(c1) * normalized_deficit**3
    if d <= d_max:
        return float(c0)
    d_ref = float(d_ref_m)
    if d < d_ref:
        return float(c0) - float(c2) * (d - d_max) / (d_ref - d_max)
    delta_d = float(previous_gap_m) - d
    return float(c3) * delta_d


def traffic_light_reward(speed_m_s: float, distance_m: float, signal_state,
                         phase_remaining_s: float,
                         c_green: float = 10.0, c_non_green: float = 10.0):
    """Reward a predicted green arrival and penalize predicted yellow/red arrival.

    Arrival time is estimated as distance / current speed. No reward is active
    while stopped, when no signal is observable, or when the current phase is
    predicted to change before arrival.
    """
    speed = max(0.0, float(speed_m_s))
    distance = max(0.0, float(distance_m))
    remaining = max(0.0, float(phase_remaining_s))
    if speed <= 1e-9 or not np.isfinite(distance) or not np.isfinite(remaining):
        return 0.0, float("inf"), False

    arrival_time = distance / speed
    if not arrival_time < remaining:
        return 0.0, arrival_time, False

    if isinstance(signal_state, str):
        state = signal_state.strip().lower()[:1]
        green = state in ("g",)
    else:
        state_value = float(signal_state)
        green = np.isclose(state_value, 1.0)
    value = float(c_green) if green else -float(c_non_green)
    return value, arrival_time, True


def compute_reward(components: dict, cfg: dict):
    """Sum signed reward terms and negate the nonnegative energy costs."""
    betas = cfg["reward_coefficients"]
    costs = {
        "following": float(components["following"]),
        "hydrogen": max(0.0, float(components["hydrogen_g"])),
        "fcs": 100.0 * max(0.0, float(components["fcs_degradation_increment"])),
        "battery": 100.0 * max(0.0, float(components["battery_soh_loss"])),
        "edu": 100.0 * max(0.0, float(components["edu_clr_increment"])),
        "comfort": float(components["comfort"]),
        "soc": max(0.0, float(components["soc_penalty"])),
    }
    coefficient_keys = {
        "following": "beta_0",
        "hydrogen": "beta_1",
        "fcs": "beta_2",
        "battery": "beta_3",
        "edu": "beta_4",
        "comfort": "beta_5",
        "soc": "beta_6",
    }
    total_reward = 0.0
    for name, cost in costs.items():
        coefficient = float(betas[coefficient_keys[name]])
        contribution = (
            coefficient * cost
            if name in ("following", "comfort")
            else -coefficient * cost
        )
        # Keep cost metrics nonnegative even though the reward component itself
        # carries its negative sign for the additive reward formulation.
        components[f"cost_{name}"] = max(0.0, -cost) if name == "comfort" else cost
        components[f"beta_{name}"] = coefficient
        components[f"reward_contribution_{name}"] = contribution
        total_reward += contribution
    return float(total_reward)
