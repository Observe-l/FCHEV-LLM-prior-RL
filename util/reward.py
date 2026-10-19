from __future__ import annotations


def safety_distance(speed_m_s: float):
    v = max(0.0, float(speed_m_s))
    # Screenshot equations use ego speed v_h directly in m/s.
    dmin = 2.0 + 0.5 * v + 0.0625 * v * v
    dmax = 10.0 + v + 0.0825 * v * v
    return dmin, dmax


def following_reward(speed_m_s: float, gap_m: float, d_ref_m: float,
                     previous_gap_m: float, c0: float, c1: float,
                     c2: float, c3: float):
    """Shape following reward using the current gap and per-step gap change."""
    v = max(0.0, float(speed_m_s))
    d = float(gap_m)
    d_safe, d_max = safety_distance(v)
    # Keep the close-gap branch continuous at zero for a collision terminal step.
    d = max(0.0, d)
    if d < d_safe:
        return float(c0) - float(c1) * (1.0 - d / d_safe) ** 2
    if d <= d_max:
        return float(c0)
    d_ref = float(d_ref_m)
    if d < d_ref:
        return float(c0) - float(c2) * (d - d_max) / (d_ref - d_max)
    delta_d = float(previous_gap_m) - d
    return float(c3) * delta_d


def compute_reward(components: dict, cfg: dict):
    """Apply signed following reward and direct per-step penalties."""
    betas = cfg["reward_coefficients"]
    costs = {
        "following": float(components["following"]),
        "hydrogen": max(0.0, float(components["hydrogen_g"])),
        "fcs": 100.0 * max(0.0, float(components["fcs_degradation_increment"])),
        "battery": 100.0 * max(0.0, float(components["battery_soh_loss"])),
        "edu": 100.0 * max(0.0, float(components["edu_clr_increment"])),
        "comfort": max(0.0, float(components["comfort"])),
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
        contribution = coefficient * cost if name == "following" else -coefficient * cost
        components[f"cost_{name}"] = cost
        components[f"beta_{name}"] = coefficient
        components[f"reward_contribution_{name}"] = contribution
        total_reward += contribution
    return float(total_reward)
