from __future__ import annotations


def safety_distance(speed_m_s: float, reward_cfg: dict):
    v = max(0.0, float(speed_m_s))
    # Peng's empirical Dmax relation uses speed in km/h; Dsafe is a physical
    # distance and therefore uses SI speed and seconds.
    v_km_h = 3.6 * v
    dmax = 0.0825 * v_km_h * v_km_h + v_km_h + 10.0
    dsafe = (
        v * reward_cfg["reaction_delay_s"]
        + v * v / 6.68
        + reward_cfg["standstill_distance_m"]
    )
    return dsafe, dmax


def safety_cost(speed_m_s: float, gap_m: float, reward_cfg: dict, collision: bool = False):
    v = max(0.0, float(speed_m_s))
    gap = float(gap_m)
    if collision or gap <= 0:
        return 19.1666666667
    dsafe, dmax = safety_distance(v, reward_cfg)
    if gap <= dsafe:
        return v
    if gap > dmax:
        return gap - dmax
    return 0.0


def compute_reward(components: dict, cfg: dict):
    r = cfg
    total_cost = (
        r["beta_hydrogen"] * components["hydrogen_g"]
        + r["beta_fcs"] * components["fcs_degradation_cumulative"]
        + r["beta_battery"] * components["battery_soh_loss"]
        + r["beta_edu"] * components["edu_clr_cumulative"]
        + r["beta_comfort"] * components["comfort"]
        + components["safety"]
        + r["soc_weight"] * components["soc_penalty"]
    )
    return -float(total_cost)
