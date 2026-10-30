"""
fcbev_follow_energy_prior.py

Deterministic physical-action rule prior for the FCHEV/MADDPG interface.

This program returns:
  - driving.action: requested acceleration in [-1.5, 0.7] m/s^2
  - energy.action: requested FCS output power in [0, 50] kW

It does not replace the environment, does not add noise, and does not bypass
physical constraints. The environment may still clip or modify the requested
actions.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Physical and task constants
# -----------------------------------------------------------------------------

A_MIN = -1.5
A_MAX = 0.7

V_CAP = 19.4444          # 70 km/h hard vehicle cap, m/s
MAIN_LIMIT = 16.6667     # ordinary road limit, m/s
SENTINEL = 30000.0       # "no visible target / outside effective range"

EPS = 1e-6
SPEED_EPS = 1e-3

# Approximate vehicle-dynamics constants used only for an energy-demand estimate.
M_EFF = 7000.0 * 1.05                         # rotational-mass-corrected mass
DRAG_COEF = 0.5 * 1.225 * 0.55 * 5.8          # aerodynamic drag coefficient
ROLL_FORCE = M_EFF * 9.8 * 0.0085             # rolling-resistance force
ETA_EST = 0.88                                # approximate combined EDU+DC/DC efficiency
SOC_GAIN_KW = 100.0                           # heuristic SOC-correction gain


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _clip_scalar(x, lo, hi):
    return float(np.clip(x, lo, hi))


def _as_2d(observation):
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        if x.size != 15:
            raise ValueError("observation must be shape (15,) or (batch, 15)")
        x = x.reshape(1, 15)
    elif x.ndim == 2:
        if x.shape[1] != 15:
            raise ValueError("observation must be shape (15,) or (batch, 15)")
    else:
        raise ValueError("observation must be shape (15,) or (batch, 15)")

    if not np.all(np.isfinite(x)):
        raise ValueError("observation contains non-finite values")

    return x


def _signal_state_int(state):
    """
    Raw signal encoding:
      0 = red
      1 = green
      2 = yellow
    """
    if not np.isfinite(state):
        return None

    state = float(state)
    if state < -0.5 or state > 2.5:
        return None

    st = int(np.floor(state + 0.5))
    if st not in (0, 1, 2):
        return None

    return st


def _predict_signal_state(arrival, state, rem):
    """
    Predict the signal state at the estimated arrival time.

    Uses the documented repeating cycle:
      green 40 s, yellow 3 s, red 47 s, total 90 s.

    Conservative convention:
      Arrival exactly at a phase boundary is treated as the following
      non-green phase, to avoid ambiguous yellow/red crossing.
    """
    if arrival <= EPS:
        return state

    if arrival < rem - EPS:
        return state

    t = max(0.0, arrival - rem)

    if state == 1:
        # Current green ends, then yellow 3 s, then red 47 s.
        if t <= 3.0:
            return 2
        t -= 3.0
        if t <= 47.0:
            return 0
        t -= 47.0

    elif state == 2:
        # Current yellow ends, then red 47 s.
        if t <= 47.0:
            return 0
        t -= 47.0

    # For state 0, current red ends after rem seconds; t is time after red end.
    t = t % 90.0

    if t < 40.0:
        return 1
    if t < 43.0:
        return 2
    return 0


def _traffic_accel_cap(ego_v, dist, state, rem):
    """
    Return an acceleration cap imposed by the traffic light, or None if no cap.

    The cap is conservative:
      - If the predicted crossing state is green, no cap is imposed.
      - If the predicted crossing state is yellow or red, the rule requests a
        non-positive acceleration sufficient to slow before the stop line.
      - If the ego is at the stop line and the signal is not clearly green,
        the cap is negative if moving, otherwise zero.
    """
    st = _signal_state_int(state)
    if st is None:
        return None

    if not np.isfinite(dist) or dist >= SENTINEL:
        return None

    dist = max(0.0, float(dist))

    if not np.isfinite(rem):
        rem = 0.0
    rem = max(0.0, float(rem))

    # At the stop line.
    if dist == 0.0:
        if st == 1 and rem > EPS:
            return None
        return A_MIN if ego_v > SPEED_EPS else 0.0

    # Stopped before the line.
    if ego_v <= SPEED_EPS:
        if st == 1:
            # Check whether the ego could reach the stop line before green ends
            # using the maximum available acceleration.
            max_reach = ego_v * rem + 0.5 * A_MAX * rem * rem
            if max_reach + 0.5 >= dist:
                return None
            return 0.0
        return 0.0

    arrival = dist / ego_v
    pred = _predict_signal_state(arrival, st, rem)

    if pred == 1:
        return None

    # Conservative stop-before-line deceleration estimate.
    a_stop = -(ego_v * ego_v) / (2.0 * dist)
    return float(max(A_MIN, min(0.0, a_stop)))


def _speed_cap(ego_v, cur_limit):
    """
    1-step speed-limit cap.

    This is not a target-speed command. It limits the requested acceleration so
    that the next-step speed is not allowed to exceed the current posted limit.
    """
    allowed = cur_limit - ego_v
    if allowed >= 0.0:
        return float(min(A_MAX, allowed))
    return float(max(A_MIN, allowed))


def _entry_cap(ego_v, cur_limit, next_limit, next_dist):
    """
    Kinematic cap for approaching a lower posted-speed region.

    Uses a 1-step look-ahead approximation:
      v_entry^2 ~= v^2 + 2 a d_entry

    This is a soft prior cap, not an exact reward reproduction.
    """
    if next_limit >= cur_limit - 1e-3 or next_dist >= SENTINEL:
        return None

    d = max(0.0, float(next_dist))

    if d <= EPS:
        if ego_v > next_limit + 1e-3:
            return A_MIN
        return 0.0

    cap = (next_limit * next_limit - ego_v * ego_v) / (2.0 * d)
    return float(max(A_MIN, min(A_MAX, cap)))


def _desired_acceleration(row):
    """
    Driving prior:
      - follow the live lead vehicle,
      - keep gap between D_safe and D_max whenever possible,
      - obey current and upcoming posted speed limits,
      - respect traffic-light crossing constraints,
      - apply only mild comfort smoothing in a safe steady-following band.
    """
    ego_v = _clip_scalar(row[5], 0.0, V_CAP)
    lead_v = _clip_scalar(row[7], 0.0, V_CAP)
    lead_a = _clip_scalar(row[6], A_MIN, A_MAX)
    ego_a = _clip_scalar(row[4], A_MIN, A_MAX)

    gap = float(row[8])
    if not np.isfinite(gap):
        gap = SENTINEL

    if gap < 0.0:
        gap_eff = 0.0
    elif gap >= SENTINEL:
        gap_eff = SENTINEL
    else:
        gap_eff = gap

    cur_limit = float(row[12])
    if not np.isfinite(cur_limit) or cur_limit <= 0.0:
        cur_limit = MAIN_LIMIT
    cur_limit = min(cur_limit, V_CAP)

    next_limit = float(row[13])
    if not np.isfinite(next_limit) or next_limit <= 0.0:
        next_limit = cur_limit
    next_limit = min(next_limit, V_CAP)

    next_dist = float(row[14])
    if not np.isfinite(next_dist):
        next_dist = SENTINEL

    # Following-distance bounds consistent with the documented reward model.
    d_safe = 2.0 + 0.5 * ego_v + 0.0625 * ego_v * ego_v
    d_max = 10.0 + ego_v + 0.0825 * ego_v * ego_v
    target_gap = d_safe + 0.5 * (d_max - d_safe)

    # Simple following controller:
    # lead-acceleration feedforward + gap error + relative-speed correction.
    k_gap = 0.03
    k_rel = 0.35
    a_follow = (
        lead_a
        + k_gap * (gap_eff - target_gap)
        + k_rel * (lead_v - ego_v)
    )

    speed_cap = _speed_cap(ego_v, cur_limit)
    entry_cap = _entry_cap(ego_v, cur_limit, next_limit, next_dist)
    tcap = _traffic_accel_cap(ego_v, row[10], row[9], row[11])

    a_candidate = min(a_follow, speed_cap)

    if entry_cap is not None:
        a_candidate = min(a_candidate, entry_cap)

    if tcap is not None:
        a_candidate = min(a_candidate, tcap)

    # If the gap is already unsafe and the ego is not clearly slower than the
    # lead, do not request positive acceleration.
    if gap_eff < d_safe and ego_v > lead_v - 0.5:
        a_candidate = min(a_candidate, 0.0)

    # Mild comfort smoothing only when following is already safe, steady, and
    # not constrained by speed-limit entry or traffic-light crossing logic.
    comfort_band = (
        gap_eff >= d_safe
        and gap_eff <= d_max
        and abs(ego_v - lead_v) <= 1.0
        and tcap is None
        and ego_v < cur_limit - 0.2
        and (next_limit >= cur_limit - 1e-3 or ego_v < next_limit - 0.2)
    )

    if comfort_band:
        max_delta = 0.20
        a_candidate = ego_a + np.clip(a_candidate - ego_a, -max_delta, max_delta)

    # Reapply hard road/physical caps after smoothing.
    speed_cap = _speed_cap(ego_v, cur_limit)
    a_candidate = min(a_candidate, speed_cap)

    if entry_cap is not None:
        a_candidate = min(a_candidate, entry_cap)

    if tcap is not None:
        a_candidate = min(a_candidate, tcap)

    return _clip_scalar(a_candidate, A_MIN, A_MAX)


def _estimate_mech_power_kw(v, a_est):
    """
    Approximate mechanical power demand at the wheel/drive-unit boundary.

    This is an estimate only. It uses the documented vehicle parameters and a
    single approximate efficiency value. It does not use the exact EPA IPM EDU
    map, exact EDU torque-speed operating regions, exact FCS hydrogen table,
    exact battery electrothermal model, or actual energy-dispatch state.
    """
    v = _clip_scalar(v, 0.0, V_CAP)
    a = _clip_scalar(a_est, A_MIN, A_MAX)

    inertial = M_EFF * a * v
    resistive = (DRAG_COEF * v * v + ROLL_FORCE) * v
    p_mech = inertial + resistive

    if p_mech <= 0.0:
        # Negative mechanical power is treated as regeneration potential.
        # The energy prior then prefers low or zero FCS request.
        return 0.0

    # Approximate combined efficiency. This is not a rated efficiency claim.
    return float(p_mech / (ETA_EST * 1000.0))


def _energy_power_kw(row, a_est):
    """
    Energy prior:
      - estimate mechanical demand from ego speed and the rule's driving prior,
      - move requested FCS power toward the SOC target of 0.5,
      - reduce requested FCS power when SOC is high or the vehicle is braking,
      - cap requested FCS power when FCS/EDU/battery SOH is low,
      - keep the requested FCS output power within [0, 50] kW.
    """
    soc = _clip_scalar(row[0], 0.15, 0.90)
    batt_soh = _clip_scalar(row[1], 0.0, 1.0)
    fcs_soh = _clip_scalar(row[2], 0.0, 1.0)
    edu_soh = _clip_scalar(row[3], 0.0, 1.0)
    v = _clip_scalar(row[5], 0.0, V_CAP)

    demand = _estimate_mech_power_kw(v, a_est)

    # Heuristic requested-power caps based on component health.
    max_fcs = 50.0
    max_fcs = min(max_fcs, 50.0 * max(0.5, fcs_soh))
    max_fcs = min(max_fcs, 50.0 * max(0.5, edu_soh))

    # If SOC is already high and battery SOH is reduced, avoid requesting
    # excess FCS power that could increase charging stress.
    if soc > 0.55:
        max_fcs = min(max_fcs, 50.0 * max(0.5, batt_soh))

    # SOC-management correction.
    # SOC > 0.5 reduces requested FCS power.
    # SOC < 0.5 increases requested FCS power.
    p = demand - SOC_GAIN_KW * (soc - 0.5)

    # If demand exceeds the health-limited FCS request and SOC is not high,
    # request the maximum allowed FCS power to reduce battery discharge.
    if demand > max_fcs and soc < 0.60:
        p = max(p, max_fcs)

    # Low-demand logic.
    if demand <= 1.0:
        if soc <= 0.45:
            # SOC below target: allow a small charging request.
            p = max(p, 5.0)
        elif soc >= 0.55:
            # SOC above target: prefer no FCS request.
            p = 0.0
        else:
            # Near target: keep request small.
            p = min(max(p, 0.0), 5.0)

    # During braking, if SOC is not very low, keep FCS request low to favor
    # regeneration and lower hydrogen use.
    if a_est < -0.2 and soc > 0.45:
        p = min(p, 0.0)

    # If SOC is very low and components are reasonably healthy, ensure at least
    # a modest charging request.
    if soc <= 0.25 and p < 10.0 and batt_soh > 0.7 and fcs_soh > 0.7:
        p = max(p, 10.0)

    # If SOC is already high and demand is low, request no FCS power.
    if soc >= 0.75 and demand < 5.0:
        p = 0.0

    # Final physical request bounds.
    p = _clip_scalar(p, 0.0, max_fcs)

    # Remove tiny nonzero requests when there is essentially no demand and SOC
    # is not below target. This reduces unnecessary load changes in the prior.
    if p < 1.0 and demand < 1.0 and soc > 0.5:
        p = 0.0

    return float(p)


def predict(observation):
    """
    Required rule-program entry point.

    Accepts:
      - observation with shape (15,), or
      - batched observation with shape (batch, 15).

    Returns:
      {
        "driving": {"action": (batch, 1) float32 acceleration m/s^2,
                    "active": (batch, 1) bool},
        "energy":  {"action": (batch, 1) float32 FCS power kW,
                    "active": (batch, 1) bool}
      }

    Both actors are active on every observation.
    """
    x = _as_2d(observation)
    n = x.shape[0]

    accel = np.empty((n, 1), dtype=np.float32)
    power = np.empty((n, 1), dtype=np.float32)

    for i in range(n):
        row = x[i]

        a = _desired_acceleration(row)
        p = _energy_power_kw(row, a)

        accel[i, 0] = np.float32(a)
        power[i, 0] = np.float32(p)

    active_driving = np.ones((n, 1), dtype=bool)
    active_energy = np.ones((n, 1), dtype=bool)

    return {
        "driving": {
            "action": accel,
            "active": active_driving,
        },
        "energy": {
            "action": power,
            "active": active_energy,
        },
    }