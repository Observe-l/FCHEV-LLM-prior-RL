"""
FCHEV driving/energy rule prior for MADDPG fusion.

This program returns normalized actions for both actors:
    driving.action: u_d in [-1, 1]
    energy.action:  u_e in [-1, 1]

It receives the raw, unnormalized 15-dimensional FCHEV observation.
It is deterministic, contains no noise/std, and does not bypass the
environment's physical constraints. It is intended to be fused as:

    u_fused = (1 - lambda) * u_MADDPG + lambda * u_rule

where lambda is supplied externally.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Traffic-light encoding
# -----------------------------------------------------------------------------
RED = 0
GREEN = 1
YELLOW = 2

# -----------------------------------------------------------------------------
# Physical/action bounds
# -----------------------------------------------------------------------------
A_MIN = -1.5
A_MAX = 0.7
V_HARD = 19.4444          # 70 km/h hard vehicle cap, used only for clipping observations

# Observation sentinel used by the environment for "no next target".
SENTINEL = 30000.0

# -----------------------------------------------------------------------------
# Small margins and controller gains
# -----------------------------------------------------------------------------
SPEED_EPS = 1e-3
LINE_AT_MARGIN = 1e-3     # treat d <= this as "at the stop line"
STOP_MARGIN = 3.0         # m before the stop line
PHASE_MARGIN = 1.0        # s safety margin around signal phase boundaries
SPEED_MARGIN = 0.5        # m/s margin below current speed limit
ENTRY_MARGIN = 5.0        # m margin before next speed-region entrance
COMFORT_GAIN = 0.2        # smoothing toward current executed acceleration when safe

GAP_GAIN = 0.02           # m^-1 acceleration gain for gap error
REL_GAIN = 0.35           # s^-1 acceleration gain for relative speed error

# Approximate combined efficiency used only to convert estimated mechanical road
# demand into an approximate electrical bus/FCS power demand.
#
# This is NOT an EDU map lookup and NOT a rated efficiency. It is a coarse
# approximation:
#   ETA_EST ~= approximate EDU combined-boundary efficiency * DC/DC efficiency
#            ~= 0.90 * 0.95
ETA_EST = 0.90 * 0.95


_PHASE_DUR = {
    RED: 47.0,
    GREEN: 40.0,
    YELLOW: 3.0,
}

_PHASE_NEXT = {
    RED: GREEN,
    GREEN: YELLOW,
    YELLOW: RED,
}


def _clip_action(values, lo, hi):
    """Return finite values clipped to [lo, hi] as float32."""
    values = np.where(np.isnan(values), 0.0, values)
    values = np.where(values > hi, hi, values)
    values = np.where(values < lo, lo, values)
    values = np.where(np.isfinite(values), values, 0.0)
    return values.astype(np.float32)


def _predict_signal_phase(t_arr, state, t2):
    """
    Predict the signal state at estimated arrival time.

    The current phase is state for t2 seconds. After the current phase ends,
    advance through the fixed sequence:
        green 40 s, yellow 3 s, red 47 s, repeat.

    Boundary handling is conservative: if arrival is exactly at a transition
    into yellow/red, return the non-green phase.
    """
    state = int(state)
    t_arr = float(t_arr)
    t2 = float(t2)

    if t_arr < 0.0:
        return state

    if t_arr < t2:
        return state

    rem = t_arr - t2
    nxt = _PHASE_NEXT.get(state, GREEN)
    rem_mod = rem % 90.0

    # Three phases cover one full 90 s cycle.
    for _ in range(3):
        dur = _PHASE_DUR.get(nxt, 40.0)
        if rem_mod < dur:
            return nxt
        rem_mod -= dur
        nxt = _PHASE_NEXT.get(nxt, GREEN)

    return nxt


def _stop_accel(v, d):
    """
    Requested acceleration needed to stop before d - STOP_MARGIN.

    This is only a requested acceleration; the environment may clip or fail to
    deliver it. If the remaining distance is too small, request maximum braking.
    """
    if v <= SPEED_EPS:
        return 0.0

    if d <= STOP_MARGIN:
        return A_MIN

    a = -(v * v) / (2.0 * (d - STOP_MARGIN))

    if a < A_MIN:
        return A_MIN
    if a > 0.0:
        return 0.0

    return a


def _delay_upper(v, d, target_time):
    """
    Upper acceleration bound to avoid arriving before target_time.

    If target_time <= 0, stop before the line.
    """
    if target_time <= 0.0:
        return _stop_accel(v, d)

    v_allow = d / target_time
    a = v_allow - v

    if a > A_MAX:
        return A_MAX
    if a < A_MIN:
        return A_MIN

    return a


def _signal_upper_row(v, d, state, t2):
    """
    Return the maximum requested acceleration allowed by the next traffic light.

    This function is evaluated per row because phase prediction is easier to
    keep explicit and conservative. It does not access SUMO.
    """
    state = int(state)

    # No next traffic light.
    if d >= SENTINEL:
        return A_MAX

    if state not in (RED, GREEN, YELLOW):
        return A_MAX

    d = max(float(d), 0.0)
    v = max(float(v), 0.0)
    t2 = max(float(t2), 0.0)

    # At or past the stop line.
    if d <= LINE_AT_MARGIN:
        if state == GREEN:
            return A_MAX
        return A_MIN if v > SPEED_EPS else 0.0

    # Stopped before the line.
    if v <= SPEED_EPS:
        if state == GREEN:
            return A_MAX

        if state == RED:
            # If already stopped, allow acceleration only if a maximum-acceleration
            # start would not reach the line before the current red phase ends.
            target = max(t2 - PHASE_MARGIN, 0.0)
            min_time = np.sqrt(2.0 * d / A_MAX) if d > SPEED_EPS else 0.0

            # Far enough that the immediate signal constraint is not relevant.
            if min_time > 90.0:
                return A_MAX

            if min_time >= target and (d > 20.0 or target <= 0.0):
                return A_MAX

            return 0.0

        # Yellow: crossing is prohibited. Wait.
        return 0.0

    # Moving estimate.
    t_arr = d / v

    # If arrival is more than one full cycle away, do not impose a near-term
    # signal constraint. The rule is recomputed every 1 s step.
    if t_arr > 90.0:
        return A_MAX

    pred = _predict_signal_phase(t_arr, state, t2)

    # If the predicted crossing state is red or yellow, keep before the line.
    if pred != GREEN:
        return _stop_accel(v, d)

    # Predicted green, but currently red/yellow and arrival would occur before
    # the current non-green phase ends. Delay or stop.
    if state != GREEN and t_arr < t2 - PHASE_MARGIN:
        return _delay_upper(v, d, t2 - PHASE_MARGIN)

    # Currently green but arrival is too close to the end of the green phase.
    # This is conservative: it may forgo a green-crossing bonus, but avoids
    # crossing on yellow/red.
    if state == GREEN and t_arr > t2 - PHASE_MARGIN:
        return _stop_accel(v, d)

    return A_MAX


def _estimate_power_kw(v, a):
    """
    Approximate electrical power demand in kW.

    This is a kinematic estimate only:
        P_road = m_eff * v * a + 0.5*rho*A*Cd*v^3 + m*g*Crr*v

    It does not use the actual EDU map, battery model, FCS map, DC/DC dispatch,
    accessory loads, or degradation model. The division by ETA_EST is a coarse
    approximation to convert mechanical road demand into bus/FCS-side demand.
    """
    m_eff = 7000.0 * 1.05          # rotational-mass coefficient
    drag_coeff = 0.5 * 1.225 * 5.8 * 0.55
    roll_coeff = 7000.0 * 9.8 * 0.0085

    p_w = m_eff * v * a + drag_coeff * v**3 + roll_coeff * v

    # Negative road power means braking/regenerative demand. For an FCS request,
    # treat this as zero direct traction demand and allow battery/regen handling.
    p_w = np.maximum(0.0, p_w / ETA_EST)

    # Cap the estimate to avoid absurd values; actual EDU/FCS limits are enforced
    # by the environment.
    return np.clip(p_w / 1000.0, 0.0, 200.0).astype(np.float32)


def predict(observation):
    """
    Deterministic FCHEV rule prior.

    Input:
        observation: raw unnormalized NumPy array shaped (15,) or (batch, 15).

    Output:
        {
            "driving": {"action": (batch,1) float32 in [-1,1],
                        "active": (batch,1) bool True},
            "energy":  {"action": (batch,1) float32 in [-1,1],
                        "active": (batch,1) bool True},
        }
    """
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        if x.shape[0] != 15:
            raise ValueError("observation must be finite with shape (15,) or (batch, 15)")
        x = x.reshape(1, 15)

    if x.ndim != 2 or x.shape[1] != 15:
        raise ValueError("observation must be finite with shape (15,) or (batch, 15)")

    if not np.all(np.isfinite(x)):
        raise ValueError("observation must contain only finite values")

    n = x.shape[0]

    # -----------------------------------------------------------------
    # Raw observation extraction
    # -----------------------------------------------------------------
    soc = np.clip(x[:, 0], 0.0, 1.0)
    battery_soh = np.clip(x[:, 1], 0.0, 1.0)
    fcs_soh = np.clip(x[:, 2], 0.0, 1.0)
    edu_soh = np.clip(x[:, 3], 0.0, 1.0)

    a_ego = np.clip(x[:, 4], A_MIN, A_MAX)
    v = np.clip(x[:, 5], 0.0, V_HARD)

    a_lead = np.clip(x[:, 6], -2.0, 2.0)
    v_lead = np.clip(x[:, 7], 0.0, V_HARD)

    gap = np.clip(x[:, 8], 0.0, SENTINEL)

    tl_state = np.rint(np.clip(x[:, 9], 0.0, 2.0)).astype(np.int32)
    tl_dist = np.maximum(x[:, 10], 0.0)
    tl_time = np.clip(x[:, 11], 0.0, 100.0)

    v_cur = np.clip(x[:, 12], 0.0, V_HARD)
    v_next = np.clip(x[:, 13], 0.0, V_HARD)
    d_entry = np.maximum(x[:, 14], 0.0)

    # -----------------------------------------------------------------
    # Driving prior: following term
    # -----------------------------------------------------------------
    d_safe = 2.0 + 0.5 * v + 0.0625 * v * v
    d_max = 10.0 + v + 0.0825 * v * v

    # Target gap is inside the prescribed safe/maximum range.
    target_gap = d_safe + 0.6 * (d_max - d_safe)

    # If gap is the sentinel value, do not use a huge gap-error term. Relative
    # speed control is still used.
    gap_valid = gap < SENTINEL
    gap_err = gap - target_gap
    gap_err = np.clip(gap_err, -100.0, 200.0)
    gap_err = np.where(gap_valid, gap_err, 0.0)

    a_follow = a_lead + GAP_GAIN * gap_err - REL_GAIN * (v - v_lead)

    # Hard safety term when gap is below D_safe.
    close_gap = np.maximum(0.0, d_safe - gap)
    closing_speed = np.maximum(0.0, v - v_lead)
    safe_brake = a_lead - 0.5 * closing_speed - 0.1 * close_gap

    a_follow = np.where(
        gap < d_safe,
        np.minimum(a_follow, safe_brake),
        a_follow,
    )

    # Collision/overlap: request maximum braking.
    a_follow = np.where(gap <= 1e-3, A_MIN, a_follow)

    a_follow = np.clip(a_follow, A_MIN, A_MAX)

    # -----------------------------------------------------------------
    # Driving prior: current speed-limit constraint
    # -----------------------------------------------------------------
    speed_upper = np.where(
        v > v_cur,
        v_cur - v,
        np.maximum(0.0, v_cur - v - SPEED_MARGIN),
    )
    speed_upper = np.clip(speed_upper, A_MIN, A_MAX)

    # -----------------------------------------------------------------
    # Driving prior: lower speed-region entry shaping
    # -----------------------------------------------------------------
    entry_upper = np.full(n, A_MAX, dtype=np.float32)

    lower_region = (v_next < v_cur - 0.1) & (d_entry < SENTINEL) & (d_entry >= 0.0)

    if np.any(lower_region):
        d = d_entry[lower_region]
        vv = v[lower_region]
        vn = v_next[lower_region]

        a_entry = np.full(d.shape, A_MAX, dtype=np.float32)

        mask_d = d > ENTRY_MARGIN
        if np.any(mask_d):
            denom = 2.0 * (d[mask_d] - ENTRY_MARGIN)
            a_entry[mask_d] = (
                vn[mask_d] * vn[mask_d] - vv[mask_d] * vv[mask_d]
            ) / denom

        mask_no = ~mask_d
        if np.any(mask_no):
            a_entry[mask_no] = np.where(
                vv[mask_no] > vn[mask_no],
                vn[mask_no] - vv[mask_no],
                A_MAX,
            )

        entry_upper[lower_region] = np.clip(a_entry, A_MIN, A_MAX)

    # -----------------------------------------------------------------
    # Driving prior: traffic-light constraint
    # -----------------------------------------------------------------
    signal_upper = np.empty(n, dtype=np.float32)
    for i in range(n):
        signal_upper[i] = _signal_upper_row(
            float(v[i]),
            float(tl_dist[i]),
            int(tl_state[i]),
            float(tl_time[i]),
        )

    # -----------------------------------------------------------------
    # Combine driving constraints
    # -----------------------------------------------------------------
    a_req = a_follow.copy()

    # Safety/limit constraints are upper bounds on acceleration.
    a_req = np.minimum(a_req, speed_upper)
    a_req = np.minimum(a_req, entry_upper)
    a_req = np.minimum(a_req, signal_upper)

    # Comfort smoothing only when no immediate safety/limit constraint is active.
    critical = (
        (gap < d_safe)
        | (v > v_cur + 1e-3)
        | (signal_upper < -1e-3)
        | (entry_upper < -1e-3)
    )

    a_smooth = a_req + COMFORT_GAIN * (a_ego - a_req)
    a_req = np.where(critical, a_req, a_smooth)

    # Reapply constraints after smoothing.
    a_req = np.minimum(a_req, speed_upper)
    a_req = np.minimum(a_req, entry_upper)
    a_req = np.minimum(a_req, signal_upper)

    # Avoid requesting positive acceleration when already near the posted limit.
    at_limit = v >= v_cur - 0.1
    a_req = np.where(at_limit, np.minimum(a_req, 0.0), a_req)

    a_req = np.clip(a_req, A_MIN, A_MAX)
    a_req = np.where(np.isfinite(a_req), a_req, 0.0).astype(np.float32)

    # Physical acceleration -> normalized driving action.
    u_d = np.where(a_req < 0.0, a_req / 1.5, a_req / 0.7)
    u_d = _clip_action(u_d, -1.0, 1.0).reshape(n, 1)

    # -----------------------------------------------------------------
    # Energy prior: approximate power demand
    # -----------------------------------------------------------------
    # The energy actor only chooses FCS power request. This estimate uses the
    # rule's driving prior acceleration as a proxy for upcoming traction demand.
    # It is an approximation and may mismatch the final fused driving action.
    p_demand_kw = _estimate_power_kw(v, a_req)

    # SOC regulation around target 0.5.
    # If SOC is high, reduce FCS request to allow battery discharge.
    # If SOC is low, increase FCS request to allow battery charging.
    soc_gain = 20.0 + 40.0 * battery_soh
    p_req_kw = p_demand_kw - soc_gain * (soc - 0.5)
    p_req_kw = np.clip(p_req_kw, 0.0, 50.0)

    # Stronger SOC-boundary behavior.
    high_soc = soc > 0.85
    low_soc = soc < 0.25

    p_req_kw = np.where(
        high_soc,
        np.minimum(p_req_kw, np.maximum(0.0, p_demand_kw - 20.0)),
        p_req_kw,
    )

    p_req_kw = np.where(
        low_soc,
        np.maximum(p_req_kw, np.minimum(50.0, p_demand_kw + 20.0)),
        p_req_kw,
    )

    # When estimated demand is very low, cap FCS request to avoid unnecessary
    # high power during low-load states.
    p_req_kw = np.where(
        p_demand_kw < 1.0,
        np.clip(p_req_kw, 0.0, 25.0),
        p_req_kw,
    )

    # Heuristic degradation caps based on observed SOH values.
    # This is not the exact component aging model.
    min_soh = np.minimum(np.minimum(battery_soh, fcs_soh), edu_soh)
    cap_kw = 50.0 * (0.70 + 0.30 * min_soh)
    p_req_kw = np.minimum(p_req_kw, cap_kw)
    p_req_kw = _clip_action(p_req_kw, 0.0, 50.0)

    # Requested FCS power -> normalized energy action.
    # P_req = 25 * (u_e + 1), so u_e = P_req / 25 - 1.
    u_e = p_req_kw / 25.0 - 1.0
    u_e = _clip_action(u_e, -1.0, 1.0).reshape(n, 1)

    # -----------------------------------------------------------------
    # Always active masks
    # -----------------------------------------------------------------
    driving_mask = np.ones((n, 1), dtype=bool)
    energy_mask = np.ones((n, 1), dtype=bool)

    return {
        "driving": {
            "action": u_d,
            "active": driving_mask,
        },
        "energy": {
            "action": u_e,
            "active": energy_mask,
        },
    }