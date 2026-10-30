import numpy as np


# -----------------------------------------------------------------------------
# Constants and physical approximations
# -----------------------------------------------------------------------------

_DT = 1.0
_H = 12                         # finite horizon used by the rule: 12 s
_ACTIONS = np.linspace(-1.5, 0.7, 23, dtype=np.float32)

_SPEED_CAP = 70.0 / 3.6         # 19.444... m/s hard ego vehicle cap
_NO_SIGNAL_DIST = 29999.0       # sentinel threshold for “no visible target”

# Traffic-light cycle:
# state 0 = red, state 1 = green, state 2 = yellow
# duration by state index: red=47 s, green=40 s, yellow=3 s
# next phase order: red -> green -> yellow -> red
_SIGNAL_DUR = np.array([47.0, 40.0, 3.0], dtype=np.float64)
_SIGNAL_NEXT = np.array([1, 2, 0], dtype=np.int64)

# Vehicle parameters from the project description.
_MASS = 7000.0
_ROTATIONAL_MASS_COEFF = 1.05
_M_EFF = _MASS * _ROTATIONAL_MASS_COEFF

_CRR = 0.0085
_CD = 0.55
_FRONT_AREA = 5.8
_RHO = 1.225
_G = 9.8

# Approximate battery pack energy from nominal cell data:
# 106 series * 3.3 V nominal, 134 parallel * 2.3 Ah nominal.
_BATTERY_ENERGY_KWH = (106.0 * 3.3 * 134.0 * 2.3) / 1000.0
_BATTERY_ENERGY_KWS = _BATTERY_ENERGY_KWH * 3600.0

# Approximate 1C charge and 2C discharge power limits.
_MAX_CHARGE_KW = _BATTERY_ENERGY_KWH
_MAX_DISCHARGE_KW = 2.0 * _MAX_CHARGE_KW

_SOC_MIN = 0.15
_SOC_MAX = 0.9
_SOC_TARGET = 0.5

# SOC control is intentionally smoothed.  The pack energy is large, so a
# strict one-horizon SOC correction would demand unrealistically large power.
_SOC_DEADBAND = 0.005
_SOC_CORRECTION_GAIN = 0.1


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _resistive_power(v):
    """
    Approximate road resistance power in kW.

    This uses only speed and documented vehicle parameters.  It does not
    include unknown EDU losses, driveline losses, accessory loads, or slope.
    """
    v = np.asarray(v, dtype=np.float64)
    rolling = _CRR * _MASS * _G * v
    drag = 0.5 * _RHO * _CD * _FRONT_AREA * v * v * v
    return rolling + drag


def _electrical_demand(p_road, v):
    """
    Convert wheel/mechanical power to an approximate electrical demand.

    Positive p_road means traction demand.  Negative p_road means regen.
    The efficiency is a coarse approximation of the combined EDU boundary;
    it is not a rated efficiency and does not add inverter/gearbox losses a
    second time.
    """
    p_road = np.asarray(p_road, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    out = np.zeros_like(p_road, dtype=np.float64)

    pos = p_road > 1e-6
    if np.any(pos):
        p_norm = np.clip(p_road[pos] / 50.0, 0.0, 1.0)
        v_norm = np.clip(v[pos] / 20.0, 0.0, 1.0)

        # Coarse total EDU efficiency approximation.
        eta = 0.78 + 0.10 * p_norm + 0.04 * v_norm
        eta = np.clip(eta, 0.75, 0.92)

        out[pos] = p_road[pos] / eta

    neg = p_road < -1e-6
    if np.any(neg):
        # Coarse regen efficiency approximation.
        out[neg] = p_road[neg] * 0.85

    return out


def _signal_is_green_at_time(state, remaining, t, no_signal):
    """
    Return True if the next traffic light would be green at predicted time t.

    Only the current state, current remaining phase time, and the fixed cycle
    are used.  t is limited to the rule horizon.  This does not extrapolate
    beyond the horizon.
    """
    state = np.clip(np.round(state).astype(np.int64), 0, 2)
    remaining = np.clip(remaining, 0.0, 100.0)
    t = np.clip(t, 0.0, _H + 1.0)

    cur = state.copy()
    rem = remaining.copy()
    tt = t.copy()

    # The horizon is 12 s, so a few phase advances are enough.
    for _ in range(6):
        mask = tt >= rem
        if not np.any(mask):
            break

        tt = np.where(mask, tt - rem, tt)
        cur = np.where(mask, _SIGNAL_NEXT[cur], cur)
        rem = np.where(mask, _SIGNAL_DUR[cur], rem)

    green = cur == 1
    return np.where(no_signal, True, green)


# -----------------------------------------------------------------------------
# Driving prior
# -----------------------------------------------------------------------------

def _drive_prior(obs):
    """
    Physical driving prior: requested acceleration in [-1.5, 0.7] m/s^2.

    The rule evaluates a small deterministic candidate grid over a finite
    horizon and selects the candidate with the lowest approximate cost.
    """
    x = obs
    B = x.shape[0]

    if B == 0:
        return np.empty((0, 1), dtype=np.float32)

    C = _ACTIONS.shape[0]

    # Raw observation extraction.
    a_cur = np.clip(x[:, 4], -1.5, 0.7).astype(np.float64)
    v0 = np.clip(x[:, 5], 0.0, _SPEED_CAP).astype(np.float64)
    a_lead = np.clip(x[:, 6], -5.0, 5.0).astype(np.float64)
    v_lead = np.clip(x[:, 7], 0.0, 30.0).astype(np.float64)
    gap0 = np.clip(x[:, 8], 0.0, 30000.0).astype(np.float64)

    signal = np.clip(np.round(x[:, 9]), 0, 2).astype(np.float64)
    stop0 = np.clip(x[:, 10], -30000.0, 30000.0).astype(np.float64)
    phase0 = np.clip(x[:, 11], 0.0, 100.0).astype(np.float64)

    limit0 = np.clip(x[:, 12], 0.1, 30.0).astype(np.float64)
    limit1 = np.clip(x[:, 13], 0.1, 30.0).astype(np.float64)
    entry0 = np.clip(x[:, 14], 0.0, 30000.0).astype(np.float64)

    b_soh = np.clip(x[:, 1], 0.0, 1.0).astype(np.float64)
    fcs_soh = np.clip(x[:, 2], 0.0, 1.0).astype(np.float64)
    edu_soh = np.clip(x[:, 3], 0.0, 1.0).astype(np.float64)

    # Broadcast observation fields to candidate dimension.
    v0b = v0[:, None]
    a_leadb = a_lead[:, None]
    v_leadb = v_lead[:, None]
    gap0b = gap0[:, None]
    stop0b = stop0[:, None]
    limit0b = limit0[:, None]
    limit1b = limit1[:, None]
    entry0b = entry0[:, None]
    signalb = signal[:, None]
    phase0b = phase0[:, None]
    a_cur_b = a_cur[:, None]

    b_soh_b = b_soh[:, None]
    fcs_soh_b = fcs_soh[:, None]
    edu_soh_b = edu_soh[:, None]

    a_grid = _ACTIONS.reshape(1, C).astype(np.float64)

    # Candidate simulation state.
    ego_v = np.broadcast_to(v0b, (B, C)).copy()
    lead_v = np.broadcast_to(v_leadb, (B, C)).copy()
    ego_pos = np.zeros((B, C), dtype=np.float64)
    lead_pos = np.zeros((B, C), dtype=np.float64)

    # Used for the conservative one-step-earlier speed-region check.
    prev_v = ego_v.copy()

    no_signal = stop0b >= _NO_SIGNAL_DIST
    crossed = no_signal | (stop0b < -1e-3)

    valid_entry = (entry0b < _NO_SIGNAL_DIST) & (np.abs(limit1b - limit0b) > 1e-3)
    lower_entry = valid_entry & (limit1b < limit0b - 1e-3)
    entry_done = (~valid_entry) | (entry0b <= 1e-3)

    gap_valid = gap0b < _NO_SIGNAL_DIST

    total_cost = np.zeros((B, C), dtype=np.float64)
    invalid_cross = np.zeros((B, C), dtype=bool)

    for step in range(1, _H + 1):
        ego_v_prev = ego_v
        lead_v_prev = lead_v
        prev_prev_v = prev_v

        # Conservative within-step motion approximation.
        ego_v_step = np.maximum(0.0, ego_v_prev + 0.5 * a_grid * _DT)
        ego_v_step = np.minimum(ego_v_step, _SPEED_CAP)

        lead_v_step = np.maximum(0.0, lead_v_prev + 0.5 * a_leadb * _DT)
        lead_v_step = np.minimum(lead_v_step, 30.0)

        ego_pos_after = ego_pos + ego_v_step * _DT
        lead_pos_after = lead_pos + lead_v_step * _DT

        # End-of-step speeds.  The environment enforces no reverse and the
        # hard 70 km/h cap.  The rule approximates those effects.
        ego_v_after = np.maximum(0.0, ego_v_prev + a_grid * _DT)
        ego_v_after = np.minimum(ego_v_after, _SPEED_CAP)

        lead_v_after = np.maximum(0.0, lead_v_prev + a_leadb * _DT)
        lead_v_after = np.minimum(lead_v_after, 30.0)

        gap_after = gap0b + lead_pos_after - ego_pos_after
        gap_after = np.clip(gap_after, 0.0, 30000.0)

        # -----------------------------------------------------------------
        # Speed-region entry.  If a lower-limit region is crossed, check the
        # nominal entry speed and the conservative one-step-earlier speed.
        # -----------------------------------------------------------------
        entry_cross = (
            (~entry_done)
            & valid_entry
            & (ego_pos < entry0b - 1e-3)
            & (ego_pos_after >= entry0b - 1e-3)
        )

        entry_speed = np.maximum(prev_prev_v, ego_v_prev)

        entry_cost = np.where(
            entry_cross & lower_entry,
            500.0 * np.maximum(0.0, (entry_speed - limit1b) / np.maximum(limit1b, 1.0)),
            0.0,
        )
        entry_cost = np.where(
            entry_cross & lower_entry & (entry_speed >= 1.2 * limit1b),
            entry_cost + 1.0e5,
            entry_cost,
        )

        entry_done_after = entry_done | entry_cross
        limit_eff = np.where(entry_done_after, limit1b, limit0b)

        # -----------------------------------------------------------------
        # Posted speed-limit cost.  The hard 70 km/h cap is already applied
        # to ego_v_after.  The 120% termination threshold is penalized.
        # -----------------------------------------------------------------
        speed_excess = np.maximum(0.0, ego_v_after - limit_eff)
        speed_cost = 100.0 * speed_excess / np.maximum(limit_eff, 1.0)
        speed_cost = np.where(
            ego_v_after >= 1.2 * limit_eff,
            speed_cost + 1.0e5,
            speed_cost,
        )

        # -----------------------------------------------------------------
        # Following cost.  Primary objective: keep gap between D_safe and
        # D_max whenever feasible.  Too-small gaps and overlap are heavily
        # penalized.  Excessive gaps are penalized to encourage catch-up.
        # -----------------------------------------------------------------
        v_for_dist = ego_v_after

        D_safe = 2.0 + 0.5 * v_for_dist + 0.0625 * v_for_dist * v_for_dist
        D_max = 10.0 + v_for_dist + 0.0825 * v_for_dist * v_for_dist

        gap_cost = np.zeros((B, C), dtype=np.float64)

        collision = gap_valid & (gap_after <= 1e-3)
        gap_cost = np.where(collision, 1.0e6, gap_cost)

        below = gap_valid & (gap_after > 1e-3) & (gap_after < D_safe)
        gap_cost = np.where(
            below,
            gap_cost + 1000.0 * (D_safe - gap_after) / np.maximum(D_safe, 1.0),
            gap_cost,
        )

        above = gap_valid & (gap_after > D_max)
        gap_cost = np.where(
            above,
            gap_cost
            + 5.0 * np.minimum(100.0, gap_after - D_max)
            + 20.0 * np.minimum(100.0, np.maximum(0.0, gap_after - 200.0)),
            gap_cost,
        )

        # -----------------------------------------------------------------
        # Traffic-light arrival penalty.  If the next stop line would be
        # reached within the horizon before the current phase ends and the
        # current phase is not green, add a small penalty.  The hard green-only
        # crossing check is applied separately below.
        # -----------------------------------------------------------------
        t1 = np.where(
            ego_v_after > 0.1,
            stop0b / np.maximum(ego_v_after, 0.1),
            1.0e9,
        )

        bad_arrival = (
            (~no_signal)
            & (~crossed)
            & (ego_v_after > 0.1)
            & (t1 <= _H)
            & (t1 < phase0b)
            & (signalb != 1)
        )

        traffic_cost = np.where(bad_arrival, 5.0, 0.0)

        # -----------------------------------------------------------------
        # Hard green-only crossing check.
        #
        # For any candidate predicted to reach the stop line within the
        # horizon, the nominal crossing time and a conservative one-step-
        # earlier crossing time are evaluated.  If either is not green, the
        # candidate receives a large penalty.
        # -----------------------------------------------------------------
        rem_before = stop0b - ego_pos
        moved_forward = ego_pos_after > ego_pos + 1e-3

        crossed_now = (
            (~crossed)
            & (~no_signal)
            & (rem_before <= 1e-3)
            & (moved_forward | (ego_v_prev > 1e-3))
        )

        crossed_during = (
            (~crossed)
            & (~crossed_now)
            & (~no_signal)
            & (rem_before > 1e-3)
            & (ego_pos_after >= stop0b - 1e-3)
        )

        crossed_mask = crossed_now | crossed_during

        if np.any(crossed_mask):
            t_cross = np.where(
                crossed_now,
                float(step - 1),
                (step - 1) + rem_before / np.maximum(ego_v_step, 1e-3),
            )
            t_cross = np.clip(t_cross, 0.0, float(_H))

            # Conservative earlier-arrival case: one simulation step earlier.
            t_early = np.maximum(0.0, t_cross - 1.0)

            green_nom = _signal_is_green_at_time(
                signalb, phase0b, t_cross, no_signal
            )
            green_early = _signal_is_green_at_time(
                signalb, phase0b, t_early, no_signal
            )

            invalid_cross = invalid_cross | (
                crossed_mask & (~green_nom | ~green_early)
            )

        crossed = crossed | crossed_mask

        # -----------------------------------------------------------------
        # Comfort and component-health regularization.
        #
        # These terms are small compared with following, speed, and signal
        # costs.  They gently discourage abrupt acceleration and high positive
        # acceleration when component SOH is already low.
        # -----------------------------------------------------------------
        comfort_cost = 0.2 * np.abs(a_grid - a_cur_b)

        health_cost = (
            0.15 * (1.0 - b_soh_b)
            + 0.15 * (1.0 - fcs_soh_b)
            + 0.30 * (1.0 - edu_soh_b)
        ) * np.maximum(0.0, a_grid)

        total_cost += (
            gap_cost
            + speed_cost
            + entry_cost
            + traffic_cost
            + comfort_cost
            + health_cost
        )

        # Advance candidate simulation state.
        ego_pos = ego_pos_after
        lead_pos = lead_pos_after
        ego_v = ego_v_after
        lead_v = lead_v_after
        prev_v = ego_v_prev
        entry_done = entry_done_after

    # Penalize candidates that would cross a red or yellow stop line.
    total_cost = np.where(invalid_cross, total_cost + 1.0e6, total_cost)

    # Small tie-breakers.  They do not change the main objective hierarchy.
    total_cost += 0.01 * np.abs(a_grid)
    total_cost += 0.001 * np.abs(a_grid - a_cur_b)

    total_cost = np.nan_to_num(
        total_cost,
        nan=1.0e9,
        posinf=1.0e9,
        neginf=1.0e9,
    )

    idx = np.argmin(total_cost, axis=1)
    a_out = _ACTIONS[idx].astype(np.float64)

    # ---------------------------------------------------------------------
    # Post-processing safety filters.
    #
    # These filters do not bypass the environment.  They only keep the rule
    # request consistent with the posted limit, hard speed cap, and minimum
    # safe gap.  Negative requests are not forced to zero at low speed; the
    # environment handles the no-reverse clipping.
    # ---------------------------------------------------------------------

    # Do not request positive acceleration at the hard vehicle cap.
    a_out = np.where(
        v0 >= _SPEED_CAP - 0.05,
        np.minimum(a_out, 0.0),
        a_out,
    )

    # Do not request positive acceleration when already at or above the
    # current posted limit.
    a_out = np.where(
        v0 >= limit0 - 0.05,
        np.minimum(a_out, 0.0),
        a_out,
    )

    # If already above the 120% termination threshold region, request braking.
    a_out = np.where(
        v0 >= 1.1 * limit0,
        np.minimum(a_out, -0.25),
        a_out,
    )

    # Protect the minimum safe gap.
    D_safe_cur = 2.0 + 0.5 * v0 + 0.0625 * v0 * v0
    too_close = (gap0 < _NO_SIGNAL_DIST) & (gap0 < D_safe_cur)

    a_out = np.where(
        too_close & (a_out > 0.0),
        np.minimum(a_out, -0.25),
        a_out,
    )

    very_close = too_close & (gap0 < 0.5 * D_safe_cur)
    a_out = np.where(
        very_close & (a_out > -0.5),
        np.minimum(a_out, -0.5),
        a_out,
    )

    a_out = np.clip(a_out, -1.5, 0.7).astype(np.float32)
    return a_out.reshape(-1, 1)


# -----------------------------------------------------------------------------
# Energy prior
# -----------------------------------------------------------------------------

def _energy_prior(obs, driving_action):
    """
    Physical energy prior: requested FCS output power in [0, 50] kW.

    This is not battery power.  It is a requested FCS output power.  The
    environment may clip or modify the request due to battery, EDU, and FCS
    feasibility limits.
    """
    x = obs
    B = x.shape[0]

    if B == 0:
        return np.empty((0, 1), dtype=np.float32)

    soc = np.clip(x[:, 0], _SOC_MIN, _SOC_MAX).astype(np.float64)
    b_soh = np.clip(x[:, 1], 0.0, 1.0).astype(np.float64)
    fcs_soh = np.clip(x[:, 2], 0.0, 1.0).astype(np.float64)
    edu_soh = np.clip(x[:, 3], 0.0, 1.0).astype(np.float64)

    a_cur = np.clip(x[:, 4], -1.5, 0.7).astype(np.float64)
    v = np.clip(x[:, 5], 0.0, _SPEED_CAP).astype(np.float64)

    a_rule = np.clip(driving_action.reshape(-1), -1.5, 0.7).astype(np.float64)

    # ---------------------------------------------------------------------
    # Estimate future electrical demand using the driving prior trajectory.
    #
    # The rule does not have direct access to motor torque, instantaneous EDU
    # loss, battery power, hydrogen consumption, or actual FCS power.  The
    # following is an approximation based only on speed, acceleration, and
    # documented vehicle parameters.
    # ---------------------------------------------------------------------
    pred_v = v.copy()
    p_load_sum = np.zeros(B, dtype=np.float64)

    for _ in range(_H):
        v_prev = pred_v

        # Approximate environment no-reverse clipping.
        a_exec = np.maximum(a_rule, -v_prev / _DT)

        # Do not request positive traction power at the hard speed cap.
        a_exec = np.where(
            (v_prev >= _SPEED_CAP - 1e-3) & (a_exec > 0.0),
            0.0,
            a_exec,
        )

        p_road = _resistive_power(v_prev) + _M_EFF * a_exec * v_prev
        p_load_sum += _electrical_demand(p_road, v_prev)

        v_next = v_prev + a_exec * _DT
        pred_v = np.maximum(0.0, np.minimum(_SPEED_CAP, v_next))

    p_load_pred = p_load_sum / _H

    # Current demand is weighted with the predicted demand so the energy prior
    # reacts to the present step while still using the prediction horizon.
    p_load_cur = _electrical_demand(
        _resistive_power(v) + _M_EFF * a_cur * v,
        v,
    )
    p_load_ref = 0.35 * p_load_cur + 0.65 * p_load_pred

    # ---------------------------------------------------------------------
    # SOH-based conservative limits.
    #
    # These are not exact degradation models.  They are simple health caps:
    # low FCS SOH reduces the maximum FCS request; low battery SOH reduces
    # allowed battery power; low EDU SOH applies only a mild cap because the
    # energy actor cannot directly change EDU mechanical demand.
    # ---------------------------------------------------------------------
    b_scale = np.clip(0.5 + 0.5 * b_soh, 0.5, 1.0)
    f_scale = np.clip(0.5 + 0.5 * fcs_soh, 0.5, 1.0)
    edu_scale = np.clip(0.85 + 0.15 * edu_soh, 0.85, 1.0)

    max_fcs = np.clip(50.0 * f_scale * edu_scale, 0.0, 50.0)
    max_discharge = _MAX_DISCHARGE_KW * b_scale
    max_charge = _MAX_CHARGE_KW * b_scale

    # ---------------------------------------------------------------------
    # SOC regulation.
    #
    # Desired battery power over the horizon:
    #   P_batt_target = P_fcs - P_load
    # so:
    #   P_fcs_target = P_load + P_batt_target.
    #
    # A deadband and gain are used to avoid aggressive high-rate SOC
    # correction, because the pack energy is large and the rule is recomputed
    # every step.
    # ---------------------------------------------------------------------
    soc_err = _SOC_TARGET - soc
    soc_err = np.where(np.abs(soc_err) <= _SOC_DEADBAND, 0.0, soc_err)

    p_soc_target = soc_err * _BATTERY_ENERGY_KWS * _SOC_CORRECTION_GAIN / _H
    p_soc_target = np.clip(p_soc_target, -max_discharge, max_charge)

    p_req = p_load_ref + p_soc_target

    # Battery feasibility approximation.
    # If battery discharge would exceed the approximate limit, raise FCS request.
    p_req = np.maximum(p_req, p_load_ref - max_discharge)

    # If battery charge would exceed the approximate limit, lower FCS request.
    p_req = np.minimum(p_req, p_load_ref + max_charge)

    # FCS request cannot exceed the health-scaled maximum.
    p_req = np.minimum(p_req, max_fcs)

    # Physical request bounds.
    p_req = np.clip(p_req, 0.0, 50.0)

    # ---------------------------------------------------------------------
    # Very low FCS output avoidance.
    #
    # This is an approximation intended to reduce hydrogen use and possible
    # low-power/startup-related FCS degradation when the battery is healthy
    # and SOC is not dangerously low.  It is not a validated degradation model.
    # ---------------------------------------------------------------------
    low_power = (
        (p_req > 0.0)
        & (p_req < 5.0)
        & (soc > _SOC_MIN + 0.05)
        & (b_soh > 0.90)
    )
    p_req = np.where(low_power, 0.0, p_req)

    p_req = np.clip(p_req, 0.0, 50.0).astype(np.float32)
    return p_req.reshape(-1, 1)


# -----------------------------------------------------------------------------
# Required rule interface
# -----------------------------------------------------------------------------

def predict(observation):
    """
    FCHEV rule prior.

    Input:
        Raw unnormalized observation, shape (15,) or (batch, 15).

    Output:
        Physical action priors:
            driving.action: requested acceleration in [-1.5, 0.7] m/s^2
            energy.action:  requested FCS output power in [0, 50] kW

        active masks are always True for both actors.
    """
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15 or not np.all(np.isfinite(x)):
        raise ValueError(
            "observation must be finite with shape (15,) or (batch, 15)"
        )

    batch_size = x.shape[0]

    if batch_size == 0:
        empty_action = np.empty((0, 1), dtype=np.float32)
        empty_mask = np.empty((0, 1), dtype=bool)
        return {
            "driving": {"action": empty_action, "active": empty_mask},
            "energy": {"action": empty_action, "active": empty_mask},
        }

    driving_action = _drive_prior(x)
    energy_action = _energy_prior(x, driving_action)

    active = np.ones((batch_size, 1), dtype=bool)

    return {
        "driving": {
            "action": driving_action.astype(np.float32),
            "active": active,
        },
        "energy": {
            "action": energy_action.astype(np.float32),
            "active": active,
        },
    }