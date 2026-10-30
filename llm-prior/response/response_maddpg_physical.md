## A. Suggested program file

Create the file:

```text
llm-prior/programs/fchev_rule_prior.py
```

with the following complete module:

```python
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
```

---

## B. Behavior, fusion, and limitations

### 1. Interface and output mode

The module exposes the required function:

```python
predict(observation)
```

It accepts the complete raw 15-dimensional observation in the documented order. It supports both:

- single observation: shape `(15,)`
- batch observation: shape `(batch, 15)`

It validates that the observation is finite and has 15 columns. It returns physical action values:

```python
{
    "driving": {
        "action": requested_acceleration_m_s2,
        "active": True,
    },
    "energy": {
        "action": requested_fcs_power_kw,
        "active": True,
    },
}
```

The returned arrays have shape `(batch, 1)`. The masks are always `True`, so both actors are evaluated every step. The rule does not return `std`, does not add noise, does not call SUMO, does not access an online service, and does not change rewards or configuration.

---

### 2. Fusion with MADDPG actions

This program is a soft prior. It does not replace MADDPG and does not guarantee the executed action.

The adapter first maps the MADDPG normalized actions to physical units:

For the driving actor:

```text
u_d = tanh(z_d)

if u_d < 0:
    a_MADDPG = 1.5 * u_d        # [-1, 0] -> [-1.5, 0] m/s^2
else:
    a_MADDPG = 0.7 * u_d        # [0, 1] -> [0, 0.7] m/s^2
```

For the energy actor:

```text
u_e = tanh(z_e)
P_MADDPG = 25 * (u_e + 1)       # [0, 50] kW
```

The physical action prior returned by this program is blended externally:

```text
a_fused = (1 - lambda) * a_MADDPG,physical + lambda * a_rule,physical
```

and

```text
P_fused = (1 - lambda) * P_MADDPG,physical + lambda * P_rule,physical
```

`lambda` is supplied by the trainer or fusion adapter. It is not hard-coded in the rule.

Because both MADDPG physical actions and rule physical actions are bounded, the blended physical actions remain within the documented physical action ranges. The environment may still clip or modify them due to vehicle dynamics, no-reverse handling, speed caps, battery limits, EDU limits, and FCS feasibility limits.

---

### 3. Driving prior behavior

The driving prior is designed around the stated primary objective: follow the lead vehicle while keeping the gap between `D_safe` and `D_max`, obeying the posted speed limit and the green-only stop-line crossing rule.

#### Prediction horizon

The rule uses:

```text
H = 12 steps = 12 s
```

This lies within the required 5–20 step horizon.

The rule does not extrapolate signal phase beyond the horizon. It evaluates only candidate actions whose predicted consequences are assessed within 12 s. The rule is recomputed from the new observation at the next step.

#### Candidate acceleration grid

The driving prior evaluates 23 deterministic candidate accelerations:

```text
[-1.5, -1.4, -1.3, ..., 0.6, 0.7] m/s^2
```

For each candidate, it simulates a simplified 12-step longitudinal trajectory using the current ego speed, ego acceleration, lead speed, lead acceleration, gap, signal state, signal distance, signal phase remaining time, current speed limit, next speed limit, and distance to the next speed-region entrance.

The simulation is approximate. It uses a conservative within-step speed approximation and evaluates all predicted steps. This supports the required one-step conservative uncertainty margin.

---

### 4. Driving cost terms

The selected candidate is the one with the lowest approximate total cost.

#### Following cost

The rule computes the documented following reference distances:

```text
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

using the predicted post-action ego speed.

The cost hierarchy is:

1. Gap overlap or very small gap: very large penalty.
2. Gap below `D_safe`: large penalty.
3. Gap between `D_safe` and `D_max`: no following penalty.
4. Gap above `D_max`: increasing penalty to encourage catch-up.
5. Gap near or above 200 m: additional catch-up encouragement.

If the gap observation is sentinel-like, approximately 30000 m, the following cost is suppressed because that value is treated as no valid visible following target.

This is not an exact implementation of the MADDPG following reward. It is a rule prior that supports the same task priority.

---

#### Speed-limit cost

The rule penalizes predicted post-action speeds above the current effective posted limit.

The effective limit is updated when the candidate trajectory crosses a speed-region entrance. If the next region has a lower limit, the rule applies a stronger entry penalty. This mirrors the project’s speed-shaping objective before entering a 30 km/h intersection zone.

The rule also penalizes predicted speeds near or above 120% of the active posted limit, matching the termination threshold.

The hard 70 km/h cap is applied in the simulated post-action speed.

---

#### Traffic-light cost

The rule uses the current signal state, distance to the next stop line, and remaining time in the current phase.

It does not assume a zero speed limit for red or yellow. Red and yellow are treated only as crossing prohibitions.

For each candidate action, if the predicted trajectory reaches the stop line within the 12-step horizon, the rule evaluates:

- the nominal crossing time, and
- the conservative one-step-earlier crossing time.

If either crossing time corresponds to red or yellow, the candidate receives a large penalty. Only green crossing is allowed.

The conservative earlier-arrival check is important because the nominal prediction could be optimistic. If a candidate nominally crosses just after a phase change, the earlier-arrival case may still be non-green, and the candidate is rejected.

If the stop line is far enough that no crossing occurs within the 12-step horizon, the rule does not make unbounded future signal predictions. It reassesses on the next step.

---

#### Comfort and health regularization

The rule adds small penalties for:

- large changes from the current executed acceleration,
- positive acceleration when battery SOH, FCS SOH, or EDU SOH is reduced.

These terms are intentionally small. They should not override the primary following objective, speed-limit compliance, or green-only crossing requirement.

---

### 5. Driving post-processing filters

After candidate selection, the rule applies conservative filters:

- no positive acceleration request at the hard 70 km/h cap;
- no positive acceleration request when already at or above the current posted limit;
- stronger braking request when already near or above the 120% speed-limit termination region;
- no positive acceleration request when the current gap is below `D_safe`;
- stronger braking request when the current gap is very close to overlap.

These filters do not bypass the environment. They only keep the requested action consistent with the documented road and vehicle constraints.

The rule does not force negative acceleration requests to zero at low speed. If a braking request is selected at standstill, the environment handles the no-reverse clipping.

---

### 6. Energy prior behavior

The energy prior returns a requested FCS output power in `[0, 50] kW`. It is not battery power.

The energy rule estimates future electrical demand from speed and acceleration, then adjusts the requested FCS power to support SOC regulation and component health.

---

### 7. Energy demand estimation

The rule estimates mechanical road power as:

```text
P_road = P_rolling + P_drag + P_inertial
```

with:

```text
P_rolling = C_rr * mass * g * v
P_drag    = 0.5 * rho * C_d * A * v^3
P_inertial = m_eff * a * v
```

where:

```text
m_eff = mass * rotational-mass coefficient
```

This estimate uses only available observations and documented vehicle parameters.

The rule does not have direct access to:

- motor torque,
- instantaneous EDU loss,
- actual requested FCS power,
- battery power,
- hydrogen consumption,
- EDU map lookup,
- FCS hydrogen table,
- battery voltage/current state,
- actual powertrain feasibility clipping.

Therefore, the electrical demand is an approximation.

The approximate electrical demand is:

```text
P_load = P_road / eta_edu      if P_road > 0
P_load = P_road * eta_regen    if P_road < 0
```

The efficiency approximation is a coarse representation of the combined EDU boundary. It is not a rated efficiency and does not add inverter or gearbox losses a second time.

---

### 8. SOC regulation

The rule estimates battery pack energy from the documented cell configuration:

```text
106 series * 3.3 V nominal
134 parallel * 2.3 Ah nominal
```

The approximate pack energy is:

```text
approximately 107.8 kWh
```

The approximate 1C charge and 2C discharge power limits are derived from that energy estimate. These are approximations, not exact battery model outputs.

The SOC target is 0.5. The rule computes a smoothed SOC correction target:

```text
P_batt_target ≈ (0.5 - SOC) * battery_energy * correction_gain / H
```

with:

```text
correction_gain = 0.1
H = 12 s
deadband = 0.005 SOC
```

The desired FCS power is then approximately:

```text
P_fcs_target = P_load_ref + P_batt_target
```

where:

- if SOC is below target, the rule requests more FCS power than demand to charge the battery;
- if SOC is above target, the rule requests less FCS power than demand to discharge the battery;
- if SOC is within the deadband, the rule requests approximately the estimated demand.

This avoids aggressive one-horizon SOC correction because the battery pack energy is large and the rule is recomputed every step.

---

### 9. Component SOH handling

The energy rule applies conservative SOH-based caps.

#### Battery SOH

If battery SOH is reduced, the approximate battery charge and discharge limits are reduced. If the battery cannot safely carry enough discharge power, the rule raises the requested FCS power.

#### FCS SOH

If FCS SOH is reduced, the maximum requested FCS power is reduced. This is a conservative high-power protection heuristic, not an exact FCS aging model.

#### EDU SOH

The energy actor cannot directly reduce EDU mechanical demand, because EDU demand is primarily determined by the driving action. The rule therefore applies only a mild EDU-SOH-based cap on FCS request. The main EDU degradation limitation is discussed below.

---

### 10. Low-power FCS avoidance

The rule sets very small positive FCS requests to zero when:

- the requested FCS power is below 5 kW,
- SOC is not dangerously low,
- battery SOH is high.

This is an approximate heuristic intended to reduce hydrogen use and possible low-power/startup-related FCS degradation. It is not based on the exact FCS aging model.

---

### 11. Relationship to the stated rewards

The rule does not implement the MADDPG reward equations term by term. It uses the reward structure only as context for priorities.

#### Driving rewards

The driving prior supports:

- following within `D_safe` and `D_max`;
- avoiding gaps below `D_safe`;
- closing excessive gaps when feasible;
- obeying posted speed limits;
- avoiding the 120% speed termination region;
- crossing stop lines only on green;
- approaching lower-limit regions with reduced entry speed;
- modest comfort behavior.

It does not attempt to reproduce the exact following reward piecewise formula, the exact traffic-light reward, the exact speed-shaping term, or the exact terminal penalties.

#### Energy rewards

The energy prior supports:

- reducing FCS power when not needed;
- avoiding excessive FCS output;
- keeping SOC near 0.5;
- reducing battery power when battery SOH is low;
- reducing FCS power when FCS SOH is low;
- applying a mild conservative cap when EDU SOH is low.

It does not directly compute hydrogen consumption, FCS degradation increments, battery SOH loss, or EDU CLR increments because those quantities are not available in the observation.

---

### 12. Masks

The output masks are always:

```python
active = True
```

for both actors.

This means:

- the driving prior is available every step;
- the energy prior is available every step;
- the rule is not used to skip fusion;
- the rule is not used to declare itself inactive.

The internal decision logic may choose zero acceleration or zero FCS power, but the mask remains active.

---

### 13. Physical constraints and environment clipping

The rule outputs requests, not executed states.

The environment may still modify the executed action because of:

- no-reverse clipping,
- hard 70 km/h ego speed cap,
- battery current/voltage/SOC limits,
- battery charge/discharge rate limits,
- EDU torque-speed-validity limits,
- EDU efficiency-map validity,
- FCS output limits,
- DC/DC and dispatch feasibility,
- actual powertrain feasibility.

The rule cannot bypass these constraints. It only produces a physical request within the documented action ranges.

---

### 14. Main limitations

1. **The rule is approximate.**  
   It uses simplified longitudinal prediction and approximate power estimation. It is not a full FCHEV simulation.

2. **The energy demand estimate is indirect.**  
   The observation does not include motor torque, EDU loss, battery power, hydrogen rate, or actual FCS power. The rule estimates these from speed, acceleration, and vehicle parameters.

3. **EDU degradation is only indirectly addressed.**  
   EDU degradation is mainly driven by EDU mechanical operating point. The energy actor cannot directly change that operating point. The rule can only apply a mild conservative cap and can support lower acceleration through the driving prior.

4. **FCS hydrogen and degradation are not computed from the exact tables.**  
   The exact FCS hydrogen-consumption table and aging model are not available to the rule. The rule uses conservative caps and power heuristics.

5. **Battery pack energy and rate limits are approximate.**  
   The rule uses nominal cell voltage and nominal cell capacity to estimate pack energy and 1C/2C power limits. Actual electrothermal behavior, voltage/current limits, and SOH-dependent limits are not modeled.

6. **The energy prior anticipates the driving prior, but fusion may change the executed driving action.**  
   If the final fused driving action differs from the rule driving action, the energy demand estimate may be imperfect.

7. **No hidden state is used.**  
   The rule does not remember the previous FCS request or previous driving action. Therefore, it cannot directly penalize FCS load-change degradation across steps. It uses current observations only.

8. **No validation is claimed.**  
   This is a static rule design. It has not been executed or validated in SUMO here.

---

### 15. CLI selection

Select this program with:

```bash
--prior-program-name fchev_rule_prior
```

Use physical-action output mode:

```bash
--maddpg-rule-output-mode physical_action
```

Select the fusion strength externally, for example:

```bash
--maddpg-rule-lambda 0.25
```

where:

```text
lambda = 0   -> rule prior disabled in fusion
lambda = 1   -> rule prior fully used in fusion
0 < lambda < 1 -> soft fusion
```

The first 1000 steps of a fresh run remain prior-free random warm-up if that is the trainer behavior. Evaluation does not add exploration noise.