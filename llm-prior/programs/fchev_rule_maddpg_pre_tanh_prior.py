"""
Rule-prior program for the FCHEV/MADDPG longitudinal driving and energy-management task.

This program returns deterministic pre-tanh latent priors for the two MADDPG actors:
    driving.prior : latent z_rule for the driving actor
    energy.prior  : latent z_rule for the energy actor

The rule is a soft prior only. It does not replace the environment, does not bypass
physical constraints, and does not guarantee the executed action.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Observation indices, raw unnormalized observation
# -----------------------------------------------------------------------------
_OBS_DIM = 15

_IDX_SOC = 0
_IDX_BATTERY_SOH = 1
_IDX_FCS_SOH = 2
_IDX_EDU_SOH = 3
_IDX_A_EGO = 4
_IDX_V_EGO = 5
_IDX_A_LEAD = 6
_IDX_V_LEAD = 7
_IDX_GAP = 8
_IDX_TL_STATE = 9
_IDX_TL_DIST = 10
_IDX_TL_TIME = 11
_IDX_LIMIT = 12
_IDX_NEXT_LIMIT = 13
_IDX_NEXT_DIST = 14


# -----------------------------------------------------------------------------
# Environment-derived constants
# -----------------------------------------------------------------------------
_SENTINEL = 30000.0

_MAX_ACCEL = 0.7
_MAX_DECEL = 1.5

_APPROACH_ZONE = 100.0
_YELLOW_DURATION = 3.0

_GAP_VALID_MAX = 1000.0
_GAP_REF = 200.0

_SOC_TARGET = 0.5
_SOC_DEADBAND = 0.04

_LATENT_CLIP = 0.9999

_DEFAULT_MAIN_LIMIT = 16.6667


# -----------------------------------------------------------------------------
# Vehicle constants used only for approximate mechanical power estimation.
# These are not used to bypass dynamics; they are used only to form a soft prior.
# -----------------------------------------------------------------------------
_MASS = 7000.0
_ROT_MASS_COEFF = 1.05
_EFFECTIVE_MASS = _MASS * _ROT_MASS_COEFF

_GRAVITY = 9.8
_ROLL_RESIST = 0.0085

_AIR_DENSITY = 1.225
_FRONT_AREA = 5.8
_DRAG_COEFF = 0.55

_DRAG_COEFF_TOTAL = 0.5 * _AIR_DENSITY * _DRAG_COEFF * _FRONT_AREA
_ROLL_FORCE = _ROLL_RESIST * _MASS * _GRAVITY


# -----------------------------------------------------------------------------
# IDM-like following prior constants.
# These are heuristic parameters, not environment constraints.
# -----------------------------------------------------------------------------
_IDM_A_MAX = 0.7
_IDM_B_COMF = 1.5
_IDM_S0 = 5.0
_IDM_T = 0.5
_IDM_DELTA = 4


# -----------------------------------------------------------------------------
# Input validation
# -----------------------------------------------------------------------------
def _validate_observation(observation):
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        if x.shape[0] != _OBS_DIM:
            raise ValueError(
                "observation must be finite with shape (15,) or (batch, 15)"
            )
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != _OBS_DIM:
        raise ValueError(
            "observation must be finite with shape (15,) or (batch, 15)"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "observation must be finite with shape (15,) or (batch, 15)"
        )

    return x


# -----------------------------------------------------------------------------
# Latent conversions
# -----------------------------------------------------------------------------
def _accel_to_latent(a):
    """
    Convert desired acceleration to pre-tanh latent z_rule.

    Environment mapping:
        u_d < 0: a_req = 1.5 u_d
        u_d >= 0: a_req = 0.7 u_d

    Inverse:
        a < 0: u = a / 1.5
        a >= 0: u = a / 0.7

    z = atanh(clip(u, -0.9999, 0.9999))
    """
    a = np.asarray(a, dtype=np.float32)
    a = np.clip(a, -_MAX_DECEL, _MAX_ACCEL)

    u = np.where(a < 0.0, a / _MAX_DECEL, a / _MAX_ACCEL)
    u = np.clip(u, -_LATENT_CLIP, _LATENT_CLIP)

    return np.arctanh(u).astype(np.float32)


def _power_to_latent(p):
    """
    Convert desired FCS power request to pre-tanh latent z_rule.

    Environment mapping:
        u_e = tanh(z_e)
        P_req = 25 (u_e + 1) kW

    Inverse:
        u_e = 2 P_req / 50 - 1

    z = atanh(clip(u, -0.9999, 0.9999))
    """
    p = np.asarray(p, dtype=np.float32)
    p = np.clip(p, 0.0, 50.0)

    u = 2.0 * p / 50.0 - 1.0
    u = np.clip(u, -_LATENT_CLIP, _LATENT_CLIP)

    return np.arctanh(u).astype(np.float32)


# -----------------------------------------------------------------------------
# Driving prior
# -----------------------------------------------------------------------------
def _driving_prior(x):
    """
    Compute a soft driving acceleration prior and active mask.

    Uses raw observation values:
        4: ego acceleration
        5: ego speed
        6: lead acceleration
        7: lead speed
        8: gap
        9: traffic-light state
        10: distance to traffic-light stop line
        11: remaining phase time
        12: current speed limit
        13: next speed limit
        14: distance to next speed region
    """
    a_cur = x[:, _IDX_A_EGO]
    v = x[:, _IDX_V_EGO]
    a_lead = x[:, _IDX_A_LEAD]
    v_lead = x[:, _IDX_V_LEAD]
    gap = x[:, _IDX_GAP]

    tl_state = x[:, _IDX_TL_STATE]
    tl_dist = x[:, _IDX_TL_DIST]
    tl_time = x[:, _IDX_TL_TIME]

    limit = x[:, _IDX_LIMIT]
    next_limit = x[:, _IDX_NEXT_LIMIT]
    next_dist = x[:, _IDX_NEXT_DIST]

    # Safety fallback for malformed but finite speed-limit observations.
    limit_safe = np.asarray(
        np.where(limit > 0.0, limit, _DEFAULT_MAIN_LIMIT),
        dtype=np.float32,
    )
    next_limit_safe = np.asarray(
        np.where(next_limit > 0.0, next_limit, limit_safe),
        dtype=np.float32,
    )

    # ---------------------------------------------------------------------
    # 1. Speed-region target
    # ---------------------------------------------------------------------
    has_next = (
        (next_dist < _SENTINEL - 1.0)
        & (next_dist >= 0.0)
        & (next_limit > 0.0)
    )
    lower_next = has_next & (next_limit_safe < limit_safe - 1e-3)

    speed_target = limit_safe.copy()
    speed_target = np.asarray(
        np.where(
            lower_next,
            np.minimum(limit_safe, next_limit_safe),
            speed_target,
        ),
        dtype=np.float32,
    )

    # IDM target speed is slightly above the posted limit so that the vehicle
    # can remain near the posted limit instead of braking unnecessarily.
    v0 = np.maximum(speed_target + 0.5, 0.1)

    # ---------------------------------------------------------------------
    # 2. Following prior
    # ---------------------------------------------------------------------
    # Treat very large values as sentinel/invalid for short-horizon following.
    # Negative small gaps are still treated as unsafe.
    gap_valid = (gap < _GAP_VALID_MAX) & (gap > -1000.0)

    D_safe = 2.0 + 0.5 * v + 0.0625 * v * v
    D_max = 10.0 + v + 0.0825 * v * v

    gap_for_idm = np.where(gap_valid, np.maximum(gap, 1.0), 10000.0)

    dv = v - v_lead
    sqrt_ab = np.sqrt(_IDM_A_MAX * _IDM_B_COMF)

    s_star = _IDM_S0 + v * _IDM_T + v * dv / (2.0 * sqrt_ab)
    s_star = np.maximum(s_star, 0.1)

    ratio = np.clip(v / v0, 0.0, 1.5)

    a_idm = _IDM_A_MAX * (
        1.0
        - ratio ** _IDM_DELTA
        - (s_star / gap_for_idm) ** 2
    )
    a_idm = np.clip(a_idm, -_MAX_DECEL, _MAX_ACCEL)

    a_speed = np.clip(speed_target - v, -_MAX_DECEL, _MAX_ACCEL)

    a_follow = np.where(gap_valid, a_idm, a_speed)

    # Emergency deceleration if the gap is already unsafe.
    gap_unsafe = gap_valid & (gap <= D_safe)
    emergency_follow = gap_unsafe & (v > 0.05)

    closing_speed = np.maximum(0.0, v - v_lead)
    gap_abs = np.maximum(gap, 1.0)

    a_emergency = (
        -(closing_speed * closing_speed) / (2.0 * gap_abs)
        - np.maximum(0.0, -a_lead)
    )
    a_emergency = np.clip(a_emergency, -_MAX_DECEL, 0.0)

    hard_brake = (
        gap_unsafe
        & (gap < 0.5 * D_safe)
        & (v > 0.05)
    )
    a_emergency = np.where(hard_brake, -_MAX_DECEL, a_emergency)

    a_follow = np.where(
        emergency_follow,
        np.minimum(a_follow, a_emergency),
        a_follow,
    )

    # ---------------------------------------------------------------------
    # 3. Traffic-light prior
    # ---------------------------------------------------------------------
    eps = 1e-3

    # A distance of 0 m is valid. Sentinel 30000 m means no visible target.
    has_tl = (tl_dist < _SENTINEL - 1.0) & (tl_dist >= 0.0)

    # Signal phase is used only from entry into the 100 m approach zone.
    in_approach = has_tl & (tl_dist <= _APPROACH_ZONE)

    arrival_time = np.where(
        v > eps,
        tl_dist / np.maximum(v, eps),
        np.inf,
    )

    phase_time = tl_time

    red = tl_state == 0.0
    yellow = tl_state == 2.0
    green = tl_state == 1.0

    # Red: stop if arrival is likely during the red phase.
    red_stop = (
        in_approach
        & red
        & (v > eps)
        & (arrival_time < np.maximum(phase_time, 0.5))
    )
    red_stop = red_stop | (
        in_approach
        & red
        & (v > eps)
        & (tl_dist <= 5.0)
    )

    # Yellow: stop if arrival is likely after yellow ends, i.e. likely red.
    # Crossing yellow itself does not trigger red-light termination, so the rule
    # only stops when the arrival time is clearly outside the yellow window.
    yellow_stop = (
        in_approach
        & yellow
        & (v > eps)
        & (arrival_time > phase_time + 0.5)
    )

    # Green: stop if arrival is likely after green + yellow, i.e. likely red.
    green_stop = (
        in_approach
        & green
        & (v > eps)
        & (arrival_time > phase_time + _YELLOW_DURATION + 0.5)
    )

    need_stop = red_stop | yellow_stop | green_stop

    dist_stop = np.maximum(tl_dist, eps)
    a_stop = np.where(
        v > eps,
        -(v * v) / (2.0 * dist_stop),
        0.0,
    )
    a_stop = np.clip(a_stop, -_MAX_DECEL, 0.0)
    a_stop = np.where(
        (tl_dist <= 0.0) & (v > eps),
        -_MAX_DECEL,
        a_stop,
    )

    a_follow = np.where(
        need_stop,
        np.minimum(a_follow, a_stop),
        a_follow,
    )

    # ---------------------------------------------------------------------
    # 4. Lower-limit region entry shaping
    # ---------------------------------------------------------------------
    dist_next = np.maximum(next_dist, eps)

    a_entry = (
        next_limit_safe * next_limit_safe
        - v * v
    ) / (2.0 * dist_next)
    a_entry = np.clip(a_entry, -_MAX_DECEL, _MAX_ACCEL)

    # Uses the same assumed braking magnitude as the reward shaping term:
    # v_entry = sqrt(max(0, v^2 - 2 * 0.75 * d_entry))
    v_entry_assumed = np.sqrt(
        np.maximum(
            0.0,
            v * v - 2.0 * 0.75 * np.maximum(next_dist, 0.0),
        )
    )

    entry_over = (
        lower_next
        & (next_dist < 1000.0)
        & (v_entry_assumed > next_limit_safe + 0.2)
    )

    a_follow = np.where(
        entry_over,
        np.minimum(a_follow, a_entry),
        a_follow,
    )

    # ---------------------------------------------------------------------
    # 5. Current speed-limit enforcement
    # ---------------------------------------------------------------------
    overspeed = (limit_safe > 0.0) & (v > limit_safe + 0.05)
    a_limit = np.clip(limit_safe - v, -_MAX_DECEL, _MAX_ACCEL)

    a_follow = np.where(
        overspeed,
        np.minimum(a_follow, a_limit),
        a_follow,
    )

    # Reverse motion is prohibited. If the vehicle is effectively stopped,
    # do not create a negative prior.
    a_follow = np.where(
        (v <= 0.05) & (a_follow < 0.0),
        0.0,
        a_follow,
    )

    # ---------------------------------------------------------------------
    # 6. Comfort smoothing
    # ---------------------------------------------------------------------
    stopped_close = (
        (v <= 0.05)
        & gap_valid
        & (gap <= 0.5 * D_safe)
        & (v_lead <= 0.05)
    )

    safety_required = (
        emergency_follow
        | need_stop
        | overspeed
        | entry_over
        | (gap_valid & (gap <= 0.0))
        | stopped_close
    )

    # In non-safety cases, limit acceleration changes to reduce jerk.
    max_change = np.where(
        safety_required,
        _MAX_ACCEL + _MAX_DECEL,
        0.5,
    )

    a_final = np.clip(
        a_follow,
        a_cur - max_change,
        a_cur + max_change,
    )
    a_final = np.clip(a_final, -_MAX_DECEL, _MAX_ACCEL)

    # Again enforce no reverse at zero speed.
    a_final = np.where(
        (v <= 0.05) & (a_final < 0.0),
        0.0,
        a_final,
    )

    # ---------------------------------------------------------------------
    # 7. Driving active mask
    # ---------------------------------------------------------------------
    prior = _accel_to_latent(a_final)
    prior_abs = np.abs(prior)

    follow_active = gap_valid & (
        (gap < D_safe)
        | (gap > D_max)
        | (np.abs(v - v_lead) > 1.0)
    )

    traffic_active = in_approach & (
        (red | yellow | green)
        & ((v > eps) | need_stop)
    )

    # Hold zero if stopped at a red light.
    red_hold = in_approach & red & (v <= eps)

    speed_active = (
        overspeed
        | entry_over
        | ((speed_target - v) > 0.5)
    )

    active = (
        safety_required
        | red_hold
        | (need_stop & (v > eps))
    )

    # Activate non-safety priors only when the prior is not numerically neutral.
    active = active | (
        (follow_active | speed_active | traffic_active)
        & (prior_abs > 1e-3)
    )

    prior = np.where(active, prior, 0.0).astype(np.float32)

    return prior.reshape(-1, 1), active.reshape(-1, 1)


# -----------------------------------------------------------------------------
# Energy prior
# -----------------------------------------------------------------------------
def _energy_prior(x):
    """
    Compute a soft FCS power-request prior and active mask.

    The rule does not directly observe:
        - EDU torque
        - EDU loss
        - battery power
        - hydrogen consumption
        - actual FCS output power

    It estimates only a mechanical drive demand from ego speed and ego acceleration.
    This estimate is approximate and may be clipped by the environment.
    """
    soc = x[:, _IDX_SOC]
    battery_soh = x[:, _IDX_BATTERY_SOH]
    fcs_soh = x[:, _IDX_FCS_SOH]
    edu_soh = x[:, _IDX_EDU_SOH]

    v = x[:, _IDX_V_EGO]
    a = x[:, _IDX_A_EGO]

    # ---------------------------------------------------------------------
    # Approximate mechanical power demand at the drive-unit boundary.
    # This is not exact EDU electrical power and does not include EDU losses.
    # ---------------------------------------------------------------------
    f_resist = _ROLL_FORCE + _DRAG_COEFF_TOTAL * v * v
    f_drive = _EFFECTIVE_MASS * a + f_resist

    p_mech_kw = f_drive * v / 1000.0
    p_mech_kw = np.where(v > 0.0, p_mech_kw, 0.0)

    p_base = np.clip(p_mech_kw, 0.0, 50.0)

    # ---------------------------------------------------------------------
    # SOC-target adjustment.
    # SOC > 0.5: request less FCS power, discharge battery.
    # SOC < 0.5: request more FCS power, charge battery.
    # ---------------------------------------------------------------------
    soc_err = soc - _SOC_TARGET

    soc_adjust = np.zeros_like(soc, dtype=np.float32)

    over = soc_err > _SOC_DEADBAND
    under = soc_err < -_SOC_DEADBAND

    soc_adjust = np.where(
        over,
        -np.minimum(
            20.0,
            np.maximum(0.0, (soc_err - _SOC_DEADBAND) * 50.0),
        ),
        soc_adjust,
    )

    soc_adjust = np.where(
        under,
        np.minimum(
            20.0,
            np.maximum(0.0, (-soc_err - _SOC_DEADBAND) * 50.0),
        ),
        soc_adjust,
    )

    # If battery SOH is low, reduce aggressive battery buffering.
    battery_soh_factor = np.clip(battery_soh, 0.0, 1.0)
    soc_adjust = soc_adjust * (0.6 + 0.4 * battery_soh_factor)

    p_req = p_base + soc_adjust

    # ---------------------------------------------------------------------
    # Braking / regeneration handling.
    # If mechanical power is negative, prefer low FCS request unless SOC is low.
    # ---------------------------------------------------------------------
    braking = p_mech_kw < 0.0

    p_req = np.where(braking, np.maximum(0.0, p_req), p_req)
    p_req = np.where(
        braking & (soc_err > -_SOC_DEADBAND),
        0.0,
        p_req,
    )
    p_req = np.where(
        braking & under,
        np.minimum(p_req, 10.0),
        p_req,
    )

    # ---------------------------------------------------------------------
    # Degradation-aware FCS caps.
    # These are heuristic caps, not exact degradation models.
    # ---------------------------------------------------------------------
    fcs_cap = np.full_like(soc, 50.0, dtype=np.float32)

    fcs_cap = np.where(
        fcs_soh < 0.95,
        np.minimum(fcs_cap, 45.0),
        fcs_cap,
    )
    fcs_cap = np.where(
        fcs_soh < 0.85,
        np.minimum(fcs_cap, 35.0),
        fcs_cap,
    )
    fcs_cap = np.where(
        fcs_soh < 0.65,
        np.minimum(fcs_cap, 25.0),
        fcs_cap,
    )

    # If SOC is high and demand is low, avoid unnecessary high FCS power.
    high_soc_low_demand = (soc_err > 0.1) & (p_base < 20.0)
    fcs_cap = np.where(
        high_soc_low_demand,
        np.minimum(fcs_cap, 20.0),
        fcs_cap,
    )

    # If battery SOH is low, bias toward supplying power closer to demand
    # to reduce battery charge/discharge extremes.
    low_battery = battery_soh < 0.9
    p_req = np.where(
        low_battery,
        0.5 * p_req + 0.5 * p_base,
        p_req,
    )

    # If EDU SOH is low, cap FCS request as a conservative proxy for reducing
    # high-power operating stress. This does not model EDU aging directly.
    fcs_cap = np.where(
        edu_soh < 0.8,
        np.minimum(fcs_cap, 40.0),
        fcs_cap,
    )

    p_req = np.clip(p_req, 0.0, fcs_cap)

    # ---------------------------------------------------------------------
    # Energy active mask
    # ---------------------------------------------------------------------
    prior = _power_to_latent(p_req)

    active_reason = (
        (np.abs(soc_err) > 0.05)
        | (p_mech_kw > 35.0)
        | (p_mech_kw < -10.0)
        | (fcs_soh < 0.95)
        | (battery_soh < 0.95)
        | (edu_soh < 0.95)
    )

    # If the FCS cap is binding, activate even if the latent prior is numerically
    # neutral, because the rule is intentionally limiting the actor.
    cap_binding = (fcs_cap < 50.0) & (p_req >= fcs_cap - 1e-3)

    active = (
        active_reason & (np.abs(prior) > 1e-3)
    ) | cap_binding

    prior = np.where(active, prior, 0.0).astype(np.float32)

    return prior.reshape(-1, 1), active.reshape(-1, 1)


# -----------------------------------------------------------------------------
# Required entry point
# -----------------------------------------------------------------------------
def predict(observation):
    x = _validate_observation(observation)

    driving_prior, driving_active = _driving_prior(x)
    energy_prior, energy_active = _energy_prior(x)

    return {
        "driving": {
            "prior": driving_prior,
            "active": driving_active,
        },
        "energy": {
            "prior": energy_prior,
            "active": energy_active,
        },
    }