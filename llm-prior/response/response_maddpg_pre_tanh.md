## A. Suggested module

Suggested filename:

```text
llm-prior/programs/fchev_rule_prior.py
```

Suggested program name for CLI selection:

```text
fchev_rule_prior
```

```python
# llm-prior/programs/fchev_rule_prior.py
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
```

---

## B. Explanation of behavior, triggers, masks, fusion, and limitations

### 1. Interface and output format

The module defines:

```python
predict(observation)
```

It accepts either:

```python
observation.shape == (15,)
```

or:

```python
observation.shape == (batch, 15)
```

It validates that the observation is finite and has the required dimension. It returns:

```python
{
    "driving": {
        "prior": driving_latent_prior,
        "active": driving_mask,
    },
    "energy": {
        "prior": energy_latent_prior,
        "active": energy_mask,
    },
}
```

The recommended arrays are:

```python
driving_prior.shape == (batch, 1)
driving_active.shape == (batch, 1)

energy_prior.shape == (batch, 1)
energy_active.shape == (batch, 1)
```

The latent priors are `float32`. The masks are boolean arrays.

For a single observation, the program internally converts it to batch size 1 and returns `(1, 1)` arrays.

---

## 2. Pre-`tanh` latent conversion

The rule computes physical desired actions first, then converts them to latent values before `tanh`.

### Driving actor

The desired acceleration is first clipped to the environment-requested acceleration range:

```python
a_des ∈ [-1.5, 0.7] m/s²
```

Then:

```python
if a_des < 0:
    u_d = a_des / 1.5
else:
    u_d = a_des / 0.7
```

Then:

```python
u_d = clip(u_d, -0.9999, 0.9999)
z_rule = atanh(u_d)
```

This matches the environment mapping:

```python
u_d < 0: a_req = 1.5 u_d
u_d >= 0: a_req = 0.7 u_d
```

### Energy actor

The desired FCS power request is clipped to:

```python
P_req ∈ [0, 50] kW
```

Then:

```python
u_e = 2 P_req / 50 - 1
u_e = clip(u_e, -0.9999, 0.9999)
z_rule = atanh(u_e)
```

This matches:

```python
P_req = 25 (u_e + 1) kW
```

The clipping avoids infinite latent values at `u = ±1`.

---

## 3. Fusion with MADDPG

The MADDPG actor produces a latent value:

```python
z_MADDPG
```

The rule produces:

```python
z_rule
```

For each active actor channel, the trainer blends:

```python
z_fused = (1 - λ) z_MADDPG + λ z_rule
```

Then:

```python
u_fused = tanh(z_fused)
```

The environment maps `u_fused` to the physical requested action.

Important properties:

- `λ` is external and is not hard-coded in the rule.
- If `active = False`, the corresponding rule prior is not fused.
- If `active = True` and `z_rule = 0`, the prior biases the actor latent toward zero, i.e. toward the neutral action:
  - driving neutral: `a_req = 0`
  - energy neutral: `P_req = 25 kW`
- At `λ = 0`, the fused action is essentially the MADDPG action, modulo numerical clipping.
- At `λ = 1`, an active channel uses `tanh(z_rule)`.

The rule is deterministic. It does not add exploration noise. Exploration noise remains trainer-side only.

---

## 4. Driving prior behavior

### 4.1 Observations used for driving

The driving prior uses raw unnormalized observations:

| Index | Name | Use |
|---:|---|---|
| 4 | ego acceleration | comfort smoothing |
| 5 | ego speed | following, speed limits, traffic stopping |
| 6 | lead acceleration | emergency following |
| 7 | lead speed | relative speed and following |
| 8 | gap | following safety |
| 9 | traffic-light state | stop/proceed decision |
| 10 | distance to stop line | approach zone and stopping distance |
| 11 | remaining phase time | whether arrival is during current phase |
| 12 | current speed limit | current overspeed handling |
| 13 | next speed limit | lower-limit region shaping |
| 14 | distance to next speed region | entry-speed shaping |

The rule does not use normalized observations. It assumes the raw values described in the interface.

---

### 4.2 Following prior

The rule uses an IDM-like following heuristic.

It computes safe and maximum following distances consistent with the reward definitions:

```python
D_safe = 2 + 0.5 v + 0.0625 v²
D_max = 10 + v + 0.0825 v²
```

The heuristic target gap is approximately between these values. The IDM-like parameters are:

```python
a_max = 0.7 m/s²
b_comf = 1.5 m/s²
s0 = 5 m
T = 0.5 s
delta = 4
```

The IDM-like desired acceleration is:

```python
a_idm = a_max * [
    1 - (v / v0)^4
    - (s_star / gap)^2
]
```

where:

```python
s_star = s0 + v T + v (v - v_lead) / (2 sqrt(a_max b_comf))
```

This is a short-horizon heuristic, not a certified controller.

The target speed `v0` is slightly above the current speed target:

```python
v0 = speed_target + 0.5
```

This prevents the following prior from braking too aggressively when the ego is already near the posted limit.

If the gap is invalid or sentinel-like, the rule falls back to a speed-target acceleration:

```python
a_speed = speed_target - v
```

The gap is treated as sentinel/invalid for short-horizon following when:

```python
gap >= 1000 m
```

Small negative gaps are treated as unsafe.

---

### 4.3 Emergency following

If:

```python
gap <= D_safe
```

and the ego is moving, the rule applies an emergency deceleration estimate:

```python
a_emergency =
    - (closing_speed)^2 / (2 gap)
    - max(0, -a_lead)
```

where:

```python
closing_speed = max(0, v - v_lead)
```

This approximates the acceleration needed to reduce relative closing speed and also accounts for lead deceleration.

If the gap is less than half the safe distance, the rule requests maximum deceleration:

```python
a = -1.5 m/s²
```

This is a prior request only. The environment may clip the executed acceleration.

---

### 4.4 Traffic-light handling

The rule treats traffic-light phase as relevant only when:

```python
0 <= distance_to_stop_line <= 100 m
```

The sentinel value:

```python
30000 m
```

is treated as no visible target.

A distance of:

```python
0 m
```

is treated as valid and means the ego is at the stop line.

The rule does not use a traffic-light state from a distance greater than 100 m to impose a stop.

---

### 4.5 Red-light handling

For a red light in the 100 m approach zone, the rule estimates:

```python
arrival_time = distance / ego_speed
```

It requests stopping if:

```python
arrival_time < max(remaining_red_time, 0.5)
```

or if the ego is very close to the stop line:

```python
distance <= 5 m
```

The stopping acceleration is estimated from kinematics:

```python
a_stop = -v² / (2 distance)
```

clipped to:

```python
[-1.5, 0] m/s²
```

If the ego is at the stop line and moving, it requests maximum deceleration.

The rule also keeps a red-hold mask active when the ego is stopped at a red light. This allows the prior to bias the actor toward zero acceleration even when the rule’s physical prior is numerically zero.

---

### 4.6 Yellow-light handling

Yellow crossing does not terminate the episode, but crossing yellow may produce a negative traffic-prediction reward.

The rule stops on yellow only when arrival is clearly after the yellow phase:

```python
arrival_time > remaining_yellow_time + 0.5
```

If arrival is during yellow, the rule does not force a stop. This avoids excessive hard braking very close to the stop line when yellow crossing is allowed.

---

### 4.7 Green-light handling

For a green light, the rule stops if arrival is likely after green plus yellow:

```python
arrival_time > remaining_green_time + 3.0 + 0.5
```

If arrival is during green, the rule proceeds.

If arrival is after green but during yellow, the rule does not force a stop because yellow crossing does not trigger red-light termination. It may still produce a negative traffic-prediction reward, but the rule prioritizes avoiding likely red crossing.

---

### 4.8 Speed-limit handling

The rule enforces the current posted limit:

```python
limit = observation[12]
```

If:

```python
v > limit + 0.05
```

it requests deceleration toward the posted limit.

The rule does not treat a red light as a zero speed limit. Traffic-light stopping is handled through the acceleration prior, not by changing the speed-limit observation.

---

### 4.9 Lower-limit region shaping

The rule uses the reward-shaping idea for entering a lower-limit region.

It identifies a lower next region when:

```python
next_limit < current_limit
```

and the next distance is not sentinel.

It estimates the entry speed using the same assumed braking magnitude as the environment reward shaping:

```python
v_entry_assumed = sqrt(max(0, v² - 2 * 0.75 * d_entry))
```

If this estimated entry speed exceeds the next limit, the rule requests braking:

```python
a_entry = (next_limit² - v²) / (2 * d_entry)
```

clipped to the allowed acceleration range.

This is intended to reduce the speed-shaping error before entering the 30 km/h intersection zone.

If the ego is already inside the 30 km/h zone and the next region is 60 km/h, the rule does not apply lower-limit braking because the next limit is higher.

---

### 4.10 Comfort smoothing

The comfort reward penalizes acceleration changes:

```python
c_comfort = -|a_t - a_{t-1}| / 2.2
```

The rule does not have access to future actions, but it can use the current executed acceleration:

```python
observation[4]
```

as a proxy for the previous acceleration.

In non-safety cases, it limits the desired acceleration change to:

```python
0.5 m/s²
```

In safety cases, such as emergency following, red stopping, overspeed, or lower-limit entry braking, it allows larger changes.

This is a heuristic comfort prior. It does not guarantee smoothness because the final executed action depends on MADDPG fusion, exploration noise, and environment clipping.

---

### 4.11 Driving active mask

The driving prior is activated when the rule has a meaningful safety or objective-related correction.

Driving active is true when any of the following is true:

1. Emergency following is required.
2. Traffic-light stopping is required.
3. Current overspeed is detected.
4. Lower-limit entry shaping is required.
5. Gap is invalid or negative.
6. Ego is stopped very close to a stopped lead vehicle.
7. Ego is stopped at a red light.
8. Following, speed, or traffic prior is nonzero and meaningful.

The rule tries to avoid activating a neutral prior unnecessarily. For example, if the ego is already near the speed limit, the gap is safe, and there is no immediate traffic constraint, the driving prior may be inactive. This lets MADDPG act without damping.

However, safety-critical cases remain active even if the physical prior is numerically zero. For example, a stopped ego at a red light activates a red-hold prior with zero acceleration.

---

## 5. Energy prior behavior

### 5.1 Observations used for energy

The energy prior uses:

| Index | Name | Use |
|---:|---|---|
| 0 | battery SOC | SOC-target adjustment |
| 1 | battery SOH | battery stress scaling |
| 2 | FCS SOH | FCS power cap |
| 3 | EDU SOH | conservative EDU-related cap |
| 4 | ego acceleration | approximate mechanical demand |
| 5 | ego speed | approximate mechanical demand |

The rule does not directly observe:

- requested FCS power,
- actual FCS output power,
- battery charge/discharge power,
- EDU torque,
- EDU speed,
- EDU loss,
- hydrogen consumption,
- FCS load-change history,
- battery current,
- battery voltage,
- temperatures.

Therefore, the energy prior is approximate.

---

### 5.2 Approximate mechanical demand

The rule estimates approximate mechanical drive demand from vehicle dynamics:

```python
effective_mass = 7000 * 1.05 = 7350 kg
```

Resistance forces:

```python
F_roll = 0.0085 * 7000 * 9.8
F_drag = 0.5 * 1.225 * 0.55 * 5.8 * v²
```

Drive force estimate:

```python
F_drive = effective_mass * a + F_roll + F_drag
```

Mechanical power estimate:

```python
P_mech ≈ F_drive * v
```

Converted to kW.

This estimate is approximate for several reasons:

1. It uses the current executed acceleration, not the next requested acceleration.
2. It estimates wheel mechanical demand, not exact EDU electrical power.
3. It does not include EDU efficiency losses.
4. It does not include battery charge/discharge losses.
5. It does not include FCS conversion losses.
6. It does not know the actual EDU operating point.
7. It does not know whether the EDU is in motoring or regenerating mode except through the sign of the estimated mechanical power.

The estimate is used only as a soft prior.

---

### 5.3 Base FCS request

If estimated mechanical power is positive, the rule forms a base FCS request:

```python
P_base = clip(P_mech, 0, 50) kW
```

If estimated mechanical power is negative, the base request is zero because the rule interprets this as braking/regeneration.

---

### 5.4 SOC-target adjustment

The SOC objective is to keep:

```python
SOC ≈ 0.5
```

The rule computes:

```python
soc_err = SOC - 0.5
```

If SOC is high:

```python
soc_err > 0.04
```

the rule reduces the FCS request so the battery discharges.

If SOC is low:

```python
soc_err < -0.04
```

the rule increases the FCS request so the battery charges.

The adjustment is capped at 20 kW.

This does not mean the rule controls battery power directly. It only requests FCS power. Battery power is determined by the environment’s energy dispatch and physical limits.

---

### 5.5 Battery SOH scaling

If battery SOH is low, the rule reduces the magnitude of SOC-driven battery buffering:

```python
soc_adjust *= 0.6 + 0.4 * battery_SOH
```

If battery SOH is 1, the factor is 1.0.

If battery SOH is 0.8, the factor is 0.92.

This is a heuristic to reduce aggressive battery charge/discharge when battery health is degraded.

The rule does not model battery current, voltage, temperature, or electrothermal aging directly.

---

### 5.6 Braking and regeneration handling

If estimated mechanical power is negative, the rule interprets this as braking/regeneration.

It then:

- sets FCS request to zero if SOC is not low;
- requests a small FCS power, capped at 10 kW, if SOC is low.

This is approximate. The environment may still dispatch FCS, battery, and EDU power differently.

---

### 5.7 FCS degradation caps

The rule caps FCS request based on FCS SOH:

| FCS SOH | FCS cap |
|---:|---:|
| < 0.95 | 45 kW |
| < 0.85 | 35 kW |
| < 0.65 | 25 kW |

These are heuristic caps, not a direct FCS aging model.

The rule also caps FCS request to 20 kW when:

```python
SOC > 0.6
P_base < 20 kW
```

This tries to avoid unnecessary high FCS power when the battery is already above target and drive demand is low.

---

### 5.8 EDU SOH cap

If EDU SOH is low:

```python
EDU_SOH < 0.8
```

the rule caps FCS request to 40 kW.

This is a conservative proxy. The rule does not directly observe EDU torque, EDU speed, or EDU load-change history. It cannot compute EDU CLR or EDU aging directly.

---

### 5.9 Battery SOH low blending

If battery SOH is below 0.9, the rule blends the requested FCS power toward the estimated mechanical demand:

```python
P_req = 0.5 P_req + 0.5 P_base
```

This reduces excessive battery charge/discharge when battery health is degraded.

Again, this is a heuristic. It does not directly control battery current or voltage.

---

### 5.10 Energy active mask

Energy active is true when the rule has a meaningful reason to influence the energy actor.

It activates when:

1. SOC deviation is significant:
   ```python
   |SOC - 0.5| > 0.05
   ```

2. Estimated mechanical demand is high:
   ```python
   P_mech > 35 kW
   ```

3. Estimated mechanical demand is strongly negative, indicating braking/regeneration:
   ```python
   P_mech < -10 kW
   ```

4. FCS SOH is below 0.95.

5. Battery SOH is below 0.95.

6. EDU SOH is below 0.95.

7. The FCS cap is binding, even if the resulting latent prior is numerically neutral.

If energy active is false, the energy prior is returned as zero but is not fused.

---

## 6. Relation to the stated objectives

### 6.1 Following objective

The driving prior attempts to:

- avoid gaps below `D_safe`;
- avoid excessive gaps beyond `D_max`;
- match lead speed when the gap is in the safe following band;
- accelerate when the gap is too large and speed is below the target;
- decelerate when the ego is closing too quickly;
- account for lead acceleration.

This corresponds to the following reward term using `D_safe`, `D_max`, and `d_ref`.

The rule does not guarantee following reward maximization. It is a soft prior.

---

### 6.2 Collision avoidance

The rule applies emergency deceleration when:

```python
gap <= D_safe
```

and especially when:

```python
gap < 0.5 D_safe
```

or when the gap is nonpositive.

This is intended to reduce the probability of collision, but it cannot guarantee collision avoidance because the final action is fused with MADDPG and may be clipped or modified by the environment.

---

### 6.3 Speed-limit compliance

The rule uses:

```python
observation[12]
```

to enforce the current posted limit.

It also uses:

```python
observation[13]
observation[14]
```

to brake before entering a lower-speed region.

This targets the continuous speeding penalty and the lower-limit entry shaping reward.

It does not explicitly target the 120% termination threshold, but by targeting the posted limit it attempts to stay below it.

---

### 6.4 Traffic-light safety

The rule uses traffic-light phase only in the 100 m approach zone.

It attempts to:

- stop before red crossing;
- proceed on green when arrival is likely during green;
- proceed on yellow when arrival is likely during yellow;
- stop when arrival is likely after yellow, i.e. likely red.

This corresponds to the traffic-light prediction reward and red-light termination condition.

The rule cannot guarantee safe crossing because it only uses current speed, current distance, and current phase time.

---

### 6.5 Comfort

The rule limits acceleration changes relative to the current executed acceleration in non-safety cases.

This is intended to reduce the comfort penalty:

```python
-|a_t - a_{t-1}| / 2.2
```

However, because the rule is stateless and the final action is fused with MADDPG, comfort is not guaranteed.

---

### 6.6 SOC target

The energy prior biases FCS power request to reduce:

```python
|SOC - 0.5|
```

This corresponds to the SOC-deviation objective.

The rule does not control battery power directly. It only requests FCS power. Battery power is determined by the environment’s dispatch logic.

---

### 6.7 Hydrogen consumption and FCS degradation

The rule attempts to reduce unnecessary FCS power by:

- lowering FCS request when SOC is high;
- capping FCS request when FCS SOH is low;
- capping FCS request when SOC is high and demand is low.

This is intended to reduce hydrogen consumption and high-power FCS degradation.

The rule does not directly observe hydrogen consumption or FCS degradation increments.

---

### 6.8 Battery SOH loss

The rule attempts to reduce aggressive battery charge/discharge by:

- scaling SOC adjustments with battery SOH;
- blending FCS request toward mechanical demand when battery SOH is low.

This is intended to reduce battery SOH loss.

The rule does not directly observe battery current, voltage, temperature, or SOH-loss increment.

---

### 6.9 EDU losses and degradation

The rule does not directly observe EDU loss or EDU degradation.

It only uses EDU SOH as a conservative cap on FCS request. It does not compute EDU torque, EDU speed, EDU efficiency, EDU loss, or EDU CLR.

Therefore, EDU-related benefits are indirect and approximate.

---

## 7. Limitations and unavailable information

### 7.1 No direct EDU or battery power observability

The observation does not include:

- motor torque,
- EDU speed,
- EDU power,
- EDU loss,
- battery power,
- battery current,
- battery voltage,
- battery temperature,
- FCS actual output power,
- hydrogen consumption.

The energy prior estimates mechanical demand from speed and acceleration. This estimate may be inaccurate because:

- the current acceleration is executed acceleration, not future requested acceleration;
- EDU efficiency is not known;
- battery losses are not known;
- FCS conversion losses are not known;
- actual power dispatch is environment-determined.

---

### 7.2 No future horizon

The rule uses only the current observation. It does not maintain hidden state across steps.

Traffic-light timing, lead-vehicle behavior, and energy dispatch are estimated only from current values. Therefore, long-horizon planning is limited.

---

### 7.3 Traffic-light predictions are approximate

The rule estimates arrival time as:

```python
distance / ego_speed
```

It does not predict:

- future ego acceleration over multiple steps;
- future lead behavior;
- exact signal phase transitions;
- intersection geometry beyond the stop-line distance;
- vehicle length effects;
- delayed actuation.

Thus, traffic-light stopping and crossing decisions are heuristic.

---

### 7.4 Following prediction is short-horizon

The following prior uses current gap, ego speed, lead speed, and lead acceleration. It does not explicitly solve a multi-step collision-avoidance problem.

It is intended as a safety prior, not a certified controller.

---

### 7.5 Physical constraints remain active

The rule cannot bypass:

- maximum vehicle speed;
- no-reverse handling;
- maximum requested acceleration;
- maximum requested deceleration;
- battery SOC bounds;
- battery current limits;
- battery voltage limits;
- EDU torque-speed validity;
- FCS maximum output;
- FCS/battery/EDU feasibility clipping.

The rule only produces requested-action priors. The environment may still clip or modify the executed action.

---

### 7.6 Rule prior may be neutral

If the rule computes `z_rule = 0`, it is not necessarily inactive.

For example:

- a stopped ego at a red light may use `a_des = 0`, giving `z_rule = 0`, but the driving mask is active to hold the actor near zero;
- an energy request of 25 kW gives `u_e = 0` and `z_rule = 0`, but the energy mask may be active if degradation caps are binding.

When active and `z_rule = 0`, fusion biases the actor latent toward zero. This can damp actor actions toward the neutral action:

- driving neutral: zero acceleration;
- energy neutral: 25 kW FCS request.

The masks try to reduce unnecessary neutral damping, but some safety or degradation constraints intentionally use neutral priors.

---

### 7.7 No exploration noise in the rule

The rule is deterministic. It does not add noise.

Exploration noise is trainer-side only. During evaluation, no exploration noise is added.

---

## 8. CLI selection

Assuming the project CLI uses the specified rule-interface options, select the program as:

```bash
--prior-program-name fchev_rule_prior
```

Select pre-`tanh` latent fusion as:

```bash
--maddpg-rule-output-mode pre_tanh
```

Set the blend coefficient externally, for example:

```bash
--maddpg-rule-lambda 0.2
```

A complete example command pattern would be:

```bash
python <project_train_script>.py \
    --prior-program-name fchev_rule_prior \
    --maddpg-rule-output-mode pre_tanh \
    --maddpg-rule-lambda 0.2
```

The first 1,000 steps of a fresh run should use the project’s prior-free random warm-up. A resumed run should skip that warm-up. Evaluation should not add exploration noise.

---

## 9. Summary

The proposed `fchev_rule_prior` program provides a deterministic pre-`tanh` latent prior for both MADDPG actors.

For driving, it combines:

- IDM-like following;
- emergency gap deceleration;
- current speed-limit enforcement;
- lower-limit entry braking;
- traffic-light stop/proceed logic limited to the 100 m approach zone;
- comfort smoothing using current acceleration;
- no-reverse handling.

For energy, it forms an approximate FCS power request using:

- estimated mechanical drive demand;
- SOC-target adjustment;
- battery SOH-aware buffering;
- FCS SOH caps;
- EDU SOH caps;
- braking/regeneration handling.

It returns per-row latent priors and active masks. The priors are fused before `tanh` with MADDPG latents using the externally supplied `λ`. The rule does not replace the environment, does not guarantee the final executed action, and cannot bypass physical constraints.