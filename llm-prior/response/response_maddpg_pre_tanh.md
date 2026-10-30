## (A) Complete Python rule-prior module

Suggested filename:

```text
llm-prior/programs/fchev_follow_energy_prior.py
```

Suggested program name for the CLI:

```text
fchev_follow_energy_prior
```

```python
"""
FCHEV MADDPG rule prior: driving + energy latent priors before tanh.

This rule is a soft prior. It does not replace MADDPG, does not add exploration
noise, and does not bypass environment clipping or physical feasibility limits.

Output mode:
    pre_tanh latent priors
        driving.prior  = z_rule for driving actor
        energy.prior   = z_rule for energy actor

Fusion is performed externally as:
    z_fused = (1 - lambda) * z_MADDPG + lambda * z_rule
    u_fused = tanh(z_fused)

The rule receives the raw, unnormalized 15-dimensional observation:
    0  battery SOC
    1  battery SOH
    2  FCS SOH
    3  EDU SOH
    4  current ego acceleration
    5  current ego speed
    6  current lead acceleration
    7  current lead speed
    8  ego-to-lead gap
    9  next traffic light state: red 0, green 1, yellow 2
    10 distance to next traffic-light stop line
    11 remaining time in current signal phase
    12 current speed limit
    13 next speed limit
    14 distance to next speed-region entrance
"""

import numpy as np


# -----------------------------------------------------------------------------
# Interface and physical constants
# -----------------------------------------------------------------------------

_SENTINEL = 30000.0

_MIN_ACCEL = -1.5
_MAX_ACCEL = 0.7

_HARD_MAX_SPEED = 19.4444          # 70 km/h hard vehicle cap

_MAX_FCS = 50.0                    # kW

_TARGET_SOC = 0.5
_SOC_DEADBAND = 0.03
_SOC_GAIN = 50.0                   # kW per unit SOC deviation, before SOH scaling

# Traffic-light cycle constants.
_CYCLE = 90.0
_GREEN = 40.0
_YELLOW = 3.0
_RED = 47.0

# Safety margins used only inside the rule.
_SIGNAL_MARGIN = 0.25              # s, phase-boundary safety margin
_CROSS_MARGIN = 0.35               # s, margin for crossing before current green ends
_STOP_MARGIN = 0.5                 # m, margin for treating a stop as safely before the line

# Vehicle/powertrain constants used only for energy estimation.
_M_EFF = 7000.0 * 1.05
_DRAG_COEFF = 0.5 * 1.225 * 0.55 * 5.8
_ROLL_COEFF = 7000.0 * 9.8 * 0.0085


# -----------------------------------------------------------------------------
# Latent-space helpers
# -----------------------------------------------------------------------------

def _clip_u(u):
    """Clip normalized action u to avoid infinite atanh."""
    return np.clip(u, -0.9999, 0.9999)


def _atanh_u(u):
    """Convert normalized u to pre-tanh latent z = atanh(u)."""
    return np.arctanh(_clip_u(u))


def _accel_to_u(a):
    """
    Convert desired acceleration to driving latent prior.

    Environment mapping:
        a < 0:  a_req = 1.5 u_d,  u_d in [-1, 0]
        a >= 0: a_req = 0.7 u_d,  u_d in [ 0, 1]
    """
    a = np.clip(a, _MIN_ACCEL, _MAX_ACCEL)
    u = np.where(a < 0.0, a / 1.5, a / 0.7)
    return _atanh_u(u)


def _power_to_u(p_req_kw):
    """
    Convert requested FCS power in kW to energy latent prior.

    Environment mapping:
        P_req = 25 * (u_e + 1), u_e in [-1, 1]
        equivalently: u_e = 2 P_req / 50 - 1
    """
    p_req_kw = np.clip(p_req_kw, 0.0, _MAX_FCS)
    u = 2.0 * p_req_kw / _MAX_FCS - 1.0
    return _atanh_u(u)


# -----------------------------------------------------------------------------
# Traffic-light helpers
# -----------------------------------------------------------------------------

def _phase_state_at(state, rem, t):
    """
    Predict signal state at estimated arrival time t.

    state:
        0 red, 1 green, 2 yellow
    rem:
        remaining time in the currently observed phase.
    t:
        estimated arrival time from the current step.

    The rule uses a repeating cycle:
        green 40 s, yellow 3 s, red 47 s.
    """
    state = int(state)
    if state not in (0, 1, 2):
        state = 0

    rem = float(rem)
    t = float(t)

    if t < 0.0:
        t = 0.0
    if rem < 0.0:
        rem = 0.0

    # If the current phase has already ended, advance from the phase boundary.
    if rem <= 1.0e-6:
        if t <= 1.0e-6:
            return state
        offset = t
    elif t < rem:
        return state
    else:
        offset = t - rem

    offset = offset % _CYCLE

    if state == 1:  # currently green; after current green: yellow, red, green
        if offset < _YELLOW:
            return 2
        if offset < _YELLOW + _RED:
            return 0
        return 1

    if state == 2:  # currently yellow; after current yellow: red, green, yellow
        if offset < _RED:
            return 0
        if offset < _RED + _GREEN:
            return 1
        return 2

    # currently red; after current red: green, yellow, red
    if offset < _GREEN:
        return 1
    if offset < _GREEN + _YELLOW:
        return 2
    return 0


def _crossing_time(v, a, d):
    """
    Estimate time for ego to reach the stop line under constant acceleration a,
    with no-reverse handling approximated by stopping before reversing.

    Returns np.inf if the vehicle never reaches the stop line before stopping.
    """
    v = float(v)
    a = float(a)
    d = float(d)

    if d <= 0.0:
        return 0.0

    if v <= 1.0e-6:
        if a > 1.0e-6:
            return float(np.sqrt(2.0 * d / a))
        return np.inf

    if a > 1.0e-6:
        disc = v * v + 2.0 * a * d
        if disc < 0.0:
            return np.inf
        return float((-v + np.sqrt(disc)) / a)

    if a >= -1.0e-6:
        return float(d / v)

    # Negative acceleration: check whether the vehicle stops before the line.
    stop_dist = v * v / (2.0 * (-a))
    if stop_dist < d - _STOP_MARGIN:
        return np.inf

    disc = v * v + 2.0 * a * d
    if disc < 0.0:
        return np.inf

    # Positive root corresponding to crossing before stopping.
    t_cross = (v - np.sqrt(disc)) / (-a)
    return float(max(0.0, t_cross))


def _traffic_action_safe(v, a, d, state, rem):
    """
    Return True if candidate acceleration a does not make the ego cross the
    stop line on red or yellow, according to this rule's one-step forecast.
    """
    # Sentinel means no visible target/outside effective range.
    if d >= _SENTINEL - 1.0:
        return True

    # Distance 0 is valid: ego is at the stop line. Crossing is allowed only
    # when the observed state is green.
    if d <= 0.0:
        return int(state) == 1

    t_cross = _crossing_time(v, a, d)
    if np.isinf(t_cross):
        return True

    # Add a small safety margin so exact phase-boundary crossings are avoided.
    pred_state = _phase_state_at(state, rem, t_cross + _SIGNAL_MARGIN)
    return pred_state == 1


def _traffic_candidates(v, a_base, d, state, rem, speed_max):
    """
    Generate candidate accelerations for the local traffic-light safety search.

    This is not a fixed braking formula. The rule evaluates candidate accelerations
    and keeps the closest safe candidate that also respects the speed-limit cap.
    """
    cands = [a_base, 0.0, _MIN_ACCEL, _MAX_ACCEL, speed_max]

    if d < _SENTINEL - 1.0:
        d_eff = max(float(d), 0.1)
        v = float(v)
        state = int(state)

        if v > 1.0e-6:
            a_stop = -v * v / (2.0 * d_eff)
        else:
            a_stop = _MIN_ACCEL

        # Exact stop-at-line and slightly stronger stop-before-line candidates.
        cands.extend([a_stop, a_stop - 0.2])

        if state == 1:  # green
            # Candidate that crosses before the current green ends.
            t_current = rem - _CROSS_MARGIN
            if t_current > 0.5:
                a_cross = 2.0 * (d_eff - v * t_current) / (t_current * t_current)
                cands.extend([a_cross, a_cross + 0.2])

            # Candidate that avoids crossing until the next green after the
            # current green ends: yellow 3 s + red 47 s.
            t_next_green = rem + _YELLOW + _RED + _SIGNAL_MARGIN
            if t_next_green > 0.5:
                a_no_cross = 2.0 * (d_eff - v * t_next_green) / (t_next_green * t_next_green)
                cands.extend([a_no_cross, min(a_no_cross, a_stop)])

        elif state == 0:  # red
            # Avoid crossing before the current red phase ends.
            t_next_green = rem + _SIGNAL_MARGIN
            if t_next_green > 0.5:
                a_no_cross = 2.0 * (d_eff - v * t_next_green) / (t_next_green * t_next_green)
                cands.extend([a_no_cross, min(a_no_cross, a_stop)])

        elif state == 2:  # yellow
            # Avoid crossing before the next green after yellow + red.
            t_next_green = rem + _RED + _SIGNAL_MARGIN
            if t_next_green > 0.5:
                a_no_cross = 2.0 * (d_eff - v * t_next_green) / (t_next_green * t_next_green)
                cands.extend([a_no_cross, min(a_no_cross, a_stop)])

    # Clip to action bounds and remove near-duplicates.
    out = []
    for c in cands:
        c = float(np.clip(c, _MIN_ACCEL, _MAX_ACCEL))
        if not np.isfinite(c):
            continue
        if all(abs(c - u) > 1.0e-4 for u in out):
            out.append(c)
    return out


# -----------------------------------------------------------------------------
# Main rule interface
# -----------------------------------------------------------------------------

def predict(observation):
    """
    Return deterministic pre-tanh latent priors for both MADDPG actors.

    Input:
        observation: raw unnormalized observation, shape (15,) or (batch, 15)

    Output:
        {
            "driving": {"prior": (batch, 1) float32, "active": (batch, 1) bool},
            "energy":  {"prior": (batch, 1) float32, "active": (batch, 1) bool}
        }

    Both active masks are always True.
    """
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15 or not np.all(np.isfinite(x)):
        raise ValueError("observation must be finite with shape (15,) or (batch, 15)")

    n = x.shape[0]

    drive_prior = np.empty((n, 1), dtype=np.float32)
    energy_prior = np.empty((n, 1), dtype=np.float32)
    active_drive = np.ones((n, 1), dtype=bool)
    active_energy = np.ones((n, 1), dtype=bool)

    if n == 0:
        return {
            "driving": {"prior": drive_prior, "active": active_drive},
            "energy": {"prior": energy_prior, "active": active_energy},
        }

    # ---------------------------------------------------------------------
    # Extract raw observations
    # ---------------------------------------------------------------------
    v = np.maximum(x[:, 5].astype(np.float64), 0.0)
    a_cur = x[:, 4].astype(np.float64)
    a_lead = x[:, 6].astype(np.float64)
    v_lead = np.maximum(x[:, 7].astype(np.float64), 0.0)

    gap = np.maximum(x[:, 8].astype(np.float64), 0.0)

    dist_light = np.maximum(x[:, 10].astype(np.float64), 0.0)
    sentinel_light = dist_light >= (_SENTINEL - 1.0)

    state = np.clip(np.round(x[:, 9]).astype(np.int64), 0, 2)
    rem = np.maximum(x[:, 11].astype(np.float64), 0.0)

    limit_cur = x[:, 12].astype(np.float64)
    limit_next = x[:, 13].astype(np.float64)
    d_entry = x[:, 14].astype(np.float64)

    # ---------------------------------------------------------------------
    # Driving prior: following + speed limit + speed-region entry shaping
    # ---------------------------------------------------------------------

    # Following-distance band. The rule keeps a small margin inside the reward
    # thresholds: D_safe + 1 m and D_max - 1 m.
    D_safe = 2.0 + 0.5 * v + 0.0625 * v * v
    D_max = 10.0 + v + 0.0825 * v * v

    D_safe_eff = D_safe + 1.0
    D_max_eff = np.maximum(D_safe_eff, D_max - 1.0)

    rel_des = np.zeros_like(v)

    too_close = gap < D_safe_eff
    too_far = gap > D_max_eff

    if np.any(too_close):
        rel_des[too_close] = np.clip(
            gap[too_close] - D_safe_eff[too_close],
            -8.0,
            0.0,
        )

    if np.any(too_far):
        rel_des[too_far] = np.clip(
            gap[too_far] - D_max_eff[too_far],
            0.0,
            8.0,
        )

    # Lead-vehicle feedforward plus gap correction.
    # If gap is in the safe band, the rule mostly tries to match the lead speed.
    a_follow = a_lead + (v_lead - v) + rel_des
    a_follow = np.clip(a_follow, _MIN_ACCEL, _MAX_ACCEL)

    # Current posted speed limit, respecting the 70 km/h hard cap.
    v_limit = np.minimum(limit_cur, _HARD_MAX_SPEED)
    v_allowed = v_limit.copy()

    # Speed-region entry shaping: if the next speed limit is lower, decelerate
    # early enough to enter the lower-limit region at or below its limit.
    lower_entry = (
        (limit_next < limit_cur - 0.1)
        & (d_entry < _SENTINEL - 1.0)
        & (d_entry >= 0.0)
    )

    if np.any(lower_entry):
        idx = lower_entry
        d_e = np.maximum(d_entry[idx], 0.0)
        v_allow_entry = np.sqrt(
            np.maximum(
                0.0,
                limit_next[idx] * limit_next[idx] + 2.0 * 1.0 * d_e,
            )
        )
        v_allowed[idx] = np.minimum(v_allowed[idx], v_allow_entry)

    # Speed-limit acceleration cap.
    a_speed = v_allowed - v
    a_speed = np.where(
        v >= v_allowed - 0.2,
        np.minimum(a_speed, 0.0),
        a_speed - 0.1,
    )
    a_speed = np.clip(a_speed, _MIN_ACCEL, _MAX_ACCEL)

    # Base driving action before traffic-light safety adjustment.
    a_base = np.minimum(a_follow, a_speed)

    # Predicted gap for emergency detection.
    pred_gap = gap + v_lead - v + 0.5 * (a_lead - a_base)
    emergency = (gap < D_safe_eff + 2.0) | (pred_gap < D_safe_eff - 2.0)

    # Mild jerk/comfort limiting, except when following safety is urgent.
    a_smooth = np.clip(
        np.clip(a_base, a_cur - 0.8, a_cur + 0.8),
        _MIN_ACCEL,
        _MAX_ACCEL,
    )
    a_base = np.where(emergency, a_base, a_smooth)
    a_base = np.clip(a_base, _MIN_ACCEL, _MAX_ACCEL)

    # ---------------------------------------------------------------------
    # Driving prior: local traffic-light safety search
    # ---------------------------------------------------------------------
    for i in range(n):
        v_i = float(v[i])
        a_base_i = float(a_base[i])
        d_i = float(dist_light[i])
        st_i = int(state[i])
        r_i = float(rem[i])

        # Speed-limit cap used as a soft feasibility filter for candidates.
        speed_max = float(np.clip(v_allowed[i] - v_i + 0.05, _MIN_ACCEL, _MAX_ACCEL))

        if sentinel_light[i]:
            a_final = a_base_i
        else:
            candidates = _traffic_candidates(
                v_i,
                a_base_i,
                d_i,
                st_i,
                r_i,
                speed_max,
            )

            best_safe = None
            best_safe_cost = np.inf

            for a_c in candidates:
                safe = _traffic_action_safe(v_i, a_c, d_i, st_i, r_i)

                speed_violation = max(0.0, a_c - speed_max)
                harsh_brake = max(0.0, -a_c - 0.5)
                harsh_accel = max(0.0, a_c - 0.5)

                cost = (
                    abs(a_c - a_base_i)
                    + 0.25 * abs(a_c - float(a_cur[i]))
                    + 1000.0 * speed_violation
                    + 0.2 * harsh_brake
                    + 0.1 * harsh_accel
                )

                if safe and cost < best_safe_cost:
                    best_safe = a_c
                    best_safe_cost = cost

            if best_safe is not None:
                a_final = best_safe
            else:
                # No safe candidate found in the local search. If the ego is
                # already at the stop line and the observed signal is green,
                # keep the base action; otherwise request maximum braking.
                if d_i <= 0.0 and st_i == 1:
                    a_final = a_base_i
                else:
                    a_final = _MIN_ACCEL

            a_final = float(np.clip(a_final, _MIN_ACCEL, _MAX_ACCEL))

        drive_prior[i, 0] = _accel_to_u(a_final)

    # ---------------------------------------------------------------------
    # Energy prior: FCS power request latent prior
    # ---------------------------------------------------------------------
    soc = np.clip(x[:, 0].astype(np.float64), 0.0, 1.0)
    bat_soh = np.clip(x[:, 1].astype(np.float64), 0.0, 1.0)
    fcs_soh = np.clip(x[:, 2].astype(np.float64), 0.0, 1.0)
    edu_soh = np.clip(x[:, 3].astype(np.float64), 0.0, 1.0)

    # Approximate wheel-level mechanical power demand.
    # This is an estimate, not an observed electrical/EDU/FCS power value.
    p_mech_kw = (
        _M_EFF * a_cur * v
        + _DRAG_COEFF * v * v * v
        + _ROLL_COEFF * v
    ) / 1000.0

    braking = p_mech_kw < 0.0
    p_demand_kw = np.maximum(0.0, p_mech_kw)

    # SOC regulation with a small deadband to avoid unnecessary battery/FCS work.
    dev = soc - _TARGET_SOC
    soc_gain = _SOC_GAIN * np.clip(bat_soh, 0.5, 1.0)

    p_soc_kw = -soc_gain * dev
    p_soc_kw = np.where(np.abs(dev) <= _SOC_DEADBAND, 0.0, p_soc_kw)

    # During net braking, regenerative braking is assumed to help charge the
    # battery. Therefore FCS charge correction is reduced.
    p_soc_kw = np.where(
        braking,
        np.maximum(0.0, p_soc_kw) * 0.25,
        p_soc_kw,
    )

    p_req_kw = p_demand_kw + p_soc_kw

    # Conservative battery-power feasibility limits.
    # These are approximations; actual battery limits depend on voltage, current,
    # temperature, cell state, and the environment's dispatch algorithm.
    bat_soh_clip = np.clip(bat_soh, 0.5, 1.0)
    p_batt_dis_max = 120.0 * bat_soh_clip
    p_batt_chg_max = 60.0 * bat_soh_clip

    p_min = np.maximum(0.0, p_demand_kw - p_batt_dis_max)
    p_max = np.minimum(_MAX_FCS, p_demand_kw + p_batt_chg_max)

    # If battery SOH is very low, feasibility limits can conflict. Keep a valid
    # interval and prioritize physical feasibility over the conservative estimate.
    p_min = np.minimum(p_min, p_max)

    p_req_kw = np.clip(p_req_kw, p_min, p_max)

    # FCS health cap. EDU SOH is used only as a very mild system-health factor;
    # FCS request does not directly set EDU mechanical torque/power.
    fcs_health = np.clip(fcs_soh, 0.5, 1.0)
    edu_health = np.clip(edu_soh, 0.5, 1.0)
    p_health_cap = (
        _MAX_FCS
        * (0.7 + 0.3 * fcs_health)
        * (0.9 + 0.1 * edu_health)
    )

    if p_health_cap >= p_min:
        p_req_kw = np.minimum(p_req_kw, p_health_cap)
    else:
        # If health cap conflicts with battery-feasibility minimum, keep the
        # feasibility minimum. The environment will still apply actual limits.
        p_req_kw = np.maximum(p_req_kw, p_min)

    p_req_kw = np.clip(p_req_kw, p_min, p_max)
    p_req_kw = np.clip(p_req_kw, 0.0, _MAX_FCS)

    energy_prior[:, 0] = _power_to_u(p_req_kw)

    return {
        "driving": {"prior": drive_prior, "active": active_drive},
        "energy": {"prior": energy_prior, "active": active_energy},
    }
```

---

## (B) Explanation of the rule behavior

### 1. Output mode and fusion

The module returns pre-`tanh` latent priors:

```python
{
    "driving": {"prior": z_rule_d, "active": True},
    "energy":  {"prior": z_rule_e, "active": True}
}
```

The MADDPG fusion is performed externally:

```python
z_fused = (1 - lambda) * z_MADDPG + lambda * z_rule
u_fused = tanh(z_fused)
```

Then the environment maps the fused normalized action:

Driving actor:

```python
if u_fused < 0:
    a_req = 1.5 * u_fused      # maps [-1, 0] to [-1.5, 0] m/s^2
else:
    a_req = 0.7 * u_fused      # maps [0, 1] to [0, 0.7] m/s^2
```

Energy actor:

```python
P_req = 25 * (u_fused + 1)     # maps [-1, 1] to [0, 50] kW
```

The rule does not hard-code `lambda`. It only supplies `z_rule`.

The rule is deterministic and does not add exploration noise. It does not return `std`.

Both masks are always `True`, so the rule is evaluated and fused every step for both actors.

---

## 2. Driving prior behavior

The driving prior tries to support the dominant task: follow the lead vehicle while respecting speed limits and traffic signals.

### 2.1 Following-distance band

The rule uses the documented following-distance definitions:

```python
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

It uses slightly conservative margins:

```python
D_safe_eff = D_safe + 1 m
D_max_eff  = D_max - 1 m
```

The rule behavior is:

- If `gap < D_safe_eff`, the ego is too close. The rule requests deceleration.
- If `gap > D_max_eff`, the gap is too large. The rule requests acceleration to close the gap.
- If the gap is inside the safe band, the rule mostly tries to match the lead vehicle speed.

The acceleration prior is:

```python
a_follow = a_lead + (v_lead - v) + rel_des
```

where `rel_des` is a gap-error-based relative-speed correction.

This gives:

- lead acceleration feedforward;
- speed matching when the gap is acceptable;
- deceleration when too close;
- acceleration when too far.

The final acceleration is clipped to the allowed action range:

```python
[-1.5, 0.7] m/s^2
```

The rule does not convert negative acceleration to zero at low speed. It may request negative acceleration at standstill if the gap is unsafe, but the environment handles the no-reverse clipping.

---

### 2.2 Speed-limit and speed-region entry shaping

The rule uses:

- index 12: current speed limit;
- index 13: next speed limit;
- index 14: distance to the next speed-region entrance.

It caps the ego speed to the current posted limit and the 70 km/h hard cap.

When the next speed limit is lower than the current one, the rule estimates the maximum speed allowed now so that the ego can decelerate before entering the lower-limit region:

```python
v_allow_entry = sqrt(v_next_limit^2 + 2 * brake_mag * d_entry)
```

with `brake_mag = 1.0 m/s^2`.

This is intended to support the β₈ speed-limit-entry shaping objective. It is not a reward-function implementation; it is a soft prior.

If the ego is already at or near the allowed speed, the rule suppresses positive acceleration.

---

### 2.3 Comfort/jerk limiting

The rule applies a mild jerk limit:

```python
a_base = clip(a_base, a_cur - 0.8, a_cur + 0.8)
```

except when the predicted gap is close to the safe threshold.

This supports the β₅ comfort reward by discouraging abrupt acceleration changes, while still allowing emergency braking when following safety is urgent.

---

## 3. Traffic-light handling

The rule does not use a single fixed braking formula. It evaluates candidate accelerations and chooses the closest candidate that does not make the ego cross the stop line on red or yellow.

### 3.1 Sentinel handling

If the traffic-light distance is near `30000 m`, the rule treats the light as no visible target/outside the effective range.

Traffic-light distance `0 m` is treated as valid: the ego is at the stop line.

Negative distance values are clipped to zero for safety calculations.

---

### 3.2 Arrival-state prediction

For a candidate acceleration `a`, the rule estimates the crossing time:

```python
t_cross = crossing_time(v, a, distance_to_stop_line)
```

It then predicts the signal state at:

```python
t_cross + 0.25 s
```

The small margin avoids exact phase-boundary crossings.

The phase prediction follows the documented cycle:

```text
green 40 s
yellow 3 s
red 47 s
total 90 s
```

If the predicted state at crossing is green, the candidate is considered safe.

If the predicted state is red or yellow, the candidate is rejected unless the vehicle stops before the stop line.

---

### 3.3 Candidate search

For each step, the rule generates candidate accelerations such as:

- the base following/speed-limit action;
- `0`;
- maximum braking `-1.5`;
- maximum acceleration `+0.7`;
- a stop-at-line acceleration;
- a slightly stronger stop-before-line acceleration;
- an acceleration that crosses before the current green ends, if feasible;
- an acceleration that avoids crossing until the next green phase.

For each candidate, the rule checks:

1. Does it avoid crossing on red or yellow?
2. Does it violate the current speed-limit cap?
3. Is it close to the base following action?
4. Is it comfortable?

It selects the safe candidate with the lowest cost.

If no safe candidate is found, the rule requests maximum braking, except when the ego is already at the stop line and the observed signal is green.

This traffic logic is recomputed every 1 s step from the new observation. It is not an open-loop trajectory.

---

## 4. Energy prior behavior

The energy prior produces a requested FCS power latent prior. It does not directly control driving acceleration, battery torque, EDU torque, or actual executed power. It only provides a soft prior for the energy actor.

### 4.1 Observations used

The energy prior uses:

- index 0: battery SOC;
- index 1: battery SOH;
- index 2: FCS SOH;
- index 3: EDU SOH;
- index 4: current ego acceleration;
- index 5: current ego speed.

It does not directly observe:

- motor torque;
- instantaneous EDU loss;
- requested FCS power;
- actual battery power;
- hydrogen consumption;
- EDU efficiency operating point;
- DC/DC operating point.

Therefore the power-demand estimate is explicitly approximate.

---

### 4.2 Approximate mechanical power demand

The rule estimates wheel-level mechanical power using the documented vehicle constants:

```python
P_mech = m_eff * a * v
         + 0.5 * rho * Cd * A * v^3
         + m * g * Crr * v
```

with:

```python
m_eff = 7000 * 1.05 kg
```

This is a mechanical power estimate. It is not the actual electrical EDU input power, battery power, or FCS output power. It does not include the full EPA EDU efficiency map, accessory loads, battery voltage/current limits, or the environment's actual power-dispatch logic.

If the estimated mechanical power is negative, the rule treats the vehicle as braking/regenerating and sets the baseline FCS demand to zero.

---

### 4.3 SOC regulation

The energy prior tries to keep SOC near:

```python
SOC_target = 0.5
```

It uses a deadband:

```python
|SOC - 0.5| <= 0.03
```

Inside the deadband, it does not add SOC correction.

Outside the deadband:

- if SOC is low, it increases the FCS power request;
- if SOC is high, it decreases the FCS power request.

During braking, the SOC charge correction is reduced because regenerative braking is assumed to help charge the battery.

This supports the SOC-deviation objective without mechanically optimizing the reward equation.

---

### 4.4 Battery feasibility approximation

The rule applies conservative approximate battery charge/discharge limits:

```python
P_batt_discharge_max ≈ 120 * battery_SOH kW
P_batt_charge_max    ≈ 60  * battery_SOH kW
```

These are approximations. The actual battery model is electrothermal and includes current, voltage, temperature, RC dynamics, and aging effects.

The rule clips the requested FCS power so that the estimated battery power:

```python
P_battery ≈ P_FCS - P_demand
```

does not exceed these approximate limits.

If the approximate limits conflict, the rule chooses a valid power request and leaves final feasibility to the environment.

---

### 4.5 FCS health and degradation approximation

The rule limits high FCS request when FCS SOH is lower:

```python
P_health_cap = 50 * (0.7 + 0.3 * FCS_SOH) * (0.9 + 0.1 * EDU_SOH)
```

This is a soft degradation-aware cap, not a physical model of FCS aging.

The EDU SOH factor is intentionally mild. The energy action controls requested FCS power, not the actual EDU mechanical torque/power, which is primarily affected by the driving action and vehicle dynamics.

---

### 4.6 Energy latent conversion

After computing `P_req_kw` in `[0, 50]`, the rule converts it to normalized action:

```python
u_e = 2 * P_req / 50 - 1
```

then to pre-`tanh` latent:

```python
z_rule_e = atanh(clip(u_e, -0.9999, 0.9999))
```

For example:

```text
P_req = 0 kW   -> u_e ≈ -1 -> z_rule_e strongly negative
P_req = 25 kW  -> u_e ≈ 0  -> z_rule_e ≈ 0
P_req = 50 kW  -> u_e ≈ 1  -> z_rule_e strongly positive
```

---

## 5. Relation to the stated MADDPG objectives

### Driving actor

The driving prior supports:

- β₀ following: keeps the gap near the safe band and closes large gaps when feasible;
- β₅ comfort: mild jerk limiting;
- β₇ traffic-light prediction: forecasts arrival state and avoids red/yellow crossing;
- β₈ speed-limit entry shaping: decelerates before lower-limit regions;
- speed-limit compliance: caps the latent action using current and next speed regions;
- 70 km/h hard cap: included as an absolute speed cap.

It does not implement the exact reward function. It provides a reference prior that encourages behavior consistent with the driving critic's objectives.

---

### Energy actor

The energy prior supports:

- β₁ hydrogen consumption: requests lower FCS power when possible;
- β₂ FCS degradation: caps high FCS power when FCS SOH is lower;
- β₃ battery SOH loss: limits approximate battery charge/discharge power and uses SOC deadband;
- β₆ SOC deviation: regulates SOC toward 0.5;
- β₄ EDU degradation: only indirectly supported through mild system-health factors, because requested FCS power does not directly set EDU mechanical torque/power.

The rule does not assume hydrogen consumption, EDU loss, battery power, or motor torque are directly observable.

---

## 6. Important limitations

1. **The rule is a soft prior.**  
   It does not guarantee the final executed action. The MADDPG action and environment clipping determine the final executed acceleration and power request.

2. **Traffic-light safety is forecast-based.**  
   The rule uses a 1 s decision step and a constant-acceleration crossing estimate. The actual SUMO/environment dynamics, signal transition timing, and crossing instant may differ.

3. **Traffic safety can be unavoidable.**  
   If the ego is already too close to the stop line to avoid crossing before the next green, the rule requests maximum braking but cannot physically prevent a late crossing.

4. **The energy demand estimate is approximate.**  
   It uses current acceleration and speed to estimate wheel mechanical power. It does not observe EDU torque, EDU efficiency, battery current/voltage, actual FCS output, hydrogen consumption, or accessory loads.

5. **SOC deviation is not the same as degradation.**  
   SOC regulation supports the energy objective but does not directly measure battery SOH loss, FCS aging, or EDU aging.

6. **Battery and FCS limits are approximations.**  
   The rule uses conservative power bounds. The environment's actual electrothermal battery model, EDU map, and dispatch algorithm remain the authoritative constraints.

7. **No test or validation is claimed here.**  
   This is a static design explanation only.

---

## 7. CLI selection

Save the module as:

```text
llm-prior/programs/fchev_follow_energy_prior.py
```

Then select it with:

```bash
--prior-program-name fchev_follow_energy_prior
--maddpg-rule-output-mode pre_tanh
--maddpg-rule-lambda 0.3
```

Use any `lambda` in `[0, 1]`.

At `lambda = 0`, the MADDPG action is used.  
At `lambda = 1`, the rule prior is used for active channels.  
In between, the rule and MADDPG latent actions are blended before `tanh`.