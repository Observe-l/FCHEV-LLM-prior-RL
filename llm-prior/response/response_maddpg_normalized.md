## A. Suggested Python rule-prior module

Suggested filename:

```text
llm-prior/programs/fchev_follow_soc_prior.py
```

Complete module:

```python
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
```

---

## B. Explanation of behavior and limitations

### 1. Interface and output contract

The program is intended to be placed at:

```text
llm-prior/programs/fchev_follow_soc_prior.py
```

It exposes only:

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

It validates that the observation is finite and has 15 columns. It returns:

```python
{
    "driving": {"action": (batch, 1) float32, "active": (batch, 1) bool},
    "energy":  {"action": (batch, 1) float32, "active": (batch, 1) bool},
}
```

Both actions are finite and clipped to `[-1, 1]`. Both masks are always `True`, satisfying the requirement that the rule is evaluated and fused at every step.

The program does not return `std`, does not sample noise, does not hard-code the fusion coefficient `lambda`, and does not access SUMO or modify rewards/configuration.

---

### 2. Fusion with MADDPG

The rule returns normalized actions:

```python
u_d_rule in [-1, 1]
u_e_rule in [-1, 1]
```

The adapter is expected to blend them with the MADDPG actors:

```python
u_d_fused = (1 - lambda) * u_d_MADDPG + lambda * u_d_rule
u_e_fused = (1 - lambda) * u_e_MADDPG + lambda * u_e_rule
```

Then the environment applies its physical mappings:

For driving:

```python
if u_d_fused < 0:
    a_req = 1.5 * u_d_fused       # maps [-1, 0] to [-1.5, 0] m/s^2
else:
    a_req = 0.7 * u_d_fused       # maps [0, 1] to [0, 0.7] m/s^2
```

For energy:

```python
P_req = 25 * (u_e_fused + 1)      # maps [-1, 1] to [0, 50] kW
```

The rule itself never bypasses the environment. It only produces normalized prior actions within the documented action bounds. The environment may still clip acceleration, power, battery limits, EDU limits, FCS limits, no-reverse behavior, and the 70 km/h hard speed cap.

---

### 3. Driving prior behavior

The driving prior is a soft prior, not a standalone controller. It tries to support the learned driving actor by providing a reasonable normalized acceleration prior based on the raw observation.

#### 3.1 Following logic

The rule uses the live ego speed, ego acceleration, lead speed, lead acceleration, and gap.

It computes the same following gap definitions described in the environment:

```python
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

It chooses a target gap inside the safe-to-maximum interval:

```python
target_gap = D_safe + 0.6 * (D_max - D_safe)
```

The following acceleration is a simple relative controller:

```python
a_follow = a_lead + GAP_GAIN * gap_error - REL_GAIN * (v - v_lead)
```

where:

```python
gap_error = gap - target_gap
```

with gap error clipped to a reasonable range to avoid extreme acceleration requests.

If the gap is below `D_safe`, a hard safety term forces braking:

```python
safe_brake = a_lead - 0.5 * max(0, v - v_lead) - 0.1 * max(0, D_safe - gap)
```

If the gap is zero or effectively zero, the rule requests maximum braking, `a_req = -1.5 m/s^2`, mapped to `u_d = -1`.

If the gap is larger than the target gap, the rule generally requests acceleration, but only if speed limits and traffic-light constraints allow it.

If the ego is faster than the lead, the relative-speed term reduces acceleration or requests braking to avoid closing the gap too quickly.

If the gap is the sentinel value `30000 m`, the rule does not treat it as an enormous positive gap-error term. It still uses the relative speed term, but avoids accelerating solely because the gap field is sentinel-like.

#### 3.2 Current speed-limit constraint

The raw observation gives the current posted speed limit in index 12.

The rule computes an upper acceleration bound:

```python
if v > v_cur:
    speed_upper = v_cur - v
else:
    speed_upper = max(0, v_cur - v - 0.5)
```

This prevents the rule prior from requesting acceleration that would immediately exceed the current posted limit. It also avoids requesting positive acceleration when the vehicle is already near the limit.

This is a prior constraint. It does not guarantee final speed compliance because the final action is fused with MADDPG and then executed by the environment.

#### 3.3 Next speed-region entry shaping

The observation gives the next speed limit in index 13 and the distance to the next speed-region entrance in index 14.

If the next region has a lower speed limit, the rule estimates the acceleration needed to enter that region at or below the next limit:

```python
a_entry = (v_next^2 - v^2) / (2 * (d_entry - margin))
```

with a small distance margin. If the entrance is very close and the vehicle is above the next limit, the rule requests strong braking, bounded by the vehicle's maximum deceleration request range.

This is intended to support the environment's speed-limit entry shaping objective. It does not implement the reward coefficient `beta_8` directly. It is a rule-level approximation.

If the next speed region has a higher limit, the rule does not brake for that region.

#### 3.4 Traffic-light logic

The rule uses the raw traffic-light state, distance, and remaining phase time.

Traffic-light encoding:

```python
0 = red
1 = green
2 = yellow
```

The rule treats red and yellow as non-green. Crossing on yellow is prohibited by the task description, even if the current simulator termination logic does not terminate on yellow.

The rule handles three cases.

##### Case 1: no next traffic light

If the next traffic-light distance is the sentinel value `30000 m`, the traffic-light constraint is inactive.

##### Case 2: at or past the stop line

If the distance is effectively zero:

- If the current signal is green, the rule allows acceleration.
- If the current signal is red or yellow, the rule requests braking if moving, or zero acceleration if stopped.

A distance of `0 m` is treated as a valid observation meaning the vehicle is at the stop line. The rule does not infer absence of a traffic light from zero distance.

##### Case 3: moving toward the stop line

For nonzero speed, the rule estimates arrival time:

```python
t_arr = distance / speed
```

If `t_arr > 90 s`, the rule does not impose a near-term signal constraint because the rule is recomputed every 1 s step.

If `t_arr <= 90 s`, the rule predicts the signal state at the estimated arrival time using the current phase, remaining phase time, and the fixed cycle:

```text
green 40 s
yellow 3 s
red 47 s
repeat every 90 s
```

If the predicted crossing state is red or yellow, the rule requests braking sufficient to stop before the line, using a small safety margin.

If the predicted state is green but the current phase is red or yellow and the arrival would occur before the current non-green phase ends, the rule requests braking or speed reduction to delay arrival.

If the current phase is green but the arrival is too close to the end of the green phase, the rule conservatively requests braking to avoid crossing on yellow/red. This may forgo a green-crossing bonus, but it prioritizes the stated hard rule that the ego may cross only when the signal state at the crossing instant is green.

For a stopped vehicle, the rule does not divide by zero. It uses a conservative start logic:

- If green, allow acceleration.
- If red, allow acceleration only if a maximum-acceleration start would not reach the stop line before the red phase ends.
- If yellow, wait.

The rule does not implement a fixed braking formula for traffic lights. It uses the current-phase comparison and the predicted arrival state to choose a current-step acceleration prior.

#### 3.5 Comfort smoothing

The observation includes the current executed ego acceleration. The rule uses it for mild comfort smoothing when no safety, speed-limit, entry, or signal constraint is critical:

```python
a_req = a_req + 0.2 * (a_ego - a_req)
```

Then it reapplies safety and limit constraints. This is not a full comfort reward implementation; it is only a smoothing heuristic.

#### 3.6 Normalized driving action

After physical acceleration request `a_req` is computed and clipped to `[-1.5, 0.7]`, the rule maps it to normalized driving action:

```python
if a_req < 0:
    u_d = a_req / 1.5
else:
    u_d = a_req / 0.7
```

Examples:

```text
a_req = -0.75 m/s^2 -> u_d = -0.5
a_req = -1.5 m/s^2  -> u_d = -1.0
a_req =  0.35 m/s^2 -> u_d =  0.5
a_req =  0.7 m/s^2  -> u_d =  1.0
```

The rule does not add a low-speed threshold that converts negative acceleration requests to zero. If the vehicle is at standstill and the rule requests negative acceleration, the environment handles the no-reverse constraint.

---

### 4. Energy prior behavior

The energy prior selects a normalized FCS power request. It does not directly observe hydrogen consumption, battery power, EDU torque, EDU loss, instantaneous FCS output, or degradation increments. It estimates power demand from vehicle kinematics and uses SOC/SOH observations heuristically.

#### 4.1 Approximate power demand

The rule estimates mechanical road demand as:

```python
P_road = m_eff * v * a + 0.5 rho A Cd v^3 + m g Crr v
```

where:

```python
m_eff = 7000 * 1.05
rho = 1.225
A = 5.8
Cd = 0.55
Crr = 0.0085
g = 9.8
```

It then converts this to an approximate electrical bus/FCS-side demand using a coarse efficiency factor:

```python
ETA_EST = 0.90 * 0.95
```

This is explicitly an approximation. It is not the EPA EDU map, not the exact EDU efficiency/loss map, not the DC/DC dispatch model, and not a rated efficiency. The `0.90` factor is a placeholder for approximate combined EDU boundary efficiency, and `0.95` approximates the DC/DC efficiency.

If the estimated mechanical demand is negative, the rule treats direct traction demand as zero, allowing braking/regeneration to be handled by the environment and battery model.

The rule then applies SOC regulation:

```python
soc_gain = 20 + 40 * battery_soh
P_req = P_demand - soc_gain * (SOC - 0.5)
```

If SOC is above 0.5, the rule reduces the FCS request to encourage battery discharge.

If SOC is below 0.5, the rule increases the FCS request to encourage battery charging.

The rule applies stronger behavior near SOC extremes:

```python
SOC > 0.85: reduce FCS request further.
SOC < 0.25: increase FCS request further.
```

It caps the request to `[0, 50] kW`.

When estimated demand is very low, the rule caps the FCS request to at most 25 kW to avoid unnecessary high FCS power during low-load states.

#### 4.2 SOH-based heuristic caps

The rule uses the observed battery SOH, FCS SOH, and EDU SOH to apply a conservative cap:

```python
min_soh = min(battery_soh, fcs_soh, edu_soh)
cap = 50 * (0.70 + 0.30 * min_soh)
P_req = min(P_req, cap)
```

This is a heuristic degradation-avoidance cap. It is not the exact FCS, battery, or EDU aging model. It does not compute hydrogen consumption, FCS degradation increments, battery SOH-loss increments, or EDU CLR increments.

#### 4.3 Normalized energy action

The requested FCS power is mapped to normalized energy action:

```python
u_e = P_req / 25 - 1
```

Examples:

```text
P_req = 0 kW  -> u_e = -1.0
P_req = 25 kW -> u_e =  0.0
P_req = 50 kW -> u_e =  1.0
```

Again, this is only a request. The environment may clip or modify the actual FCS output due to dispatch, EDU demand, battery limits, FCS limits, or other physical constraints.

---

### 5. Relationship to the stated objectives

The driving prior supports the dominant driving objective: follow the lead vehicle while maintaining a gap at least `D_safe` and, when feasible, not exceeding `D_max`. It also supports:

- obeying the current posted speed limit,
- shaping speed before entering a lower posted-limit region,
- avoiding crossing the stop line on red or yellow,
- accelerating through the intersection only when the predicted crossing state is green,
- mild comfort smoothing,
- avoiding unnecessary stopping when a risk can be managed.

The energy prior supports the energy objective:

- reduce hydrogen consumption by avoiding high FCS power when not needed,
- reduce FCS degradation by capping high FCS requests,
- reduce battery degradation heuristically by limiting aggressive SOC regulation and using SOH-based caps,
- keep SOC near 0.5,
- avoid high FCS requests during braking/low-load states.

The rule does not mechanically optimize the reward coefficients `beta_0` through `beta_8`. Those coefficients are training signals for MADDPG. The rule uses the task description as guidance and produces a coherent prior for the learned actors.

---

### 6. Limitations and unavailable information

This rule is approximate and intentionally conservative.

Important limitations:

1. **No direct powertrain observables.**  
   The observation does not include motor torque, instantaneous EDU loss, requested FCS power, battery power, or hydrogen consumption. The energy estimate is therefore approximate.

2. **No exact EDU map usage.**  
   The rule does not query the EPA Generic IPM 150 kW EDU map. It uses a coarse efficiency approximation. It does not add inverter or gearbox losses separately.

3. **No exact battery dispatch model.**  
   The rule does not compute exact battery current, voltage, temperature, 2C discharge limit, 1C charge limit, or electrothermal aging. It uses SOC and SOH heuristics.

4. **No exact FCS map or hydrogen table.**  
   The rule does not use the 50 kW FCS polynomial or hydrogen-consumption table. It uses requested FCS power as a proxy.

5. **No previous FCS power or load-change history.**  
   FCS aging includes startup, low-power, high-power, and load-change effects. The rule does not see previous FCS power or previous energy action, so it cannot directly penalize load-change degradation.

6. **Driving and energy priors are coupled only through an estimate.**  
   The energy prior uses the rule's driving acceleration request as a proxy for upcoming traction demand. The final fused driving action may differ, especially when `lambda` is small.

7. **Traffic-light prediction is conservative.**  
   The rule may brake near green-phase boundaries to avoid yellow/red crossing. This can reduce green-crossing bonus opportunities.

8. **Stopping may not be physically achievable in every case.**  
   If the predicted crossing state is non-green and stopping distance is insufficient, the rule requests maximum braking, but the environment may still execute a crossing. The rule cannot bypass physical limits.

9. **Speed-limit and entry shaping are approximations.**  
   The entry logic assumes constant acceleration over the distance to the next speed region. Actual dynamics, fused actions, and environment execution may differ.

10. **Gap sentinel handling is heuristic.**  
   A gap of `30000 m` is treated as sentinel-like for the gap-error term. Relative speed control is still used.

11. **The rule is a soft prior.**  
   It does not guarantee the final executed action. The final action depends on MADDPG, fusion coefficient `lambda`, exploration noise during training, and environment clipping.

---

### 7. CLI selection

Select the program with the program name, not the filename:

```bash
--prior-program-name fchev_follow_soc_prior
```

Select normalized-action mode:

```bash
--maddpg-rule-output-mode normalized_action
```

Select an external fusion coefficient, for example:

```bash
--maddpg-rule-lambda 0.2
```

The value `lambda` must be in `[0, 1]` and must not be hard-coded inside the rule program.