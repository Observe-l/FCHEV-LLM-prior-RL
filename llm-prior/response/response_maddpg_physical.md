## Suggested filename

```text
llm-prior/programs/fchev_follow_energy_rule.py
```

The program name for the CLI would be:

```text
fchev_follow_energy_rule
```

---

## Complete Python module

```python
# llm-prior/programs/fchev_follow_energy_rule.py
"""
Rule prior for FCHEV driving and energy management.

This program returns physical actions:
- driving.action: requested ego acceleration in m/s^2, clipped to [-1.5, 0.7]
- energy.action: requested FCS output power in kW, clipped to [0, 50]

It is a deterministic soft prior. It does not add noise, does not return std,
does not bypass environment constraints, and does not hard-code lambda.
"""

import numpy as np


# -----------------------------------------------------------------------------
# Constants from the documented environment
# -----------------------------------------------------------------------------

DT = 1.0

A_MIN = -1.5
A_MAX = 0.7

V_CAP = 70.0 / 3.6  # 19.4444 m/s

SENTINEL = 30000.0
SENTINEL_THRESHOLD = 29000.0

EPS = 1e-6

# Traffic-light cycle
GREEN = 40.0
YELLOW = 3.0
RED = 47.0
CYCLE = 90.0

# Safety margins used by the rule. These are conservative design choices,
# not environment constants.
SIGNAL_CROSS_MARGIN = 0.5
SIGNAL_WAIT_MARGIN = 1.0

# Vehicle parameters
M = 7000.0
ROTATIONAL_MASS_COEFF = 1.05
M_EFF = M * ROTATIONAL_MASS_COEFF

RHO = 1.225
CD = 0.55
FRONTAL_AREA = 5.8
CRR = 0.0085
G = 9.8

DRAG_COEFF = 0.5 * RHO * CD * FRONTAL_AREA
ROLL_RES = CRR * M * G

# Approximate combined EDU efficiency.
# This is not a rated efficiency and not a peak efficiency. It is only a
# simplified boundary estimate for converting mechanical road power to an
# approximate electrical EDU demand estimate.
EDU_EFF_ASSUMED = 0.90

# Energy-management heuristic constants
SOC_TARGET = 0.5
SOC_GAIN = 80.0
FCS_MAX = 50.0


# -----------------------------------------------------------------------------
# Signal helpers
# -----------------------------------------------------------------------------

def _sequence_after(state):
    """
    Return the repeating phase sequence that starts after the current phase.

    State encoding:
      0 = red
      1 = green
      2 = yellow
    """
    if state == 1:
        # After green: yellow -> red -> green -> ...
        return ((2, YELLOW), (0, RED), (1, GREEN))
    if state == 2:
        # After yellow: red -> green -> yellow -> ...
        return ((0, RED), (1, GREEN), (2, YELLOW))
    # After red: green -> yellow -> red -> ...
    return ((1, GREEN), (2, YELLOW), (0, RED))


def _future_state(state, remaining, t):
    """
    Predict the signal state at future time t seconds from the current step.

    If t is infinite or invalid, return the current state.
    """
    state = int(round(float(state)))
    if state not in (0, 1, 2):
        state = 0

    remaining = max(0.0, float(remaining))

    if not np.isfinite(t):
        return state

    if t <= remaining + EPS:
        return state

    elapsed = (float(t) - remaining) % CYCLE

    for next_state, duration in _sequence_after(state):
        if elapsed < duration - EPS:
            return next_state
        elapsed -= duration

    # If precision lands exactly on a boundary, advance to the next phase.
    return _sequence_after(state)[0][0]


def _time_to_next_green(state, remaining):
    """
    Time until the next green interval starts, measured from the current step.
    """
    state = int(round(float(state)))
    if state not in (0, 1, 2):
        state = 0

    remaining = max(0.0, float(remaining))

    if state == 0:
        # Red: green starts when this red phase ends.
        return remaining
    if state == 2:
        # Yellow: red follows, then green.
        return remaining + RED
    if state == 1:
        # Green: yellow follows, then red, then green.
        return remaining + YELLOW + RED

    return remaining


def _earliest_arrival_time(distance, speed, speed_limit, max_accel):
    """
    Estimate the earliest possible time to reach a stop line if the ego
    accelerates as aggressively as allowed, without exceeding speed_limit.

    This is a look-ahead estimate only. The actual action is re-evaluated each
    1 s step.
    """
    distance = max(0.0, float(distance))
    speed = max(0.0, float(speed))
    speed_limit = max(0.0, min(float(speed_limit), V_CAP))
    max_accel = max(0.0, min(float(max_accel), A_MAX))

    if distance <= EPS:
        return 0.0

    if speed_limit <= speed + EPS:
        if speed <= EPS:
            return float("inf")
        return distance / speed

    if max_accel <= EPS:
        if speed <= EPS:
            return float("inf")
        return distance / speed

    t_acc = (speed_limit - speed) / max_accel
    d_acc = speed * t_acc + 0.5 * max_accel * t_acc * t_acc

    if d_acc >= distance:
        discriminant = speed * speed + 2.0 * max_accel * distance
        discriminant = max(0.0, discriminant)
        return (-speed + np.sqrt(discriminant)) / max_accel

    return t_acc + (distance - d_acc) / speed_limit


def _required_deceleration_to_speed_over_distance(speed, target_speed, distance):
    """
    Required acceleration, usually negative, to reduce speed to target_speed
    over the given distance.

    Returns a value in [A_MIN, 0.0].
    """
    speed = max(0.0, float(speed))
    target_speed = max(0.0, float(target_speed))
    distance = max(0.0, float(distance))

    if distance <= EPS:
        if speed > target_speed + EPS:
            return A_MIN
        return 0.0

    if speed <= target_speed + EPS:
        return 0.0

    required = (target_speed * target_speed - speed * speed) / (2.0 * distance)
    required = max(A_MIN, min(0.0, required))
    return required


# -----------------------------------------------------------------------------
# Driving prior
# -----------------------------------------------------------------------------

def _compute_driving_action(row):
    """
    Compute one physical driving action for one raw observation row.

    Returns requested acceleration in m/s^2, within [-1.5, 0.7].
    """
    a_obs = float(row[4])
    ego_speed = max(0.0, float(row[5]))
    lead_accel = float(row[6])
    lead_speed = max(0.0, float(row[7]))
    gap = float(row[8])

    light_state = int(round(float(row[9])))
    if light_state not in (0, 1, 2):
        light_state = 0

    light_dist = float(row[10])
    phase_remaining = max(0.0, min(float(row[11]), CYCLE))

    current_limit = float(row[12])
    next_limit = float(row[13])
    dist_to_next_region = float(row[14])

    # Speed limits: if an observation is invalid or missing, fall back safely.
    if current_limit <= 0.0:
        current_limit = 60.0 / 3.6
    if next_limit <= 0.0:
        next_limit = current_limit

    current_limit = max(0.0, min(current_limit, V_CAP))
    next_limit = max(0.0, min(next_limit, V_CAP))

    # Hard vehicle cap, separate from posted limit.
    speed_limit_step = min(current_limit, V_CAP)

    # If approaching a lower posted-speed region, prepare for that limit.
    approaching_lower_region = (
        next_limit < current_limit
        and 0.0 <= dist_to_next_region < SENTINEL_THRESHOLD
    )

    if approaching_lower_region and dist_to_next_region <= max(DT * ego_speed, DT) + 1.0:
        speed_limit_step = min(speed_limit_step, next_limit)

    # Gap handling.
    gap_eff = max(0.0, gap)

    # Following distance definitions from the documented reward/task description.
    d_safe = 2.0 + 0.5 * ego_speed + 0.0625 * ego_speed * ego_speed
    d_max = 10.0 + ego_speed + 0.0825 * ego_speed * ego_speed

    # Target gap inside the safe/prescribed band.
    target_gap = d_safe + 0.5 * (d_max - d_safe)
    gap_error = gap_eff - target_gap

    # Simple gap-based speed target.
    # If gap is too large, target speed rises above lead speed to close gap.
    # If gap is too small, target speed falls below lead speed.
    desired_speed = lead_speed + 0.2 * gap_error
    desired_speed = max(0.0, desired_speed)
    desired_speed = min(desired_speed, speed_limit_step)

    # ---------------------------------------------------------------------
    # Traffic-light prior
    # ---------------------------------------------------------------------
    signal_target_speed = None
    signal_upper_speed = None
    signal_urgent = False

    valid_signal = not (
        light_dist >= SENTINEL_THRESHOLD
        or light_dist < -1.0
    )

    if valid_signal:
        stop_line_dist = max(0.0, light_dist)

        if stop_line_dist <= EPS:
            # At the stop line. If not green, do not advance.
            if light_state != 1:
                signal_upper_speed = 0.0
                if ego_speed > EPS:
                    signal_urgent = True
        else:
            t_arrival_current = (
                stop_line_dist / max(ego_speed, EPS)
                if ego_speed > EPS
                else float("inf")
            )

            predicted_state = _future_state(
                light_state,
                phase_remaining,
                t_arrival_current,
            )

            time_to_next_green = _time_to_next_green(light_state, phase_remaining)
            earliest_arrival = _earliest_arrival_time(
                stop_line_dist,
                ego_speed,
                speed_limit_step,
                A_MAX,
            )

            if light_state == 1:
                # Current phase is green.
                green_end = phase_remaining

                # If current speed already reaches the line before green ends,
                # no additional signal action is needed.
                if predicted_state == 1 and t_arrival_current < green_end - SIGNAL_CROSS_MARGIN:
                    pass

                # If current speed is too slow but acceleration can still reach
                # before green ends, target a speed that crosses on green.
                elif earliest_arrival < green_end - SIGNAL_CROSS_MARGIN:
                    target_cross_speed = stop_line_dist / max(
                        green_end - SIGNAL_CROSS_MARGIN,
                        EPS,
                    )
                    signal_target_speed = min(target_cross_speed, speed_limit_step)

                # Otherwise, cannot safely cross on this green. Delay until the
                # next green interval.
                else:
                    wait_time = phase_remaining + YELLOW + RED
                    upper_speed = stop_line_dist / max(
                        wait_time + SIGNAL_WAIT_MARGIN,
                        EPS,
                    )
                    upper_speed = min(upper_speed, speed_limit_step)
                    signal_upper_speed = upper_speed

                    if upper_speed < ego_speed - EPS:
                        signal_urgent = True

            else:
                # Current phase is red or yellow. Crossing before the next green
                # is forbidden.
                if earliest_arrival < time_to_next_green + SIGNAL_WAIT_MARGIN:
                    upper_speed = stop_line_dist / max(
                        time_to_next_green + SIGNAL_WAIT_MARGIN,
                        EPS,
                    )
                    upper_speed = min(upper_speed, speed_limit_step)
                    signal_upper_speed = upper_speed

                    if upper_speed < ego_speed - EPS:
                        signal_urgent = True

    # Apply signal target/upper bounds to the desired speed.
    if signal_target_speed is not None:
        desired_speed = max(desired_speed, signal_target_speed)

    if signal_upper_speed is not None:
        desired_speed = min(desired_speed, signal_upper_speed)

    desired_speed = max(0.0, min(desired_speed, speed_limit_step))

    # Desired acceleration from the desired speed, assuming one 1 s step.
    desired_accel = desired_speed - ego_speed

    # ---------------------------------------------------------------------
    # Predictive following safety bounds
    # ---------------------------------------------------------------------
    relative_speed = lead_speed - ego_speed

    safe_accel_upper = float("inf")
    for h in (1.0, 2.0, 3.0):
        # If ego acceleration is greater than this bound, the predicted gap may
        # fall below D_safe over horizon h.
        bound = lead_accel + 2.0 * (gap_eff + h * relative_speed - d_safe) / (h * h)
        safe_accel_upper = min(safe_accel_upper, bound)

    if not np.isfinite(safe_accel_upper):
        safe_accel_upper = A_MAX

    # Entry-speed braking for a lower posted-speed region.
    if approaching_lower_region:
        entry_decel = _required_deceleration_to_speed_over_distance(
            ego_speed,
            next_limit,
            dist_to_next_region,
        )
        safe_accel_upper = min(safe_accel_upper, entry_decel)

    # Do not accelerate above the current posted limit or hard vehicle cap.
    safe_accel_upper = min(safe_accel_upper, speed_limit_step - ego_speed)
    safe_accel_upper = min(safe_accel_upper, V_CAP - ego_speed)

    # Lower bound used only when the gap is too large and closing it is safe.
    close_accel_lower = -float("inf")
    if gap_eff < SENTINEL_THRESHOLD:
        for h in (1.0, 2.0, 3.0):
            # If ego acceleration is below this bound, the predicted gap may
            # remain or become larger than D_max over horizon h.
            bound = lead_accel + 2.0 * (gap_eff + h * relative_speed - d_max) / (h * h)
            close_accel_lower = max(close_accel_lower, bound)

    if not np.isfinite(close_accel_lower):
        close_accel_lower = A_MIN

    accel_cmd = desired_accel

    # Safety and speed-limit constraints have priority.
    if accel_cmd > safe_accel_upper:
        accel_cmd = safe_accel_upper

    # If the gap is too large and closing it does not violate any upper bound,
    # encourage closing.
    if gap_eff > d_max and close_accel_lower > accel_cmd and close_accel_lower <= safe_accel_upper:
        accel_cmd = close_accel_lower

    accel_cmd = max(A_MIN, min(A_MAX, accel_cmd))

    # ---------------------------------------------------------------------
    # Optional comfort smoothing when no urgent driving constraint is active
    # ---------------------------------------------------------------------
    if (
        not signal_urgent
        and not approaching_lower_region
        and gap_eff >= d_safe * 1.05
        and gap_eff <= d_max
        and ego_speed < speed_limit_step - 0.5
    ):
        comfort_band = 0.25

        accel_cmd = max(accel_cmd, a_obs - comfort_band)
        accel_cmd = min(accel_cmd, a_obs + comfort_band)

        # Re-apply safety/limit constraints after smoothing.
        accel_cmd = min(accel_cmd, safe_accel_upper)
        accel_cmd = max(A_MIN, min(A_MAX, accel_cmd))

    return float(accel_cmd)


# -----------------------------------------------------------------------------
# Energy prior
# -----------------------------------------------------------------------------

def _compute_energy_action(row, driving_action):
    """
    Compute one physical energy action for one raw observation row.

    Returns requested FCS output power in kW, within [0, 50].

    The driving action is used internally as the current requested acceleration
    estimate for estimating traction demand. This is a rule-internal estimate,
    not an observed EDU/FCS/battery power value.
    """
    soc = float(row[0])
    battery_soh = float(row[1])
    fcs_soh = float(row[2])
    edu_soh = float(row[3])

    ego_speed = max(0.0, float(row[5]))

    # Clamp health/SOC values for the heuristic.
    soc = max(0.0, min(1.0, soc))
    battery_soh = max(0.0, min(1.0, battery_soh))
    fcs_soh = max(0.0, min(1.0, fcs_soh))
    edu_soh = max(0.0, min(1.0, edu_soh))

    # Use the rule's requested driving acceleration as the current acceleration
    # estimate for power demand. The environment may still clip it.
    accel_est = max(A_MIN, min(A_MAX, float(driving_action)))

    # Approximate mechanical road power:
    # P_road = F_road * v
    # F_road = rotational-mass-corrected acceleration force + drag + rolling
    road_force = M_EFF * accel_est + DRAG_COEFF * ego_speed * ego_speed + ROLL_RES
    p_mech = road_force * ego_speed

    # Approximate electrical EDU demand from mechanical road power.
    # This uses an assumed combined EDU boundary efficiency and is only a rough
    # estimate because the actual EDU map, torque, speed, and loss are not in
    # the observation.
    p_demand = max(0.0, p_mech / EDU_EFF_ASSUMED)

    # SOC feedback:
    # If SOC is below target, request more FCS power.
    # If SOC is above target, request less FCS power.
    p_soc = SOC_GAIN * (SOC_TARGET - soc)

    if p_mech <= 0.0:
        # During braking/regen, do not request FCS power merely to oppose
        # regeneration unless SOC is low enough to need charging.
        p_target = max(0.0, p_soc)
    else:
        p_target = p_demand + p_soc

    # If battery SOH is low and SOC is not high, prefer FCS supply over
    # additional battery discharge.
    if battery_soh < 0.95 and soc < 0.55:
        p_target += 5.0

    # If FCS SOH is low, avoid very high FCS requests unless necessary for SOC
    # or load. The cap is heuristic and does not model the actual aging model.
    fcs_cap = FCS_MAX
    if fcs_soh < 0.9:
        fcs_cap = max(
            20.0,
            FCS_MAX * max(0.5, min(1.0, fcs_soh)),
        )

    p_target = min(p_target, fcs_cap)

    # If SOC is very low, ensure a minimum charging request if possible.
    if soc < 0.30:
        p_target = max(p_target, min(20.0, fcs_cap))

    # If SOC is below 0.4, avoid very low FCS requests when charging is useful.
    if soc < 0.40 and p_target < 10.0:
        p_target = max(p_target, min(10.0, fcs_cap))

    # If SOC is high and load is small, allow regen/battery discharge and keep
    # FCS off.
    if soc > 0.60 and p_demand < 5.0:
        p_target = 0.0

    # Avoid tiny nonzero FCS requests near SOC target during braking/low load.
    if p_target < 1.0 and abs(soc - SOC_TARGET) < 0.02 and p_mech <= 0.0:
        p_target = 0.0

    p_target = max(0.0, min(FCS_MAX, p_target))

    return float(p_target)


# -----------------------------------------------------------------------------
# Required predict interface
# -----------------------------------------------------------------------------

def predict(observation):
    """
    Return physical action priors for both MADDPG actors.

    Input:
      observation: raw unnormalized NumPy array shaped (15,) or (batch, 15).

    Output:
      {
        "driving": {"action": (batch, 1) float32 acceleration m/s^2,
                    "active": (batch, 1) bool},
        "energy":  {"action": (batch, 1) float32 FCS request kW,
                    "active": (batch, 1) bool}
      }
    """
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        if x.size == 0:
            x = x.reshape(0, 15)
        else:
            x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15:
        raise ValueError(
            "observation must be finite with shape (15,) or (batch, 15)"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "observation must contain only finite values"
        )

    batch = x.shape[0]

    driving_action = np.zeros((batch, 1), dtype=np.float32)
    energy_action = np.zeros((batch, 1), dtype=np.float32)

    for i in range(batch):
        a_cmd = _compute_driving_action(x[i])
        p_cmd = _compute_energy_action(x[i], a_cmd)

        driving_action[i, 0] = np.float32(a_cmd)
        energy_action[i, 0] = np.float32(p_cmd)

    active = np.ones((batch, 1), dtype=bool)

    return {
        "driving": {
            "action": driving_action,
            "active": active,
        },
        "energy": {
            "action": energy_action,
            "active": active,
        },
    }
```

---

## Explanation of the rule behavior

### 1. Interface and output contract

The module defines `predict(observation)` as required.

It accepts:

```python
observation.shape == (15,)
```

or

```python
observation.shape == (batch, 15)
```

It validates that the observation is finite and has 15 raw dimensions.

For a single raw observation, it returns arrays shaped `(1, 1)`.

For a batched observation, it returns arrays shaped `(batch, 1)`.

The required output is:

```python
{
    "driving": {"action": acceleration_m_s2, "active": driving_mask},
    "energy":  {"action": fcs_power_kw, "active": energy_mask},
}
```

Both masks are always `True`:

```python
active = np.ones((batch, 1), dtype=bool)
```

This satisfies the requirement that the rule be evaluated and fused at every step for both actors.

The returned actions are already in physical units:

- `driving.action`: requested acceleration in `m/s²`, clipped to `[-1.5, 0.7]`.
- `energy.action`: requested FCS output power in `kW`, clipped to `[0, 50]`.

The rule does not return normalized actions, does not return `std`, does not add exploration noise, and does not hard-code `lambda`.

---

## 2. Driving prior

The driving prior uses the raw observation indices:

| Index | Meaning |
|---:|---|
| 4 | current ego acceleration |
| 5 | current ego speed |
| 6 | current lead acceleration |
| 7 | current lead speed |
| 8 | ego-to-lead distance |
| 9 | next traffic-light state |
| 10 | distance to next stop line |
| 11 | remaining time in current signal phase |
| 12 | current posted speed limit |
| 13 | next speed-region limit |
| 14 | distance to next speed-region entrance |

The driving prior is not a full MADDPG replacement. It is a soft prior intended to bias the learned actor toward safe following, speed-limit compliance, and legal traffic-light behavior.

---

### 2.1 Speed-limit handling

The rule uses the current posted limit from index 12.

It also watches the next speed-region limit from index 13 and the distance to that region from index 14.

If the next limit is lower than the current limit, the rule treats the vehicle as approaching a lower-speed region. It then applies an entry-speed braking estimate:

```python
required_deceleration = (v_next_limit^2 - v_current^2) / (2 * d_entry)
```

clipped to the physical acceleration range `[-1.5, 0.7]`.

This is intended to reduce the predicted entry speed into the lower-limit region. It does not reproduce the environment’s exact β₈ speed-entry shaping formula, but it supports the same objective: avoid entering a lower-speed region too fast.

If the next lower-limit region is very close, the rule also lowers the next-step speed target to the next region’s limit. This is a conservative look-ahead, not a bypass of the environment.

The hard vehicle cap of 70 km/h is also used as an upper acceleration bound:

```python
safe_accel_upper = min(safe_accel_upper, V_CAP - ego_speed)
```

The rule never outputs acceleration above `+0.7 m/s²` or below `-1.5 m/s²`.

The rule does not add a low-speed threshold that manually changes negative acceleration to zero. If the ego is stopped and the rule outputs negative acceleration, the environment’s no-reverse clipping handles it.

---

### 2.2 Following-distance prior

The rule computes the documented following-distance bands:

```python
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

using the current ego speed `v`.

It defines a target gap inside the safe band:

```python
target_gap = D_safe + 0.5 * (D_max - D_safe)
```

Then it forms a gap-based desired speed:

```python
desired_speed = lead_speed + 0.2 * (gap - target_gap)
```

Behavior:

- If the gap is too small, desired speed falls below lead speed.
- If the gap is in the safe band, desired speed tracks the lead speed.
- If the gap is too large, desired speed rises above lead speed to close the gap.

This desired speed is then limited by:

- current posted speed limit,
- hard 70 km/h vehicle cap,
- traffic-light constraints,
- predictive following safety bounds.

---

### 2.3 Predictive following safety bounds

The rule uses a short look-ahead over horizons of 1, 2, and 3 seconds.

For each horizon `h`, it estimates the future gap assuming current ego and lead accelerations are approximately constant over that horizon:

```python
gap_h ≈ gap + h * (v_lead - v_ego)
        + 0.5 * (a_lead - a_ego) * h^2
```

It then computes an upper bound on ego acceleration to avoid dropping below `D_safe`:

```python
a_ego <= a_lead + 2 * (gap + h * (v_lead - v_ego) - D_safe) / h^2
```

The most restrictive of these bounds is applied.

When the gap is larger than `D_max`, the rule also computes a lower bound that encourages closing the gap:

```python
a_ego >= a_lead + 2 * (gap + h * (v_lead - v_ego) - D_max) / h^2
```

This closing bound is applied only if it does not violate the more restrictive safety, speed-limit, or signal constraints.

The look-ahead is not an open-loop action sequence. The rule is recomputed at every 1 s step from the new observation.

---

### 2.4 Traffic-light prior

The traffic-light prior uses:

- signal state: index 9,
- distance to stop line: index 10,
- remaining time in current phase: index 11.

The sentinel value `30000 m` is treated as “no visible next traffic light” when the distance is near the maximum sentinel range. A distance of `0 m` is treated as a valid stop-line observation.

The rule estimates arrival time using the current speed:

```python
t1 = distance / max(v_ego, EPS)
```

If the ego speed is zero, division by zero is avoided.

It also estimates the earliest possible arrival time if the ego accelerates as strongly as allowed, subject to the current speed limit:

```python
earliest_arrival = earliest_arrival_time(distance, ego_speed, speed_limit, 0.7)
```

This is used to decide whether the vehicle can realistically cross during the current green interval.

---

### 2.5 Green-light logic

If the current phase is green:

1. If the current speed already reaches the stop line before the green phase ends, the rule does not impose extra signal braking.

2. If the current speed is too slow but acceleration can still reach before the green phase ends, the rule creates a target speed that attempts to cross on green.

3. If crossing on the current green is not feasible, the rule delays crossing until the next green interval. It does this by limiting the speed so that the estimated arrival occurs after the next green start plus a safety margin.

The next green start time is computed from the current phase:

- current green: next green after `remaining + 3 s yellow + 47 s red`
- current yellow: next green after `remaining + 47 s red`
- current red: next green after `remaining`

The rule uses margins:

```python
SIGNAL_CROSS_MARGIN = 0.5 s
SIGNAL_WAIT_MARGIN = 1.0 s
```

These margins are conservative design choices to avoid borderline yellow crossings.

---

### 2.6 Red/yellow logic

If the current phase is red or yellow, crossing is forbidden.

The rule estimates whether the ego can reach the stop line before the next green starts. If the earliest possible arrival would occur before the next green plus a safety margin, the rule limits the ego speed so that the predicted arrival is delayed until after the next green.

If the ego is already at the stop line and the signal is red or yellow, the signal upper speed is set to zero. If the ego is moving, this produces braking. If the ego is stopped, the environment’s no-reverse handling applies.

The rule does not treat a red light as a zero speed limit. It treats red/yellow as a stop-line crossing constraint, while the posted speed limit remains the active regional speed limit.

---

### 2.7 Comfort smoothing

The rule includes a limited comfort-smoothing term.

It is applied only when:

- no urgent traffic-light constraint is active,
- the vehicle is not approaching a lower speed region,
- the gap is not too close,
- the gap is not excessively large,
- the ego is not near the current speed limit.

When applied, it limits acceleration changes relative to the observed current acceleration by a small band:

```python
comfort_band = 0.25 m/s^2
```

After smoothing, the rule re-applies safety, speed-limit, and signal constraints.

This smoothing is optional and conservative. It is intended to support the comfort objective, but it never overrides urgent safety or signal constraints.

---

## 3. Energy prior

The energy prior uses raw observation indices:

| Index | Meaning |
|---:|---|
| 0 | battery SOC |
| 1 | battery SOH |
| 2 | FCS SOH |
| 3 | EDU SOH |
| 4 | current ego acceleration |
| 5 | current ego speed |

It also uses the driving action computed internally by the rule as the current requested acceleration estimate for power-demand estimation.

This is a rule-internal estimate. It is not an observed EDU power, battery power, hydrogen consumption, or actual FCS output.

---

### 3.1 Approximate traction-power estimate

The rule estimates mechanical road power from longitudinal dynamics:

```python
road_force = M_eff * a + drag_force + rolling_force
p_mech = road_force * v
```

where:

```python
M_eff = 7000 * 1.05 = 7350 kg
drag_force = 0.5 * rho * Cd * A * v^2
rolling_force = Crr * M * g
```

This estimate is approximate because the actual EDU torque, speed, operating point, efficiency, and losses are not in the observation.

---

### 3.2 Approximate EDU electrical demand

The rule converts estimated mechanical power to an approximate electrical EDU demand:

```python
p_demand = max(0, p_mech / EDU_EFF_ASSUMED)
```

with

```python
EDU_EFF_ASSUMED = 0.90
```

This is not a rated efficiency. It is not a peak efficiency. It is only a rough combined EDU boundary efficiency assumption used because the actual EDU map is not directly observable from the rule interface.

The actual EDU map represents the combined motor, inverter, and gearbox boundary. The rule does not add separate inverter or gearbox losses. It only uses a single approximate combined efficiency factor as a simplified boundary estimate.

---

### 3.3 SOC feedback

The rule applies SOC feedback:

```python
p_soc = SOC_GAIN * (0.5 - SOC)
```

with:

```python
SOC_GAIN = 80 kW per unit SOC error
```

Behavior:

- If SOC is below 0.5, the rule requests more FCS power.
- If SOC is above 0.5, the rule requests less FCS power.
- If SOC is very low, the rule enforces a minimum charging request.
- If SOC is high and the load is small, the rule allows FCS power to go to zero so that regen or battery discharge can reduce SOC.

The SOC correction is a heuristic. It does not directly model battery current, battery voltage, charge/discharge rates, or cell electrothermal state.

---

### 3.4 Braking and regen handling

If the estimated mechanical power is negative, the ego is braking.

In that case, the rule does not request FCS power merely to oppose regeneration. It only requests FCS power if SOC is low enough to need charging.

This supports energy recovery when possible while still regulating SOC.

---

### 3.5 SOH-aware heuristics

The rule includes simple SOH-aware adjustments.

#### Battery SOH

If battery SOH is below 0.95 and SOC is not high, the rule slightly increases the FCS request. The intent is to reduce additional battery discharge when battery health is degraded.

This is a heuristic. The rule does not observe actual battery current, voltage, or SOH-loss increment.

#### FCS SOH

If FCS SOH is below 0.9, the rule caps high FCS requests.

The cap is heuristic:

```python
fcs_cap = max(20 kW, 50 kW * max(0.5, min(1.0, FCS_SOH)))
```

This is intended to reduce high-power FCS operation when FCS health is degraded. It does not reproduce the actual FCS aging model.

#### EDU SOH

The rule reads EDU SOH but does not directly use it to alter FCS request.

This is intentional. The energy action changes the requested FCS power, not the EDU mechanical demand. EDU degradation is primarily driven by EDU torque, speed, and mechanical power, which are not directly observable by the rule and are not directly controlled by the FCS request. The rule therefore does not pretend to control EDU aging through the FCS request.

---

### 3.6 Low-load and startup heuristics

The rule avoids very small nonzero FCS requests when SOC is near target and the vehicle is braking or under low load.

This is intended to reduce unnecessary FCS startup or low-power operation, but it is only a heuristic. The rule does not observe startup events, FCS load-change history, or actual degradation increments.

---

## 4. Physical-action fusion and CLI

The rule outputs physical actions:

```python
driving.action: m/s^2
energy.action: kW
```

The training adapter is expected to:

1. Convert each MADDPG actor’s normalized action to physical action.
2. Blend the MADDPG physical action with the rule physical action using the externally supplied `lambda`.
3. Map the fused physical action back to the normalized joint action used by replay and centralized critics.

For driving:

```python
u_d = tanh(z_d)

if u_d < 0:
    a_maddpg = 1.5 * u_d
else:
    a_maddpg = 0.7 * u_d

a_fused = (1 - lambda) * a_maddpg + lambda * a_rule
```

For energy:

```python
u_e = tanh(z_e)

p_maddpg = 25 * (u_e + 1)  # kW

p_fused = (1 - lambda) * p_maddpg + lambda * p_rule
```

The rule does not know or hard-code `lambda`.

Recommended CLI selection:

```bash
--prior-program-name fchev_follow_energy_rule
--maddpg-rule-output-mode physical_action
--maddpg-rule-lambda <value>
```

where:

```text
lambda ∈ [0, 1]
```

A value of `lambda = 0` makes the rule inactive in fusion. A value of `lambda = 1` makes the rule the full prior before environment constraints, although the environment may still clip the executed action.

---

## 5. Relationship to the stated task objectives

### Driving objectives

The driving prior supports:

- following the lead vehicle,
- keeping gap at least `D_safe`,
- avoiding unnecessarily large gaps above `D_max` when feasible,
- respecting current posted speed limits,
- preparing for lower posted-speed regions,
- crossing stop lines only when the predicted crossing state is green,
- delaying or braking before red/yellow phases,
- using short-horizon look-ahead rather than a fixed open-loop maneuver.

It does not mechanically reproduce the reward function. It uses the reward structure only as motivation for the priority ordering.

---

### Energy objectives

The energy prior supports:

- reducing unnecessary FCS power requests,
- keeping battery SOC near 0.5,
- reducing battery discharge when SOC is high,
- increasing FCS charging when SOC is low,
- avoiding very high FCS requests when FCS SOH is low,
- allowing regen/braking energy to reduce SOC when SOC is high,
- avoiding tiny unnecessary FCS requests near SOC target during braking.

It does not directly compute hydrogen consumption, FCS degradation increments, battery SOH loss, EDU degradation, or actual battery power. Those quantities are not directly available in the observation.

---

## 6. Important limitations

1. **No direct EDU torque or power observation**  
   The rule estimates traction demand from speed, acceleration, drag, and rolling resistance. It does not observe motor torque, EDU operating point, or instantaneous EDU loss.

2. **Approximate EDU efficiency**  
   The assumed combined EDU efficiency of 0.90 is only a rough boundary estimate. The actual EDU map is not used because the rule receives only the 15D observation.

3. **No direct FCS, battery, or hydrogen power observation**  
   The rule does not observe requested FCS power, actual FCS power, battery power, hydrogen consumption, or degradation increments.

4. **No history state**  
   The rule is stateless across steps. It uses only the current observation and its own current internal estimates. It does not remember previous rule actions, previous FCS power, previous SOC rate, or previous degradation history.

5. **EDU degradation is not directly controllable through FCS request**  
   The energy action changes requested FCS power. EDU degradation depends primarily on EDU mechanical operating point. The rule cannot directly reduce EDU degradation through the FCS request.

6. **Look-ahead is approximate**  
   The predictive bounds assume approximate constant accelerations over short horizons. Actual SUMO lead behavior, traffic interactions, and powertrain clipping may dif## Suggested program name and path

Use the filename:

```text
llm-prior/programs/fcbev_follow_energy_prior.py
```

The CLI program name would then be:

```bash
--prior-program-name fcbev_follow_energy_prior
```

with the required physical-action mode and externally supplied blend value, for example:

```bash
--maddpg-rule-output-mode physical_action --maddpg-rule-lambda 0.3
```

The value of `lambda` is not hard-coded in the rule.

---

## Complete proposed Python module

```python
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
```

---

## Behavior and design explanation

### 1. Interface and output mode

The program is intended to be used as a physical-action rule prior:

```bash
--prior-program-name fcbev_follow_energy_prior
--maddpg-rule-output-mode physical_action
--maddpg-rule-lambda <lambda_value>
```

The rule receives the complete raw, unnormalized 15-dimensional observation. It accepts either:

```python
observation.shape == (15,)
```

or:

```python
observation.shape == (batch, 15)
```

It validates that the observation is finite and has the required shape. It returns two scalar physical actions per row:

```python
driving.action  # requested acceleration, m/s^2
energy.action   # requested FCS output power, kW
```

Both actions are returned as float32 arrays with shape `(batch, 1)`.

Both masks are always active:

```python
driving.active = True
energy.active = True
```

for every row and every step.

The rule does not output normalized MADDPG actions, does not output `std`, does not add exploration noise, and does not hard-code the fusion coefficient `lambda`.

---

## 2. Driving prior behavior

The driving prior uses the raw observation indices:

```text
4  ego acceleration
5  ego speed
6  lead acceleration
7  lead speed
8  ego-to-lead gap
9  next traffic-light state
10 distance to next traffic-light stop line
11 remaining time in current signal phase
12 current posted speed limit
13 next posted speed limit
14 distance to entrance of next speed region
```

It does not assume a fixed lead-vehicle profile. It uses the live lead speed and lead acceleration at every step.

### Following-distance logic

The rule computes the same style of following bounds used by the environment reward model:

```text
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

It uses a target gap near the middle of the safe and prescribed range:

```text
target_gap = D_safe + 0.5 (D_max - D_safe)
```

The base following acceleration is:

```text
a_follow =
    lead_acceleration
    + k_gap (gap - target_gap)
    + k_rel (lead_speed - ego_speed)
```

with modest gains:

```text
k_gap = 0.03
k_rel = 0.35
```

This gives a simple look-ahead following controller:

- If the gap is too small, the rule requests deceleration.
- If the gap is too large, the rule requests acceleration.
- If the ego is faster than the lead, it requests deceleration.
- If the lead is faster than the ego, it requests acceleration, subject to speed and traffic constraints.
- The lead acceleration is used as a feedforward term.

If the gap is already below `D_safe` and the ego is not clearly slower than the lead, the rule prevents positive acceleration. This avoids worsening an unsafe following gap when the ego is already close to the lead.

The rule does not abandon following merely because a manageable risk is predicted. It first computes a following-oriented acceleration, then applies safety and road-rule caps.

---

### Speed-limit logic

The rule enforces the current posted speed limit through a 1-step acceleration cap:

```text
allowed = v_limit - ego_speed
a_cap = min(0.7, allowed)   if allowed >= 0
a_cap = max(-1.5, allowed)  if allowed < 0
```

This is not a target-speed command. It only limits the requested acceleration so that the next-step speed is not allowed to exceed the current posted limit.

The rule also uses the next speed-region information:

```text
13 next speed limit
14 distance to next speed-region entrance
```

When the next region has a lower posted limit, the rule applies a kinematic entry cap:

```text
v_entry^2 ~= ego_speed^2 + 2 a d_entry
```

and chooses a cap such that:

```text
v_entry <= next_limit
```

This gives a soft look-ahead braking prior before entering a lower-speed region. It does not reproduce the exact β₈ speed-entry shaping reward, but it supports the same behavioral objective.

The rule caps speed limits at the documented 70 km/h hard vehicle cap, but the actual posted limits in the task are 30 km/h and 60 km/h.

---

### Traffic-light logic

The traffic-light prior uses:

```text
9  signal state: red 0, green 1, yellow 2
10 distance to next stop line
11 remaining time in current phase
5  ego speed
```

It ignores the sentinel distance:

```text
30000 m means no visible target / outside effective range
```

A distance of `0 m` is treated as a valid stop-line observation.

The rule estimates arrival time as:

```text
t1 = distance / ego_speed
```

If the ego speed is zero or near zero, the rule avoids division by zero. In that case, it uses a conservative low-speed logic:

- If the current signal is green, it checks whether the ego could reach the stop line before the current green phase ends using the maximum acceleration `0.7 m/s^2`.
- If it cannot, it requests zero acceleration and waits.
- If the current signal is red or yellow, it requests zero acceleration while stopped.

For moving vehicles, the rule predicts the signal state at the estimated arrival time using the documented cycle:

```text
green 40 s, yellow 3 s, red 47 s, total 90 s
```

If the predicted crossing state is green, no traffic-light cap is imposed.

If the predicted crossing state is yellow or red, the rule imposes a non-positive acceleration cap. The cap is based on a conservative stop-before-line estimate:

```text
a_stop = -v^2 / (2 distance)
```

clipped to the allowed acceleration range.

The rule is deliberately conservative:

- It does not treat `t1 >= t2` as permission to accelerate through the intersection.
- It treats yellow as non-crossable.
- It treats red as non-crossable.
- It re-evaluates every step from the new observation.
- It does not hold an action open-loop.

If the ego is at the stop line and the signal is not clearly green, the rule requests negative acceleration if moving, or zero acceleration if already stopped. If the signal is green and the remaining phase time is positive, the rule allows crossing.

This traffic-light prior is a soft prior. It does not guarantee that the final fused and executed action will prevent a red or yellow crossing, because the final action is fused with the learned MADDPG action and then executed by the environment.

---

### Comfort smoothing

The rule uses the current ego acceleration index only for mild comfort smoothing in a safe steady-following band:

```text
gap is between D_safe and D_max
|ego_speed - lead_speed| <= 1 m/s
no traffic-light cap
ego is comfortably below the current speed limit
ego is not close to a lower-limit entry constraint
```

In that band, the requested acceleration is limited to change by at most `0.20 m/s^2` from the current executed acceleration.

This smoothing is not applied when:

- the gap is unsafe,
- the gap is excessively large,
- the ego is near or above a speed limit,
- the ego is approaching a lower-speed region,
- a traffic-light crossing constraint is active.

The rule does not implement a general low-speed override that converts a braking request to zero. Negative acceleration requests are allowed when the following logic demands them. The environment remains responsible for the no-reverse clipping rule.

---

## 3. Energy prior behavior

The energy prior returns a requested FCS output power in kW, not battery power.

It uses:

```text
0  battery SOC
1  battery SOH
2  FCS SOH
3  EDU SOH
5  ego speed
```

It also uses the driving prior’s requested acceleration as an estimated demand proxy. This is an approximation. The actual executed driving action after MADDPG fusion may differ from the rule’s driving prior.

---

### Approximate mechanical demand

The rule estimates mechanical power demand using simplified vehicle dynamics:

```text
effective mass = 7000 * 1.05
drag force     = 0.5 rho Cd A v^2
rolling force  = m_eff g C_rr
mechanical power = (m_eff a + drag force + rolling force) * v
```

It then divides by an approximate combined efficiency:

```text
ETA_EST = 0.88
```

This `ETA_EST` is only a rough combined EDU/DC/DC efficiency approximation. It is not the exact EPA Generic IPM 150 kW EDU map, not a rated efficiency, and not a claim about inverter/gearbox losses. It is used only to estimate whether the FCS should be asked to supply power.

If the estimated mechanical power is negative, the rule treats it as regeneration potential and prefers a low or zero FCS request.

---

### SOC management

The energy prior uses SOC target:

```text
SOC_target = 0.5
```

It applies a heuristic SOC correction:

```text
p = estimated_FCS_demand - 100 * (SOC - 0.5) kW
```

Behavior:

- If `SOC > 0.5`, requested FCS power is reduced.
- If `SOC < 0.5`, requested FCS power is increased.
- If SOC is near target, the correction is small.
- If SOC is high and demand is low, the rule requests zero FCS power.
- If SOC is low and demand is low, the rule requests a small charging power.
- If SOC is very low and components are reasonably healthy, the rule requests at least a modest charging power.
- If the ego is braking and SOC is not very low, the rule prefers zero FCS request to favor regeneration and lower hydrogen use.

This is SOC management, not a direct battery-degradation model.

---

### Component-health caps

The rule caps the requested FCS power based on SOH observations.

The physical maximum is:

```text
FCS_max = 50 kW
```

The rule applies heuristic caps:

```text
max_fcs <= 50 * max(0.5, FCS_SOH)
max_fcs <= 50 * max(0.5, EDU_SOH)
```

If SOC is already high:

```text
max_fcs <= 50 * max(0.5, battery_SOH)
```

This reduces the chance of requesting excess FCS power that could increase charging stress when battery health is reduced.

These caps are heuristic. They do not reproduce the exact FCS aging model, battery electrothermal aging model, EDU CLR aging model, battery current/voltage limits, EDU torque-speed validity, or exact energy-dispatch feasibility. The environment still enforces all physical limits and may clip the requested FCS power.

---

## 4. Physical-action fusion

The rule returns physical actions:

```text
driving.action = requested acceleration in [-1.5, 0.7] m/s^2
energy.action  = requested FCS output power in [0, 50] kW
```

The MADDPG actors produce normalized actions:

```text
u_d = tanh(z_d)
u_e = tanh(z_e)
```

The adapter maps them to physical units:

```text
if u_d < 0:
    a_MADDPG = 1.5 u_d
else:
    a_MADDPG = 0.7 u_d

P_MADDPG = 25 (u_e + 1) kW
```

Then the physical actions are blended independently:

```text
a_fused = (1 - lambda) a_MADDPG + lambda a_rule
P_fused = (1 - lambda) P_MADDPG + lambda P_rule
```

The rule does not control `lambda`. It only supplies `a_rule` and `P_rule`.

After fusion, the trainer maps the physical result back to normalized joint actions for replay and centralized critics. The environment then applies vehicle dynamics, no-reverse handling, speed cap, battery/FCS/EDU feasibility, and traffic/road termination logic.

---

## 5. Masks and per-step evaluation

The program returns active masks for both actors at every observation:

```python
driving.active = True
energy.active  = True
```

There is no inactive state. The rule is evaluated every step. It has no internal persistent state between steps except what can be inferred from the current observation, such as current ego acceleration for comfort smoothing.

This matches the requirement that each decision is recomputed from the newly observed state at the next 1 s step.

---

## 6. Limitations and unavailable information

This rule is a soft prior, not a validated controller and not a standalone supervisor.

Important limitations:

1. **No direct powertrain observables**  
   The observation does not include:
   - motor torque,
   - instantaneous EDU loss,
   - requested FCS power,
   - actual battery power,
   - hydrogen consumption,
   - FCS operating point,
   - EDU operating point,
   - battery current/voltage state,
   - exact component aging state.

   The energy prior estimates demand from vehicle dynamics and the rule’s driving prior. This estimate is approximate.

2. **Approximate energy model**  
   The energy prior uses a simplified vehicle-dynamics power estimate and a single approximate efficiency value. It does not use the exact EPA Generic IPM 150 kW EDU map, exact FCS hydrogen-consumption polynomial, exact battery electrothermal model, or exact energy-dispatch logic.

3. **Heuristic SOH caps**  
   FCS, EDU, and battery SOH caps are heuristic. They are intended to reduce requested high-power operation when health is low, but they do not reproduce the exact degradation models or physical feasibility limits.

4. **Driving-energy coupling is approximate**  
   The energy prior uses the rule’s driving prior acceleration as a demand estimate. The final executed driving action is fused with the learned MADDPG driving action and then clipped by the environment. Therefore, the actual mechanical demand may differ from the rule’s estimate.

5. **Conservative traffic-light behavior**  
   The traffic-light prior is conservative. If the current-speed arrival estimate predicts a yellow or red crossing, the rule requests a non-positive acceleration. In some cases, a more aggressive acceleration might have allowed a green crossing, but the rule follows the documented predicted-arrival logic and avoids risky yellow/red crossing attempts.

6. **Soft prior only**  
   The rule cannot guarantee safe following, green-light crossing, SOC regulation, or component-health behavior. The final behavior depends on MADDPG fusion, exploration noise during training, and the environment’s physical constraints.

7. **No reward reproduction**  
   The rule does not mechanically compute β₀ through β₈ or reproduce the exact reward terms. It uses the reward structure only as behavioral guidance.

8. **No bypassing of environment constraints**  
   The rule outputs requests within the documented action ranges, but the environment may still clip acceleration or FCS power due to speed, no-reverse, battery, FCS, EDU, or road-rule constraints.

---

## 7. CLI selection summary

Place the file at:

```text
llm-prior/programs/fcbev_follow_energy_prior.py
```

Select it with:

```bash
--prior-program-name fcbev_follow_energy_prior
```

Use physical-action rule mode:

```bash
--maddpg-rule-output-mode physical_action
```

Provide the external blend value:

```bash
--maddpg-rule-lambda <value>
```

where:

```text
0 <= lambda <= 1
```

The rule itself remains deterministic, returns both actor priors active at every step, and produces physical actions in the required units.fer.

7. **Fusion remains soft**  
   The rule is blended with MADDPG actions using external `lambda`. The final executed action may still differ from both the MADDPG action and the rule action because the environment enforces physical and powertrain constraints.

8. **No validation claim**  
   This module is a proposed rule program. It has not been executed or validated in SUMO in this response.