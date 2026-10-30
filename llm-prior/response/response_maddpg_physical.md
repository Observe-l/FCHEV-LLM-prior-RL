## A. Suggested executable rule-prior program

Suggested filename/path:

```text
llm-prior/programs/fchev_rule_prior.py
```

```python
# llm-prior/programs/fchev_rule_prior.py
# Deterministic FCHEV rule prior for fusion with MADDPG physical actions.
# Output mode: physical_action.
# This program is a soft prior only. It does not bypass environment clipping,
# battery/FCS/EDU feasibility limits, SUMO dynamics, or the hard 70 km/h cap.

import numpy as np


# ---------------------------------------------------------------------------
# Action and environment constants
# ---------------------------------------------------------------------------
DT = 1.0

# Physical driving action bounds after the project's tanh/action mapping.
A_POS = 0.7      # maximum requested acceleration, m/s^2
A_NEG = -1.5     # minimum requested acceleration, m/s^2

# Hard vehicle speed cap: 70 km/h = 19.4444 m/s.
V_HARD_MAX = 19.4444

# Main-road posted limit: 60 km/h = 16.6667 m/s.
V_MAIN_LIMIT = 16.6667

# SOC target and energy limits.
SOC_TARGET = 0.5
FCS_MAX_KW = 50.0

# EDU aging reference point from the project description: 75 kW mechanical power.
EDU_REF_POWER_W = 75000.0

# Traffic-light approach zone: the signal is used for immediate stop/proceed
# only from 100 m before the stop line through the stop line.
TRAFFIC_APPROACH = 100.0
YELLOW_DURATION = 3.0

# Sentinel used by the observation for "no visible target / outside range".
SENTINEL = 30000.0

# Safety / comfort margins.
SAFE_MARGIN = 2.0          # added to D_safe to keep a margin against the reward threshold
STOP_MARGIN = 1.0          # time margin for traffic-light decisions
ENTRY_MARGIN = 0.5         # speed margin before entering a lower-limit region
COMFORT_DELTA = 0.75       # maximum non-emergency acceleration change from current acceleration

EPS = 1e-6

# Vehicle constants from the project background.
MASS = 7000.0
ROT_MASS_COEFF = 1.05
M_EFF = MASS * ROT_MASS_COEFF

DRAG_K = 0.5 * 1.225 * 0.55 * 5.8
ROLL_FORCE = 0.0085 * MASS * 9.8


def _safe_gap(v):
    """Project following reward's D_safe, m."""
    return 2.0 + 0.5 * v + 0.0625 * v * v


def _max_gap(v):
    """Project following reward's D_max, m."""
    return 10.0 + v + 0.0825 * v * v


def predict(observation):
    """
    Deterministic FCHEV rule prior.

    Input
    -----
    observation : np.ndarray
        Raw, unnormalized 15-dimensional observation or batch of observations.
        Shapes: (15,) or (batch, 15).

    Output
    ------
    dict
        {
            "driving": {"action": acceleration_m_s2, "active": driving_mask},
            "energy":  {"action": fcs_power_kw,      "active": energy_mask},
        }

        action arrays are float32 with shape (batch, 1).
        active masks are bool with shape (batch, 1).
    """
    x = np.asarray(observation, dtype=np.float32)

    # Accept single observation or batch.
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15 or not np.all(np.isfinite(x)):
        raise ValueError("observation must be finite with shape (15,) or (batch, 15)")

    n = x.shape[0]

    # -----------------------------------------------------------------------
    # Raw observation unpacking.
    # -----------------------------------------------------------------------
    soc = x[:, 0]
    battery_soh = x[:, 1]
    fcs_soh = x[:, 2]
    edu_soh = x[:, 3]

    a_current = np.clip(x[:, 4], A_NEG, A_POS)
    v = np.clip(x[:, 5], 0.0, V_HARD_MAX)

    # Lead acceleration is used only for short-horizon prediction. Clip it to a
    # reasonable range to avoid numerical extremes; the environment remains the
    # authority on actual motion.
    a_lead = np.clip(x[:, 6], -2.0, 2.0)
    v_lead = np.clip(x[:, 7], 0.0, V_HARD_MAX)

    gap = x[:, 8]

    signal_state = x[:, 9]
    dist_light = x[:, 10]
    phase_time = x[:, 11]

    v_limit = x[:, 12]
    next_limit = x[:, 13]
    d_next = x[:, 14]

    # Speed-limit fallback. The observation normally provides 16.6667 or 8.3333.
    v_limit = np.where((v_limit > 0.0) & np.isfinite(v_limit), v_limit, V_MAIN_LIMIT)
    v_limit = np.clip(v_limit, 1.0, V_HARD_MAX)

    next_limit_valid = (
        (next_limit > 0.0)
        & np.isfinite(next_limit)
        & (d_next >= 0.0)
        & (d_next < SENTINEL)
    )

    # -----------------------------------------------------------------------
    # 1. Following constraint.
    #
    # One-step constant-acceleration prediction:
    #   d_pred = gap + (v_lead - v) * dt + 0.5 * (a_lead - a_req) * dt^2
    #
    # Solve for acceleration bounds that keep d_pred between D_safe and D_max.
    # This is a short-horizon approximation; actual lead behavior may change.
    # -----------------------------------------------------------------------
    v_ref = v  # Following reward thresholds use ego speed.
    d_safe = _safe_gap(v_ref) + SAFE_MARGIN
    d_max = _max_gap(v_ref)
    d_max = np.maximum(d_max, d_safe + 5.0)

    delta_v = v_lead - v

    a_follow_min = np.clip(
        a_lead + 2.0 * (gap + delta_v - d_max),
        A_NEG,
        A_POS,
    )
    a_follow_max = np.clip(
        a_lead + 2.0 * (gap + delta_v - d_safe),
        A_NEG,
        A_POS,
    )

    # Gap too small: predicted gap would violate safe following, or overlap is already present.
    gap_too_small = ((a_follow_max < -0.1) | (gap < 0.0)) & (gap < SENTINEL)

    # Gap too large: predicted gap would exceed D_max, or the gap is at the sentinel maximum.
    gap_too_large = (
        ((a_follow_min > 0.1) & (gap < SENTINEL))
        | (gap >= SENTINEL - 1.0)
    )

    # -----------------------------------------------------------------------
    # 2. Current posted speed-limit constraint.
    # -----------------------------------------------------------------------
    a_speed_max = np.clip((v_limit - v) / DT, A_NEG, A_POS)
    overspeed = v > v_limit + 0.1
    speed_limit_enforce = v > (v_limit - 0.25)

    # -----------------------------------------------------------------------
    # 3. Lower-limit entry shaping.
    #
    # The project's entry-shaping reward predicts entry speed using 0.75 m/s^2
    # braking. The rule uses the same prediction to request braking before a
    # lower-limit region if needed.
    # -----------------------------------------------------------------------
    approach_lower = next_limit_valid & (v_limit > next_limit + 0.1)

    v_entry = np.sqrt(np.maximum(0.0, v * v - 2.0 * 0.75 * d_next))
    entry_excess = approach_lower & (v_entry > next_limit + ENTRY_MARGIN)

    a_entry_required = np.clip(
        (next_limit * next_limit - v * v)
        / (2.0 * np.maximum(d_next, 1.0)),
        A_NEG,
        A_POS,
    )

    a_speed_region_max = np.where(
        entry_excess,
        np.minimum(a_speed_max, a_entry_required),
        a_speed_max,
    )
    speed_region_brake = entry_excess

    # -----------------------------------------------------------------------
    # 4. EDU degradation soft cap.
    #
    # Approximation: estimate traction mechanical power from vehicle dynamics
    # and limit acceleration so estimated mechanical power remains near the
    # EDU aging reference point of 75 kW, scaled by EDU SOH.
    #
    # This is not EDU electrical loss, inverter loss, gearbox loss, or motor
    # torque. The EDU map already represents a combined boundary in the project.
    # -----------------------------------------------------------------------
    f_resistance = DRAG_K * v * v + ROLL_FORCE
    edu_ref_power_w = EDU_REF_POWER_W * np.clip(edu_soh, 0.8, 1.0)

    a_edu_max = np.where(
        v > 0.1,
        (edu_ref_power_w / v - f_resistance) / M_EFF,
        A_POS,
    )
    a_edu_max = np.clip(a_edu_max, A_NEG, A_POS)
    edu_cap_binds = a_edu_max < (A_POS - 0.05)

    # -----------------------------------------------------------------------
    # 5. Traffic-light immediate decision.
    #
    # Signal phase is used only when the ego is in the 100 m approach zone or
    # at the stop line. Traffic-light distance 0 is valid. Sentinel 30000 m is
    # ignored.
    #
    # Cycle order:
    #   green -> yellow -> red -> green
    #
    # Red crossing terminates. Yellow crossing does not terminate but receives
    # negative traffic prediction reward in the project reward definition.
    # -----------------------------------------------------------------------
    traffic_approach = (
        (dist_light >= -1e-6)
        & (dist_light <= TRAFFIC_APPROACH)
        & (dist_light < SENTINEL)
        & (signal_state >= 0.0)
        & (signal_state <= 2.0)
        & (phase_time >= 0.0)
    )

    dist_safe = np.maximum(dist_light, 0.0)

    arrival_current = np.where(
        v > 0.1,
        dist_safe / np.maximum(v, EPS),
        np.inf,
    )

    # If stopped, use a conservative minimum arrival time assuming maximum
    # positive acceleration 0.7 m/s^2 from rest.
    min_arrival = np.where(
        dist_safe > 1e-3,
        np.sqrt(2.0 * dist_safe / A_POS),
        0.0,
    )

    arrival_decision = np.where(v > 0.1, arrival_current, min_arrival)

    stopping_dist = v * v / (2.0 * 0.75)
    can_stop_comfort = (
        (dist_safe > STOP_MARGIN)
        & (stopping_dist <= dist_safe - STOP_MARGIN)
    )

    red = signal_state < 0.5
    green = (signal_state >= 0.5) & (signal_state < 1.5)
    yellow = signal_state >= 1.5

    # Red: after red ends, the next phase is green. Stop if the ego would cross
    # while the signal is still red.
    red_cross_red = arrival_decision <= phase_time + STOP_MARGIN
    red_stop = traffic_approach & red & (red_cross_red | (dist_safe <= 1e-3))

    # Yellow: after yellow ends, the next phase is red. Stop if the ego cannot
    # safely cross before yellow ends. If it can cross yellow safely, stop only
    # when comfortable stopping is possible, to avoid negative yellow reward.
    yellow_must = traffic_approach & yellow & (arrival_decision > phase_time - STOP_MARGIN)
    yellow_optional = (
        traffic_approach
        & yellow
        & (arrival_decision <= phase_time - STOP_MARGIN)
        & can_stop_comfort
    )
    yellow_stop = yellow_must | yellow_optional

    # Green: after green ends, yellow then red. Stop if crossing would occur in
    # red. If crossing would occur in yellow, stop if comfortable.
    green_must = traffic_approach & green & (
        arrival_decision > phase_time + YELLOW_DURATION + STOP_MARGIN
    )
    green_optional = (
        traffic_approach
        & green
        & (arrival_decision > phase_time - STOP_MARGIN)
        & (arrival_decision <= phase_time + YELLOW_DURATION + STOP_MARGIN)
        & can_stop_comfort
    )
    green_stop = green_must | green_optional

    traffic_stop = red_stop | yellow_stop | green_stop

    a_stop_required = np.where(
        traffic_stop,
        np.clip(
            -v * v / (2.0 * np.maximum(dist_safe, 1.0)),
            A_NEG,
            0.0,
        ),
        A_POS,
    )

    # -----------------------------------------------------------------------
    # 6. Combine driving constraints.
    #
    # Safety, speed-limit, traffic-light, entry-shaping, and EDU cap constraints
    # are treated as upper bounds on acceleration. Following-gap closure is a
    # lower bound used only when higher-priority constraints are not binding.
    # -----------------------------------------------------------------------
    raw = np.full(n, A_POS, dtype=np.float32)

    raw = np.minimum(raw, a_follow_max)
    raw = np.minimum(raw, a_speed_max)
    raw = np.minimum(raw, a_speed_region_max)
    raw = np.minimum(raw, a_edu_max)
    raw = np.minimum(raw, a_stop_required)

    # If overlap is already present, request maximum braking. If stopped, the
    # no-reverse handling below will convert this to zero.
    raw = np.where(gap < 0.0, A_NEG, raw)

    speed_limit_binds = a_speed_max < a_follow_min

    priority_block = (
        gap_too_small
        | overspeed
        | traffic_stop
        | speed_region_brake
        | edu_cap_binds
        | speed_limit_binds
    )

    # Close excessive gap only if no higher-priority constraint prevents it.
    raw = np.where(
        (a_follow_min > raw) & (~priority_block),
        np.maximum(raw, a_follow_min),
        raw,
    )

    # -----------------------------------------------------------------------
    # 7. Comfort smoothing.
    #
    # The current ego acceleration is used as a proxy for the previous executed
    # acceleration. This is approximate because the executed acceleration may
    # differ from the previous requested action due to environment clipping.
    # -----------------------------------------------------------------------
    comfort_block = priority_block | (gap < 0.0)

    raw_before = raw.copy()
    comfort_min = a_current - COMFORT_DELTA
    comfort_max = a_current + COMFORT_DELTA
    raw_comfort = np.clip(raw, comfort_min, comfort_max)

    raw = np.where(comfort_block, raw, raw_comfort)
    comfort_limit_applied = (
        (~comfort_block)
        & (np.abs(raw - raw_before) > 1e-6)
    )

    # No reverse request when stopped. The environment also clips this, but the
    # rule avoids requesting negative acceleration at zero speed.
    raw = np.where((v <= 0.1) & (raw < 0.0), 0.0, raw)

    driving_action = np.clip(raw, A_NEG, A_POS).astype(np.float32).reshape(-1, 1)

    driving_active = (
        gap_too_small
        | overspeed
        | traffic_stop
        | speed_region_brake
        | edu_cap_binds
        | speed_limit_binds
        | speed_limit_enforce
        | comfort_limit_applied
        | (gap_too_large & (v < v_limit - 0.5))
    ).astype(bool).reshape(-1, 1)

    # -----------------------------------------------------------------------
    # 8. Energy rule: approximate FCS power request.
    #
    # The observation does not provide battery power, FCS actual power, EDU
    # loss, hydrogen consumption, or motor torque. The rule estimates traction
    # mechanical power from ego speed and a driving acceleration estimate.
    #
    # If the driving rule is active, use the rule's requested acceleration as
    # the demand estimate. If inactive, use the current observed acceleration.
    # This is an approximation; the fused driving action may differ.
    # -----------------------------------------------------------------------
    a_for_energy = np.where(driving_active, raw, a_current)

    traction_kw = (M_EFF * a_for_energy + f_resistance) * v / 1000.0
    p_demand_kw = np.maximum(0.0, traction_kw)

    braking = a_for_energy < -0.1

    # SOC-based FCS request bands. These are design choices intended to keep
    # SOC near 0.5 while avoiding very low FCS power when the FCS is on.
    base = np.zeros(n, dtype=np.float32)

    base = np.where(soc < 0.20, 50.0, base)
    base = np.where((soc >= 0.20) & (soc < 0.35), 40.0, base)
    base = np.where((soc >= 0.35) & (soc < 0.45), 30.0, base)
    base = np.where((soc >= 0.45) & (soc < 0.50), 20.0, base)

    # During braking, do not request FCS unless SOC is critically low.
    # This reduces hydrogen use and avoids FCS load changes when regen may be
    # available, but it relies on the environment's actual dispatch behavior.
    base = np.where(braking & (soc >= 0.30), 0.0, base)

    # If battery SOH is low and estimated traction demand is high, request more
    # FCS power to reduce battery discharge. This is a proxy for battery SOH
    # preservation; the exact battery aging increment is not observed.
    add = np.zeros(n, dtype=np.float32)
    add = np.where(
        (battery_soh < 0.98) & (p_demand_kw > 30.0) & (soc < 0.60),
        20.0,
        add,
    )
    add = np.where(
        (battery_soh < 0.90) & (p_demand_kw > 30.0) & (soc < 0.60),
        30.0,
        add,
    )

    power = base + add

    # Avoid very low positive FCS power when SOC is not critically low. This is
    # a heuristic against low-power FCS degradation, but the exact low-power
    # threshold is not provided by the observation.
    low_power_mask = (
        (power > 0.0)
        & (power < 15.0)
        & (soc >= 0.45)
        & (soc >= 0.30)
    )
    power = np.where(low_power_mask, 0.0, power)

    # FCS SOH-based power caps. These are heuristics for high-power degradation.
    cap = np.full(n, FCS_MAX_KW, dtype=np.float32)
    cap = np.where(fcs_soh < 0.98, np.minimum(cap, 35.0), cap)
    cap = np.where(fcs_soh < 0.95, np.minimum(cap, 25.0), cap)
    cap = np.where(fcs_soh < 0.90, np.minimum(cap, 15.0), cap)

    # Critical SOC override: if SOC is very low, allow full FCS request despite
    # FCS SOH caps. The environment may still limit the actual dispatch.
    cap = np.where(soc < 0.30, FCS_MAX_KW, cap)

    power = np.clip(power, 0.0, cap)
    power = np.clip(power, 0.0, FCS_MAX_KW).astype(np.float32).reshape(-1, 1)

    energy_active = (
        (soc < 0.45)
        | (soc > 0.55)
        | (power > 1e-6)
        | ((battery_soh < 0.98) & (p_demand_kw > 30.0) & (soc < 0.60))
        | (fcs_soh < 0.95)
    ).astype(bool).reshape(-1, 1)

    return {
        "driving": {
            "action": driving_action,
            "active": driving_active,
        },
        "energy": {
            "action": power,
            "active": energy_active,
        },
    }
```

---

## B. Explanation of behavior, triggers, masks, fusion, and limitations

### 1. Program role and interface compliance

The program is a deterministic rule prior. It returns physical actions:

- `driving.action`: requested acceleration in m/s², bounded to `[-1.5, 0.7]`.
- `energy.action`: requested FCS output power in kW, bounded to `[0, 50]`.

It accepts:

- a single raw observation shaped `(15,)`, or
- a batch shaped `(batch, 15)`.

It validates shape and finiteness, then returns float32 arrays shaped `(batch, 1)` and boolean masks shaped `(batch, 1)`.

It does not:

- add exploration noise,
- return `std`,
- hard-code `lambda`,
- access SUMO directly,
- modify rewards or configuration,
- bypass environment clipping or powertrain feasibility limits.

The adapter is expected to blend the MADDPG physical actions and rule physical actions as:

```text
a_fused = (1 - lambda) * a_MADDPG_physical + lambda * a_rule_physical
```

The program does not know `lambda`; it only provides the rule-side physical action and active mask.

If an actor's `active` mask is false for a row, the adapter should pass the MADDPG actor action for that row without rule fusion. The returned action for inactive rows is still clipped to physical bounds, but it should not be used by the fusion logic if the mask is false.

---

### 2. Observation fields used

The rule uses the complete raw observation:

| Index | Field | Use in rule |
|---:|---|---|
| 0 | Battery SOC | Energy request bands and critical SOC override |
| 1 | Battery SOH | Increase FCS request when battery SOH is low and demand is high |
| 2 | FCS SOH | Cap requested FCS power to reduce high-power degradation |
| 3 | EDU SOH | Softly cap driving acceleration to limit estimated EDU mechanical power |
| 4 | Ego acceleration | Comfort smoothing, traction-demand estimate, braking proxy |
| 5 | Ego speed | Speed limits, following gap thresholds, traffic arrival, power estimate |
| 6 | Lead acceleration | Short-horizon following prediction |
| 7 | Lead speed | Short-horizon following prediction |
| 8 | Gap | Following safety and gap-closure logic |
| 9 | Traffic-light state | Red/green/yellow decision only in approach zone |
| 10 | Distance to traffic light | Approach-zone logic and arrival prediction |
| 11 | Remaining phase time | Stop/proceed prediction |
| 12 | Current speed limit | Speed-limit acceleration cap |
| 13 | Next speed limit | Lower-limit entry shaping |
| 14 | Distance to next speed region | Entry-speed prediction |

The rule treats `30000 m` as the sentinel for no visible target or outside effective range. A traffic-light distance of `0 m` is treated as valid and means the ego is at the stop line or has reached it.

---

### 3. Driving rule behavior

The driving rule returns a requested acceleration, not a target speed. It combines several constraints as acceleration bounds.

#### 3.1 Following constraint

The rule uses a one-step constant-acceleration prediction:

```text
d_pred = gap + (v_lead - v) * dt + 0.5 * (a_lead - a_req) * dt^2
```

with `dt = 1 s`.

It uses the project following thresholds:

```text
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

with a small safety margin added to `D_safe`. It then computes acceleration bounds that would keep the predicted gap between the safe and maximum following ranges:

- `a_follow_max`: upper acceleration limit to avoid predicted gap becoming too small.
- `a_follow_min`: lower acceleration limit to avoid predicted gap becoming excessively large.

The rule does not assume a fixed lead-vehicle profile. It uses live ego speed, lead speed, ego acceleration, lead acceleration, and gap at every decision step.

If the predicted gap is too small, the rule requests braking. If the gap is too large and no higher-priority constraint prevents acceleration, the rule requests positive acceleration to reduce excessive following distance.

If `gap < 0`, the rule requests maximum braking. If the ego is stopped, the no-reverse handling later converts negative requests to zero.

#### 3.2 Current posted speed limit

The current posted limit is taken from observation index 12. The rule clips it to the hard vehicle cap and uses it to produce an acceleration upper bound:

```text
a_speed_max = (v_limit - v) / dt
```

clipped to `[-1.5, 0.7]`.

This is not a target-speed command. It is converted into a requested acceleration.

The rule also enables active driving intervention when speed is close to the posted limit to reduce the risk of continuous speeding penalty or 120% speed termination.

#### 3.3 Lower-limit entry shaping

When the next speed region has a lower limit, the rule estimates entry speed using the same braking assumption used by the project entry-shaping reward:

```text
v_entry = sqrt(max(0, v^2 - 2 * 0.75 * d_entry))
```

If the predicted entry speed exceeds the next limit plus a margin, the rule requests braking so that the ego is more likely to enter the lower-limit region at an acceptable speed.

This applies before entering a lower-limit region. It does not treat a red traffic light as a zero speed limit.

#### 3.4 Traffic-light logic

The traffic-light rule is active only from entry into the 100 m approach zone through the stop line. A signal farther than 100 m away does not by itself stop or prevent starting/following.

The rule uses the actual cycle order:

```text
green -> yellow -> red -> green
```

##### Red light

For a red light, the next phase after red is green. The rule requests stopping if the ego would cross while the signal is still red. If the ego is stopped, it uses a conservative minimum arrival time assuming maximum positive acceleration from rest. If that minimum arrival time is after the red phase ends, the rule may allow starting, because the ego would not cross during red.

##### Yellow light

For a yellow light, the next phase after yellow is red. The rule requests stopping if the ego cannot safely cross before yellow ends. If the ego can cross yellow safely and cannot stop comfortably, the rule may proceed because yellow crossing does not terminate the episode, although the project reward gives negative traffic prediction reward for yellow crossing.

##### Green light

For a green light, the next phases are yellow and then red. The rule proceeds if the ego can cross while green. If the ego would cross during yellow, it stops when comfortable stopping is possible. If the ego would cross during red, it stops.

##### Stop-line distance zero

A traffic-light distance of zero is valid. At the stop line:

- red: the rule requests stop action;
- yellow: the rule proceeds if already moving and yellow remains, otherwise may stop;
- green: the rule proceeds if already moving or if already at the stop line.

This avoids treating zero distance as absence of a traffic light.

#### 3.5 EDU degradation soft cap

The rule estimates traction mechanical power using the given vehicle dynamics:

```text
F = M_eff * a + drag_force + rolling_resistance
P_mech = F * v
```

where:

```text
M_eff = 7000 * 1.05
drag_force = 0.5 * rho * Cd * A * v^2
rolling_resistance = Crr * m * g
```

The rule then softly caps acceleration so that the estimated mechanical power remains near the EDU aging reference point of 75 kW, scaled by EDU SOH.

This is an approximation. It is not EDU electrical loss, inverter loss, gearbox loss, motor torque, EDU rpm, or EDU map efficiency. The EDU map in the project already represents a combined motor/inverter/gearbox boundary, so the rule does not add those losses a second time.

#### 3.6 Comfort smoothing

For non-emergency cases, the rule limits the requested acceleration change relative to the current observed ego acceleration:

```text
|a_req - a_current| <= 0.75 m/s^2
```

The current ego acceleration is used as a proxy for the previous executed acceleration. This is approximate because the actual executed acceleration may differ from the previous requested action due to clipping.

Comfort smoothing is suppressed when higher-priority constraints are active, such as unsafe following, overspeed, traffic stopping, lower-limit entry braking, or EDU cap binding.

#### 3.7 No-reverse handling

If the ego speed is effectively zero and the requested acceleration is negative, the rule sets the action to zero. The environment also clips negative acceleration at zero speed, but this avoids unnecessary reverse requests.

---

### 4. Driving active masks

The `driving.active` mask is true when the rule has a meaningful intervention. It becomes true if any of the following are true:

- predicted following gap is unsafe or overlap exists;
- ego speed exceeds the current posted limit;
- ego speed is close to the current posted limit;
- speed-limit constraint prevents a desired following acceleration;
- traffic-light stop is required or strongly recommended;
- lower-limit entry shaping requires braking;
- EDU soft power cap binds;
- comfort smoothing changes the requested acceleration;
- gap is excessively large and the ego is below the speed limit.

If `driving.active` is false, the adapter should pass the MADDPG driving actor action through without rule fusion for that row.

---

### 5. Energy rule behavior

The energy action is a requested FCS output power in kW. It is not battery power. The environment may clip or change the requested FCS power due to energy dispatch, EDU demand, battery limits, FCS limits, or other physical constraints.

#### 5.1 Traction-demand estimate

The observation does not directly provide:

- battery power,
- FCS actual power,
- requested FCS power,
- motor torque,
- instantaneous EDU loss,
- hydrogen consumption,
- SOC change rate,
- battery current,
- battery voltage,
- EDU efficiency.

The rule therefore estimates traction mechanical power from ego speed and an acceleration estimate:

```text
P_mech = (M_eff * a + drag_force + rolling_resistance) * v
```

If the driving rule is active, the energy rule uses the driving rule's requested acceleration as the acceleration estimate. If the driving rule is inactive, it uses the current observed ego acceleration.

This is an approximation. The actual fused driving action may differ, and the actual electrical demand may differ from mechanical traction power because EDU efficiency and powertrain dispatch are not directly observed.

#### 5.2 SOC target management

The rule uses SOC bands around the target `0.5`:

- `SOC < 0.20`: request 50 kW;
- `0.20 <= SOC < 0.35`: request 40 kW;
- `0.35 <= SOC < 0.45`: request 30 kW;
- `0.45 <= SOC < 0.50`: request 20 kW;
- `0.50 <= SOC <= 0.55`: request 0 kW unless battery SOH or FCS SOH conditions activate;
- `SOC > 0.55`: request 0 kW unless battery SOH and high demand conditions justify limited FCS use.

The rule disables FCS request during braking unless SOC is critically low. This reduces hydrogen use and avoids unnecessary FCS load changes, assuming regenerative braking may be available. However, the rule does not observe actual regen or battery charging power.

#### 5.3 Battery SOH proxy

If battery SOH is low and estimated traction demand is high, the rule increases the FCS request to reduce battery discharge. This is a heuristic proxy for battery SOH preservation. The actual battery SOH loss increment is not observed.

#### 5.4 FCS SOH caps

The rule caps requested FCS power based on FCS SOH:

- `FCS_SOH < 0.98`: cap 35 kW;
- `FCS_SOH < 0.95`: cap 25 kW;
- `FCS_SOH < 0.90`: cap 15 kW.

If SOC is critically low, the rule allows a higher request despite the FCS SOH cap. This is a tradeoff between SOC deviation and FCS degradation.

#### 5.5 Low-power avoidance

When the SOC is not critically low, the rule suppresses very low positive FCS requests below 15 kW. This is a heuristic intended to reduce low-power FCS degradation. The exact low-power degradation threshold is not available from the observation.

#### 5.6 Energy active masks

The `energy.active` mask is true when the rule has a clear energy-management recommendation:

- SOC is outside the target band;
- the rule requests nonzero FCS power;
- battery SOH is low and estimated demand is high;
- FCS SOH is low, so the rule wants to limit FCS power even when its own request is zero.

If `energy.active` is false, the adapter should pass the MADDPG energy actor action through without rule fusion for that row.

---

### 6. Physical-unit fusion

The MADDPG actors produce normalized latent actions:

```text
u_d = tanh(z_d)
u_e = tanh(z_e)
```

The adapter maps them to physical actions:

Driving actor:

```text
if u_d < 0:
    a_MADDPG = 1.5 * u_d
else:
    a_MADDPG = 0.7 * u_d
```

Energy actor:

```text
P_MADDPG = 25 * (u_e + 1)
```

The rule program directly returns physical actions:

```text
a_rule ∈ [-1.5, 0.7] m/s^2
P_rule ∈ [0, 50] kW
```

The fusion is independent per actor:

```text
a_fused = (1 - lambda) * a_MADDPG + lambda * a_rule
```

```text
P_fused = (1 - lambda) * P_MADDPG + lambda * P_rule
```

After fusion, the trainer maps the physical result back to normalized actions for replay and centralized critics. The rule program does not perform this mapping itself.

The rule does not guarantee the executed action. The environment may still clip the fused action due to:

- hard maximum speed,
- no-reverse handling,
- battery SOC/current/voltage/power limits,
- EDU torque-speed/efficiency validity,
- FCS limits,
- energy dispatch feasibility,
- SUMO vehicle-motion constraints.

---

### 7. Relation to project objectives

The rule is designed to assist the stated objectives.

#### Following and collision avoidance

- Uses live gap, ego speed, lead speed, lead acceleration.
- Uses the project `D_safe` and `D_max` ideas.
- Requests braking when predicted gap is unsafe.
- Requests acceleration when gap is excessively large, unless higher-priority constraints prevent it.

#### Comfort

- Limits non-emergency acceleration changes.
- Uses current ego acceleration as a proxy for previous executed acceleration.
- Suppresses comfort smoothing during safety-critical or speed/traffic-critical situations.

#### Speed limits and intersection safety

- Enforces the current posted limit through acceleration bounds.
- Applies lower-limit entry shaping before entering a 30 km/h zone.
- Uses traffic-light phase only in the 100 m approach zone.
- Avoids red-light crossing.
- Handles yellow crossing conservatively but not as a hard termination condition.

#### Energy management

- Requests FCS power when SOC is low.
- Requests zero or limited FCS power when SOC is high or braking.
- Uses battery SOH to increase FCS request when battery SOH is low and demand is high.
- Uses FCS SOH to cap high FCS power.
- Uses EDU SOH and estimated mechanical power to cap driving acceleration near the EDU aging reference point.

#### SOC deviation

- SOC bands target the desired SOC of 0.5.
- SOC is treated as a deviation-management objective, not as a battery degradation metric.

---

### 8. Important limitations

1. **No direct EDU loss or motor torque information**  
   The observation does not include motor torque, instantaneous EDU loss, EDU efficiency, EDU rpm, or EDU map operating point. The EDU cap is based on approximate mechanical traction power.

2. **No direct battery power or FCS actual power information**  
   The rule estimates demand mechanically. It does not observe battery power, FCS actual output, hydrogen consumption, or actual dispatch.

3. **No SOC-change history**  
   The rule uses current SOC only. It does not observe SOC rate, SOC increment, battery current, voltage, temperature, or charge/discharge power.

4. **No FCS startup/load-change history**  
   FCS aging includes startup, low-power, high-power, and load-change effects. The rule uses SOC bands and SOH caps as heuristics but cannot track startup count, low-power duration, or load-change history.

5. **Approximate traction power**  
   The traction power estimate uses current speed and an acceleration estimate. It does not include EDU efficiency, inverter losses, gearbox losses, battery/FCS dispatch efficiency, or auxiliary loads.

6. **Approximate following prediction**  
   The following bounds assume constant acceleration over one step and use observed lead acceleration as a short-horizon predictor. Actual lead behavior may change.

7. **Approximate comfort proxy**  
   Comfort smoothing uses current ego acceleration as a proxy for previous acceleration. The executed acceleration may differ from the previous requested action because of environment clipping.

8. **Traffic prediction is one-step and deterministic**  
   The rule uses current speed and remaining phase time. If the ego accelerates or decelerates, the actual arrival time may differ from the predicted arrival time.

9. **Thresholds are design choices**  
   SOC bands, FCS SOH caps, EDU reference scaling, comfort delta, safety margins, and active-mask conditions are not prescribed by the environment. They are rule-design choices intended to align with the stated objectives.

10. **No validation claim**  
   This program is a static design. No test suite is included, and no claim is made that the rule has been executed or validated in SUMO.

---

### 9. CLI selection

Assuming the file is placed at:

```text
llm-prior/programs/fchev_rule_prior.py
```

select it with:

```bash
--prior-program-name fchev_rule_prior
```

Use physical-action fusion mode:

```bash
--maddpg-rule-output-mode physical_action
```

Set the blend externally, for example:

```bash
--maddpg-rule-lambda 0.2
```

The value of `lambda` must be supplied by the trainer or experiment configuration. It must not be hard-coded in the rule.

During the first 1,000 steps of a fresh run, prior-free random warm-up is used, so this rule is not fused during that warm-up. A resumed run skips that warm-up. Evaluation does not add exploration noise.