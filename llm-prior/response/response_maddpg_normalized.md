## A) Suggested Python rule-prior module

Suggested filename:

```text
llm-prior/programs/fchev_follow_power_prior.py
```

Program name for CLI selection:

```text
fchev_follow_power_prior
```

Complete module:

```python
import numpy as np

# -----------------------------------------------------------------------------
# Rule-prior program for FCHEV MADDPG fusion.
#
# This program returns deterministic normalized-action priors for:
#   1) driving actor: normalized acceleration command u_d in [-1, 1]
#   2) energy actor:  normalized FCS power request u_e in [-1, 1]
#
# It is a soft prior. It does not replace MADDPG, does not bypass physical
# constraints, does not add noise, and does not use persistent state between
# steps. The rule is recomputed from the raw 15-dimensional observation at
# every 1 s step.
# -----------------------------------------------------------------------------

H = 12  # prediction horizon in steps, i.e. 12 s. Within the allowed 5-20 range.

SENTINEL = 29999.0
HARD_SPEED_CAP = 19.444444444444442  # 70 km/h
ORDINARY_LIMIT = 16.666666666666668  # 60 km/h
INTERSECTION_LIMIT = 8.333333333333334  # 30 km/h

# Vehicle / road constants used only for approximate power-demand estimation.
MASS = 7000.0
ROTATIONAL_MASS_COEFF = 1.05
AIR_DENSITY = 1.225
DRAG_AREA_CD = 5.8 * 0.55
DRAG_COEFF = 0.5 * AIR_DENSITY * DRAG_AREA_CD
ROLLING_COEFF = 0.0085
GRAVITY = 9.8
ROLLING_FORCE = MASS * GRAVITY * ROLLING_COEFF
M_EFF = MASS * (1.0 + ROTATIONAL_MASS_COEFF)

# Candidate requested accelerations. Step size 0.05 m/s^2.
CANDIDATE_ACCEL = np.linspace(-1.5, 0.7, 45, dtype=np.float64)


def _phase_state_at(state, rem, k):
    """
    Forecast signal phase state at integer step k.

    state: 0=red, 1=green, 2=yellow
    rem:   remaining time in current phase, already capped to nominal phase length
    k:     step index, 1..H, or H+1 for no crossing

    Returns:
      0=red, 1=green, 2=yellow

    The crossing time is approximated by the midpoint of step k, tau = k - 0.5,
    to reduce boundary ambiguity. This is still a simplified predictor.
    """
    state = np.asarray(state, dtype=np.int8)
    rem = np.asarray(rem, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)

    tau = np.maximum(k - 0.5, 0.0)

    # Default invalid / red.
    out = np.zeros(state.shape, dtype=np.int8)

    # If crossing occurs before the current phase ends, use current phase.
    out = np.where((state == 1) & (tau < rem), 1, out)
    out = np.where((state == 2) & (tau < rem), 2, out)
    out = np.where((state == 0) & (tau < rem), 0, out)

    elapsed = np.maximum(tau - rem, 0.0)

    # Current phase was green: green remaining -> yellow 3 s -> red 47 s -> green.
    phase_after_green = np.where(
        elapsed < 3.0,
        2,
        np.where(elapsed < 50.0, 0, 1),
    )
    out = np.where((state == 1) & (tau >= rem), phase_after_green, out)

    # Current phase was yellow: yellow remaining -> red 47 s -> green.
    phase_after_yellow = np.where(elapsed < 47.0, 0, 1)
    out = np.where((state == 2) & (tau >= rem), phase_after_yellow, out)

    # Current phase was red: red remaining -> green 40 s -> yellow 3 s -> red.
    phase_after_red = np.where(
        elapsed < 40.0,
        1,
        np.where(elapsed < 43.0, 2, 0),
    )
    out = np.where((state == 0) & (tau >= rem), phase_after_red, out)

    return out


def predict(observation):
    """
    Return deterministic normalized-action priors for both MADDPG actors.

    Input:
      observation: raw unnormalized observation, shape (15,) or (batch, 15).

    Output:
      {
        "driving": {"action": (batch, 1) float32 in [-1, 1],
                    "active": (batch, 1) bool True},
        "energy":  {"action": (batch, 1) float32 in [-1, 1],
                    "active": (batch, 1) bool True}
      }
    """
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15 or not np.all(np.isfinite(x)):
        raise ValueError(
            "observation must be finite with shape (15,) or (batch, 15)"
        )

    batch = x.shape[0]

    if batch == 0:
        empty_action = np.zeros((0, 1), dtype=np.float32)
        empty_active = np.ones((0, 1), dtype=bool)
        return {
            "driving": {"action": empty_action, "active": empty_active},
            "energy": {"action": empty_action, "active": empty_active},
        }

    # Use float64 internally for numerical stability; return float32 actions.
    x = x.astype(np.float64)

    # ---------------------------------------------------------------------
    # Raw observation extraction.
    # ---------------------------------------------------------------------
    soc = np.clip(x[:, 0], 0.0, 1.0)
    battery_soh = np.clip(x[:, 1], 0.0, 1.0)
    fcs_soh = np.clip(x[:, 2], 0.0, 1.0)
    edu_soh = np.clip(x[:, 3], 0.0, 1.0)

    current_acc = np.clip(x[:, 4], -1.5, 0.7)
    ego_speed = np.clip(x[:, 5], 0.0, HARD_SPEED_CAP)

    lead_acc = x[:, 6]
    lead_speed = np.clip(x[:, 7], 0.0, None)

    gap = x[:, 8]

    signal_state = np.rint(np.clip(x[:, 9], 0.0, 2.0)).astype(np.int8)
    stop_dist = x[:, 10]
    signal_rem_raw = np.clip(x[:, 11], 0.0, 90.0)

    # Posted speed limit. If invalid/zero, fall back to the hard vehicle cap.
    current_limit = np.where(x[:, 12] > 1.0e-3, x[:, 12], HARD_SPEED_CAP)

    next_limit_raw = x[:, 13]
    next_dist_raw = x[:, 14]

    next_valid = (next_dist_raw < SENTINEL) & (next_limit_raw > 1.0e-3)
    next_limit = np.where(next_valid, next_limit_raw, 0.0)
    next_dist = np.where(next_valid, next_dist_raw, SENTINEL)

    signal_valid = stop_dist < SENTINEL
    no_lead = gap >= SENTINEL

    # Cap remaining phase time to nominal phase duration to avoid permissive
    # behavior if an observation contains an inconsistent remaining time.
    signal_rem = np.where(
        signal_state == 1,
        np.minimum(signal_rem_raw, 40.0),
        np.where(
            signal_state == 2,
            np.minimum(signal_rem_raw, 3.0),
            np.minimum(signal_rem_raw, 47.0),
        ),
    )

    signal_valid_any = bool(np.any(signal_valid))
    next_valid_any = bool(np.any(next_valid))

    # ---------------------------------------------------------------------
    # Driving prior: candidate acceleration search over horizon H.
    #
    # Each candidate is a constant requested acceleration for the whole
    # horizon. This is intentionally conservative and finite-horizon.
    # The rule is soft; MADDPG fusion may still choose a different action.
    # ---------------------------------------------------------------------
    best_score = np.full(batch, -np.inf, dtype=np.float64)
    best_acc = np.zeros(batch, dtype=np.float64)

    for a_req in CANDIDATE_ACCEL:
        ego_v = ego_speed.copy()
        lead_v = lead_speed.copy()

        ego_pos_nom = np.zeros(batch, dtype=np.float64)
        lead_pos = np.zeros(batch, dtype=np.float64)

        cross_cons = np.full(batch, H + 1, dtype=np.int32)
        cross_nom = np.full(batch, H + 1, dtype=np.int32)

        following_sum = np.zeros(batch, dtype=np.float64)
        cruise_sum = np.zeros(batch, dtype=np.float64)
        speed_hard_count = np.zeros(batch, dtype=np.float64)
        speed_term_count = np.zeros(batch, dtype=np.float64)
        speed_cont = np.zeros(batch, dtype=np.float64)

        final_follow_cost = np.zeros(batch, dtype=np.float64)
        first_exec_acc = np.zeros(batch, dtype=np.float64)

        for step in range(1, H + 1):
            # Ego motion with environment-like no-reverse clipping and hard cap.
            ego_v_new = np.clip(ego_v + a_req, 0.0, HARD_SPEED_CAP)
            exec_acc = ego_v_new - ego_v

            if step == 1:
                first_exec_acc = exec_acc.copy()

            # Trapezoidal one-step distance.
            ego_dist = 0.5 * (ego_v + ego_v_new)

            # Conservative uncertainty: one extra step farther than nominal.
            ego_v_next_est = np.clip(ego_v_new + exec_acc, 0.0, HARD_SPEED_CAP)
            margin_dist = 0.5 * (ego_v_new + ego_v_next_est)

            ego_pos_nom += ego_dist
            ego_pos_cons = ego_pos_nom + margin_dist

            # Lead prediction: assume current lead acceleration is constant.
            lead_v_new = np.maximum(0.0, lead_v + lead_acc)
            lead_dist = 0.5 * (lead_v + lead_v_new)
            lead_pos += lead_dist
            lead_v = lead_v_new

            predicted_gap = gap + lead_pos - ego_pos_cons

            # Following cost.
            d_safe = 2.0 + 0.5 * ego_v_new + 0.0625 * ego_v_new * ego_v_new
            d_max = 10.0 + ego_v_new + 0.0825 * ego_v_new * ego_v_new

            follow_cost = np.zeros(batch, dtype=np.float64)
            valid_gap = ~no_lead

            collision = valid_gap & (predicted_gap <= 0.0)
            follow_cost[collision] = 1.0e6

            below = valid_gap & ~collision & (predicted_gap < d_safe)
            ratio_below = np.clip(
                (d_safe - predicted_gap) / np.maximum(d_safe, 1.0),
                0.0,
                1.0,
            )
            follow_cost[below] += 500.0 * ratio_below[below] * ratio_below[below]

            above = valid_gap & ~collision & (predicted_gap > d_max)
            ratio_above = np.clip(
                (predicted_gap - d_max) / np.maximum(d_max, 1.0),
                0.0,
                100.0,
            )
            follow_cost[above] += 100.0 * ratio_above[above]

            following_sum += follow_cost
            final_follow_cost = follow_cost.copy()

            # If no valid lead target is observed, add a weak cruise objective
            # so the ego does not remain stopped when no following target exists.
            if bool(np.any(no_lead)):
                cruise_deficit = np.maximum(
                    0.0,
                    (current_limit_step := None) or (None),  # placeholder to avoid lint; replaced below
                )
            # The line above is intentionally replaced below to keep code clear.
            # Compute cruise penalty vectorized.
            cruise_sum += 10.0 * np.where(
                no_lead,
                np.maximum(
                    0.0,
                    (limit_k if False else 0.0),  # placeholder replaced below
                ),
                0.0,
            )

            # -----------------------------------------------------------------
            # Speed-limit construction for this step.
            # -----------------------------------------------------------------
            limit_k = current_limit.copy()

            mask_next = np.zeros(batch, dtype=bool)
            if next_valid_any:
                mask_next = next_valid & (ego_pos_cons >= next_dist)
                limit_k = np.where(mask_next, next_limit, limit_k)

            if signal_valid_any:
                approaching_intersection = (
                    signal_valid
                    & (stop_dist > 0.0)
                    & (ego_pos_cons < stop_dist)
                    & ((stop_dist - ego_pos_cons) <= 100.0)
                )

                at_stop_line = (
                    signal_valid
                    & (stop_dist >= -1.0e-3)
                    & (stop_dist <= 0.0)
                    & (ego_pos_cons <= 1.0e-3)
                )

                crossed_stop_line = signal_valid & (
                    ((stop_dist > 0.0) & (ego_pos_cons >= stop_dist))
                    | (stop_dist < -1.0e-3)
                    | ((stop_dist <= 0.0) & (ego_pos_cons > 1.0e-3))
                )

                limit_k = np.where(
                    approaching_intersection | at_stop_line,
                    np.minimum(limit_k, INTERSECTION_LIMIT),
                    limit_k,
                )

                # After crossing a signalized stop line, return to ordinary road
                # speed unless the next observed speed region already governs.
                limit_k = np.where(
                    crossed_stop_line & ~mask_next,
                    np.maximum(limit_k, ORDINARY_LIMIT),
                    limit_k,
                )

            # Recompute cruise penalty with the actual limit_k for this step.
            cruise_sum += 10.0 * np.where(
                no_lead,
                np.maximum(
                    0.0,
                    (limit_k - ego_v_new) / np.maximum(limit_k, 1.0),
                ),
                0.0,
            )

            # Speed penalties.
            overspeed = np.maximum(
                0.0,
                (ego_v_new - limit_k) / np.maximum(limit_k, 1.0),
            )
            speed_cont += 100.0 * overspeed

            hard_overspeed = ego_v_new > (limit_k + 1.0e-3)
            speed_hard_count += hard_overspeed.astype(np.float64)

            termination_speed = ego_v_new >= (1.2 * limit_k - 1.0e-3)
            speed_term_count += termination_speed.astype(np.float64)

            # Traffic-light crossing detection for stop lines ahead.
            if signal_valid_any:
                mask_cross = signal_valid & (stop_dist > 1.0e-3)

                set_cons = (
                    mask_cross
                    & (cross_cons == H + 1)
                    & (ego_pos_cons >= stop_dist)
                )
                cross_cons = np.where(set_cons, step, cross_cons)

                set_nom = (
                    mask_cross
                    & (cross_nom == H + 1)
                    & (ego_pos_nom >= stop_dist)
                )
                cross_nom = np.where(set_nom, step, cross_nom)

            ego_v = ego_v_new

        following_penalty = following_sum + 2.0 * final_follow_cost + cruise_sum
        speed_penalty = (
            1.0e6 * speed_hard_count
            + speed_cont
            + 1.0e6 * speed_term_count
        )

        # -----------------------------------------------------------------
        # Traffic-light penalties/bonuses.
        # -----------------------------------------------------------------
        signal_penalty = np.zeros(batch, dtype=np.float64)
        signal_bonus = np.zeros(batch, dtype=np.float64)

        if signal_valid_any:
            # Special case: ego is at or past the stop line.
            special_at_or_past = signal_valid & (stop_dist <= 1.0e-3)

            # Treat green with zero remaining time conservatively as not green.
            non_green_special = special_at_or_past & (
                (signal_state != 1) | (signal_rem <= 1.0e-3)
            )

            bad_penalty = np.where(signal_state == 0, 1.0e6, 5.0e5)
            moving_special = ego_speed > 1.0e-3

            # If already moving toward/past a non-green stop line, penalize all
            # candidates, but encourage maximum braking.
            signal_penalty += np.where(
                non_green_special & moving_special,
                bad_penalty
                + 100.0 * np.maximum(0.0, first_exec_acc + 1.5),
                0.0,
            )

            # If stopped at a non-green stop line, penalize only candidates that
            # would start moving forward.
            signal_penalty += np.where(
                non_green_special & ~moving_special & (a_req > 1.0e-3),
                bad_penalty,
                0.0,
            )

            # Small bonus for being at a green stop line with remaining green time.
            signal_bonus += np.where(
                special_at_or_past & (signal_state == 1) & (signal_rem > 1.0e-3),
                2.0,
                0.0,
            )

            # Regular case: stop line ahead and potentially crossed within H.
            regular_stop_line = signal_valid & (stop_dist > 1.0e-3)
            crossed = regular_stop_line & (cross_cons <= H)

            if bool(np.any(crossed)):
                phase_cons = _phase_state_at(signal_state, signal_rem, cross_cons)
                phase_nom = _phase_state_at(signal_state, signal_rem, cross_nom)

                allowed = (
                    crossed
                    & (phase_cons == 1)
                    & ((cross_nom > H) | (phase_nom == 1))
                )

                bad_state = np.where(phase_cons != 1, phase_cons, phase_nom)

                signal_penalty += np.where(
                    crossed & ~allowed,
                    np.where(bad_state == 2, 5.0e5, 1.0e6),
                    0.0,
                )

                bonus_allowed = crossed & allowed
                signal_bonus += np.where(
                    bonus_allowed & (signal_state == 1) & (cross_cons <= signal_rem),
                    5.0,
                    np.where(bonus_allowed, 2.0, 0.0),
                )

        # Comfort and small action regularization.
        comfort_penalty = (
            2.0 * np.abs(first_exec_acc - current_acc)
            + 0.05 * np.abs(a_req)
        )

        score = (
            -following_penalty
            - speed_penalty
            - signal_penalty
            - comfort_penalty
            + signal_bonus
        )
        score = np.nan_to_num(score, nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)

        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_acc = np.where(better, a_req, best_acc)

    # Map selected requested acceleration to normalized driving action.
    driving_u = np.where(
        best_acc < 0.0,
        best_acc / 1.5,
        best_acc / 0.7,
    )
    driving_u = np.clip(driving_u, -1.0, 1.0).astype(np.float32).reshape(-1, 1)

    # ---------------------------------------------------------------------
    # Energy prior.
    #
    # Approximate power demand from observed ego speed/acceleration and the
    # selected driving prior. This is only a proxy: the observation does not
    # include motor torque, EDU loss, battery power, hydrogen consumption, or
    # actual FCS delivered power.
    # ---------------------------------------------------------------------

    # Instantaneous approximate wheel/mechanical demand.
    force_current = (
        M_EFF * current_acc
        + DRAG_COEFF * ego_speed * ego_speed
        + ROLLING_FORCE
    )
    current_demand_kw = np.maximum(0.0, force_current * ego_speed / 1000.0)

    # Future approximate demand using the selected driving prior acceleration.
    demand_v = ego_speed.copy()
    future_demand_sum = np.zeros(batch, dtype=np.float64)

    for _ in range(H):
        demand_v_new = np.clip(demand_v + best_acc, 0.0, HARD_SPEED_CAP)
        demand_acc_eff = demand_v_new - demand_v

        force = (
            M_EFF * demand_acc_eff
            + DRAG_COEFF * demand_v_new * demand_v_new
            + ROLLING_FORCE
        )
        future_demand_sum += np.maximum(0.0, force * demand_v_new / 1000.0)
        demand_v = demand_v_new

    future_demand_kw = future_demand_sum / float(H)

    # Blend current observed demand and future predicted demand.
    demand_est_kw = 0.5 * current_demand_kw + 0.5 * future_demand_kw

    # SOC regulation term.
    k_soc = 40.0 * np.clip(battery_soh, 0.5, 1.0)
    target_power_kw = demand_est_kw + k_soc * (0.5 - soc)

    # Degradation-aware power caps.
    # Default cap is 45 kW to avoid unnecessary high FCS load.
    max_power_kw = np.where(
        (soc < 0.25) | (demand_est_kw > 45.0),
        50.0,
        45.0,
    )

    # Reduce high-power requests when component SOH is already degraded.
    max_power_kw = np.where(fcs_soh < 0.95, np.minimum(max_power_kw, 40.0), max_power_kw)
    max_power_kw = np.where(edu_soh < 0.95, np.minimum(max_power_kw, 35.0), max_power_kw)
    max_power_kw = np.where(battery_soh < 0.95, np.minimum(max_power_kw, 40.0), max_power_kw)

    # Avoid aggressive FCS charging when SOC is already high.
    max_power_kw = np.where(soc > 0.85, np.minimum(max_power_kw, 25.0), max_power_kw)
    max_power_kw = np.where(soc > 0.75, np.minimum(max_power_kw, 35.0), max_power_kw)

    target_power_kw = np.nan_to_num(
        target_power_kw,
        nan=0.0,
        posinf=50.0,
        neginf=0.0,
    )
    target_power_kw = np.clip(target_power_kw, 0.0, max_power_kw)

    # Avoid very low FCS load when SOC is below target and the vehicle is moving.
    low_target = target_power_kw < 5.0
    moving = ego_speed > 0.1

    target_power_kw = np.where(low_target & (soc >= 0.5), 0.0, target_power_kw)
    target_power_kw = np.where(low_target & (soc < 0.45), 5.0, target_power_kw)
    target_power_kw = np.where(
        low_target & (soc >= 0.45) & (soc < 0.5) & moving,
        5.0,
        target_power_kw,
    )
    target_power_kw = np.where(
        low_target & (soc >= 0.45) & (soc < 0.5) & ~moving,
        0.0,
        target_power_kw,
    )

    target_power_kw = np.clip(target_power_kw, 0.0, max_power_kw)

    # Map requested FCS power to normalized energy action.
    energy_u = target_power_kw / 25.0 - 1.0
    energy_u = np.clip(energy_u, -1.0, 1.0).astype(np.float32).reshape(-1, 1)

    active = np.ones((batch, 1), dtype=bool)

    return {
        "driving": {"action": driving_u, "active": active},
        "energy": {"action": energy_u, "active": active},
    }
```

---

## B) Explanation of behavior, triggers, masks, blending, limitations, and CLI selection

### 1. Program scope and interface

The module is intended for:

```text
llm-prior/programs/fchev_follow_power_prior.py
```

It exposes:

```python
predict(observation)
```

The observation must be the raw, unnormalized 15-dimensional observation in the documented order. The program accepts:

- a single observation with shape `(15,)`
- a batch observation with shape `(batch, 15)`

It validates:

- finite values
- correct dimensionality
- 15 columns

It returns:

```python
{
    "driving": {"action": driving_action, "active": driving_mask},
    "energy":  {"action": energy_action,  "active": energy_mask},
}
```

where:

```python
driving_action.shape == (batch, 1)
energy_action.shape  == (batch, 1)
driving_mask.shape   == (batch, 1)
energy_mask.shape    == (batch, 1)
```

Actions are `float32` and finite in `[-1, 1]`.

Masks are always `True` for both actors:

```python
active = np.ones((batch, 1), dtype=bool)
```

The rule is deterministic. It does not return `std`, does not sample noise, does not call SUMO, does not modify rewards or configuration, and does not bypass physical constraints.

---

### 2. Driving prior logic

The driving prior is built from a finite candidate search over requested accelerations.

Candidate requested accelerations are:

```python
np.linspace(-1.5, 0.7, 45)
```

This covers the full allowed requested-acceleration range:

```text
[-1.5, +0.7] m/s^2
```

with step size:

```text
0.05 m/s^2
```

For each candidate acceleration, the rule predicts ego and lead motion over:

```python
H = 12 steps
```

which is 12 seconds, within the required 5–20 step horizon.

The prediction horizon is finite. The rule does not extrapolate signal phases beyond `H`. At the next 1 s step, the rule is recomputed from the new observation.

---

### 3. Conservative uncertainty margin

The rule includes a conservative one-step uncertainty margin in ego position.

For each predicted step, it computes a nominal ego position and then adds an estimated distance corresponding to one extra step of ego motion:

```python
margin_dist = 0.5 * (ego_v_new + ego_v_next_est)
```

where `ego_v_next_est` is the estimated ego speed after one additional step using the same candidate acceleration.

This conservative ego position is used for:

- predicted following gap
- stop-line crossing detection
- speed-region entry prediction
- intersection-zone speed-limit application

The nominal position is also tracked for a second crossing-time check. This makes the rule conservative: if the earliest plausible crossing occurs during a non-green phase, the candidate is penalized even if the nominal crossing would have occurred during green.

---

### 4. Following objective

The driving prior uses the observed ego-to-lead gap and predicts future gaps.

Safe and maximum following distances are computed from predicted ego speed:

```python
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

where `v` is the predicted ego speed at the evaluation step.

The rule penalizes:

- predicted gap `<= 0`: collision/overlap penalty
- predicted gap below `D_safe`: unsafe following gap
- predicted gap above `D_max`: excessive gap / catch-up needed

The following penalty is evaluated over the whole horizon and weighted more strongly at the final predicted step:

```python
following_penalty = sum_over_steps(cost) + 2 * final_step_cost
```

This encourages actions that keep the ego inside the safe-to-prescribed gap range whenever feasible.

If the gap is already too large, the rule treats the situation as catch-up and prefers feasible acceleration that reduces the predicted gap, subject to speed limits and traffic-light constraints.

If the gap is too small, the rule prefers actions that reduce closing speed and protect the safe gap.

If the observed gap is at or above the sentinel threshold, the rule treats the lead as unavailable and suppresses the following penalty. In that case, it adds a weak cruise objective so that the ego can accelerate toward the current speed limit when no valid lead target is observed.

---

### 5. Lead prediction approximation

The rule predicts the lead vehicle using:

- current lead speed
- current lead acceleration
- constant lead acceleration over the horizon

This is an approximation. The actual lead acceleration may change due to traffic lights, lead-vehicle behavior, or SUMO dynamics. The rule does not model lead-vehicle future acceleration changes. It uses the current observed lead acceleration as a finite-horizon proxy.

Limitation:

- lead acceleration variability is not forecast
- lead speed changes are only predicted from the current lead acceleration
- lead prediction uncertainty is not explicitly modeled beyond the ego-side one-step conservative margin

---

### 6. Speed-limit handling

The rule uses the raw speed-limit observations:

- index 12: current speed limit
- index 13: next speed limit
- index 14: distance to next speed-region entrance

It also conservatively enforces the intersection-zone speed limit when a valid signal stop line is within 100 m:

```python
INTERSECTION_LIMIT = 8.333333333333334 m/s
```

This is consistent with the stated road rule that the ego limit becomes 30 km/h beginning 100 m before each stop line and continues through the intersection zone.

The rule constructs a stepwise speed limit:

1. Start with the current posted limit.
2. If the next speed region is valid and the conservative ego position has reached its entrance, switch to the next posted limit.
3. If a valid signal stop line is within 100 m, apply the intersection limit.
4. After crossing the stop line, return to the ordinary 60 km/h limit unless the next observed speed region already applies.

The rule penalizes predicted speed above the active speed limit.

It uses:

- a continuous overspeed penalty
- a hard overspeed penalty if predicted speed exceeds the active limit by more than a small tolerance
- a hard termination-speed penalty if predicted speed reaches or exceeds 120% of the active posted limit

The hard overspeed penalty is intentionally strong so the rule does not recommend accelerating through a posted speed limit within the horizon.

Because the candidate acceleration is held constant over the horizon, the rule can be conservative near the speed limit. It may avoid a small acceleration that would eventually exceed the limit within `H`, even if a multi-step acceleration/braking sequence could have been feasible. This is a limitation of the finite candidate model.

---

### 7. Traffic-light handling

The rule uses:

- index 9: next signal state, encoded as red 0, green 1, yellow 2
- index 10: distance to next stop line
- index 11: remaining time in current phase

A distance of `30000 m` is treated as a sentinel meaning no visible next signal target. A distance of `0 m` is treated as a valid observation meaning the ego is at the stop line.

The rule forecasts signal phase within the finite horizon `H = 12`.

For candidate actions that are predicted to reach the stop line within `H`, the rule evaluates:

1. the conservative earliest crossing step
2. the nominal crossing step
3. the forecast phase at both crossing times

A candidate is allowed only if both the conservative earliest crossing phase and the nominal crossing phase are green. If either crossing time falls in yellow or red, the candidate is penalized.

The rule does not make unbounded future signal predictions beyond `H`. If the predicted crossing occurs after `H`, the rule does not penalize that candidate based on signal phase in the current step; it will reassess at the next step.

Special handling is used when the ego is at or past the stop line:

- If the signal is green with remaining time, the rule allows normal driving.
- If the signal is red or yellow, the rule penalizes forward motion.
- If the ego is already moving toward a non-green stop line, the rule penalizes all candidates but encourages maximum braking.
- If the ego is stopped at a non-green stop line, the rule penalizes candidates that would start moving forward.

The rule treats a green signal with zero remaining time conservatively as non-green.

---

### 8. Comfort handling

The comfort term approximates the MADDPG comfort objective.

It uses:

- current observed ego acceleration as the previous executed acceleration
- the candidate’s first-step executed acceleration as the next executed acceleration proxy

The executed acceleration is computed with environment-like clipping:

```python
exec_acc = ego_v_new - ego_v_old
```

This respects the no-reverse behavior: if ego speed is zero and a negative acceleration is requested, the executed acceleration is zero.

The rule does not add a rule-level low-speed threshold that changes a negative request to zero. It may choose zero because of scoring, but it does not explicitly clamp negative requests to zero. The physical environment still handles no-reverse clipping.

A small regularization term is added to break ties and avoid arbitrary extreme actions when multiple candidates have nearly identical scores.

---

### 9. Driving normalized-action mapping

After selecting the best candidate requested acceleration `a_req`, the rule maps it to normalized driving action `u_d`.

The environment mapping is:

```text
if u_d < 0:
    a_req = 1.5 u_d
else:
    a_req = 0.7 u_d
```

The rule inverts this:

```python
if a_req < 0:
    u_d = a_req / 1.5
else:
    u_d = a_req / 0.7
```

Then it clips:

```python
u_d = clip(u_d, -1, 1)
```

Examples:

```text
a_req = -0.75 m/s^2 -> u_d = -0.5
a_req = +0.35 m/s^2 -> u_d = +0.5
a_req = -1.5 m/s^2  -> u_d = -1.0
a_req = +0.7 m/s^2  -> u_d = +1.0
```

The returned driving action is normalized, not physical acceleration.

---

### 10. Energy prior logic

The energy prior selects a normalized FCS power request.

The rule does not directly observe:

- motor torque
- instantaneous EDU loss
- actual battery power
- actual FCS delivered power
- hydrogen consumption
- EDU efficiency at the current operating point
- DC/DC loss
- battery current/voltage limits

Therefore, the energy rule uses an approximate demand estimate.

---

### 11. Approximate power-demand estimate

The rule estimates approximate vehicle mechanical demand from the observed ego speed and acceleration.

It uses the documented vehicle parameters:

```text
mass = 7000 kg
rotational-mass coefficient = 1.05
drag coefficient = 0.55
frontal area = 5.8 m^2
air density = 1.225 kg/m^3
rolling-resistance coefficient = 0.0085
road slope = 0
```

Approximate force:

```python
force = M_eff * acceleration + drag_force + rolling_force
```

where:

```python
M_eff = 7000 * 1.05
drag_force = 0.5 * rho * Cd * A * v^2
rolling_force = m * g * Crr
```

Approximate power:

```python
power_kw = max(0, force * v) / 1000
```

The rule computes:

1. instantaneous demand using current ego speed and current observed ego acceleration
2. future demand using the selected driving prior acceleration over `H`
3. a blended demand estimate:

```python
demand_est = 0.5 * current_demand + 0.5 * future_demand
```

This demand estimate is a proxy for required mechanical/EDU boundary power. It is not an electrical FCS power requirement. It does not include EDU efficiency, battery charge/discharge limits, FCS ramping behavior, or actual powertrain feasibility.

---

### 12. SOC regulation in the energy prior

The energy prior regulates SOC toward the target value:

```python
SOC_target = 0.5
```

It adds a SOC correction:

```python
target_power = demand_est + K_soc * (0.5 - SOC)
```

where:

```python
K_soc = 40 * max(0.5, battery_SOH)
```

Behavior:

- If SOC is below 0.5, the rule increases requested FCS power.
- If SOC is above 0.5, the rule decreases requested FCS power.
- If battery SOH is low, the SOC correction is reduced to avoid overly aggressive SOC regulation.

---

### 13. Degradation-aware FCS power caps

The rule caps FCS power to reduce high-power degradation and load-change stress.

Default cap:

```python
45 kW
```

It allows 50 kW only when:

```python
SOC < 0.25
```

or:

```python
estimated demand > 45 kW
```

It further reduces the maximum requested FCS power when SOH is degraded:

```text
FCS SOH < 0.95  -> max 40 kW
EDU SOH < 0.95  -> max 35 kW
Battery SOH < 0.95 -> max 40 kW
```

It also reduces the maximum when SOC is already high:

```text
SOC > 0.85 -> max 25 kW
SOC > 0.75 -> max 35 kW
```

These caps are heuristic. They are not derived from exact component aging models. They are intended to bias the prior toward moderate FCS loading.

---

### 14. Low-power handling

The rule tries to avoid very low FCS requests when SOC is below target and the vehicle is moving.

If the target power is below 5 kW:

- If SOC is already at or above 0.5, the rule sets the target to 0 kW.
- If SOC is below 0.45, the rule sets the target to 5 kW.
- If SOC is between 0.45 and 0.5 and the vehicle is moving, the rule sets the target to 5 kW.
- If SOC is between 0.45 and 0.5 and the vehicle is stopped, the rule sets the target to 0 kW.

This is a heuristic balance between hydrogen consumption, FCS low-power degradation, and SOC regulation.

---

### 15. Energy normalized-action mapping

The environment maps normalized energy action to requested FCS power:

```python
P_req = 25 * (u_e + 1) kW
```

The rule inverts this:

```python
u_e = target_power_kw / 25 - 1
```

Then it clips:

```python
u_e = clip(u_e, -1, 1)
```

Examples:

```text
target_power = 0 kW  -> u_e = -1
target_power = 25 kW -> u_e = 0
target_power = 50 kW -> u_e = +1
target_power = 45 kW -> u_e = +0.8
```

The returned energy action is normalized, not kW.

---

### 16. Fusion with MADDPG

The rule returns normalized actions. The adapter blends each rule action with the corresponding MADDPG actor action:

```python
u_fused = (1 - lambda) * u_MADDPG + lambda * u_rule
```

Then:

```python
u_fused = clip(u_fused, -1, 1)
```

The existing physical mapping then applies:

Driving:

```text
if u_fused < 0:
    requested acceleration = 1.5 * u_fused
else:
    requested acceleration = 0.7 * u_fused
```

Energy:

```text
requested FCS power = 25 * (u_fused + 1) kW
```

The environment may clip or modify the requested action because of:

- hard vehicle speed cap
- no-reverse handling
- battery SOC/current/voltage limits
- battery charge/discharge rate limits
- EDU torque-speed/efficiency validity
- FCS output limits
- DC/DC and dispatch constraints
- powertrain feasibility

The rule prior cannot bypass these constraints.

`lambda` is external. It is not hard-coded in the program.

---

### 17. Active masks

The rule returns active masks as always `True`:

```python
active = np.ones((batch, 1), dtype=bool)
```

This satisfies the requirement that both actor priors be evaluated and fused at every step. The rule does not use inactive masks to skip fusion.

---

### 18. Determinism and statelessness

The rule is deterministic.

It does not:

- add exploration noise
- return `std`
- store internal state between calls
- use previous rule outputs
- call online services
- access SUMO directly
- change rewards or configuration

Every action is recomputed from the current raw observation.

---

### 19. Relationship to MADDPG objectives

The driving prior supports the driving actor objectives:

- follow the lead vehicle
- maintain gap at least `D_safe`
- avoid gap above `D_max` when feasible
- close excessive gaps when feasible
- obey current posted speed limit
- obey next speed-region limits when approaching them
- cross stop lines only when the forecast signal state is green
- avoid yellow/red crossings
- avoid collision/overlap
- avoid 120% speed-limit termination behavior
- approximate comfort by discouraging abrupt acceleration changes

The energy prior supports the energy actor objectives:

- reduce hydrogen consumption by avoiding unnecessary high FCS power
- reduce FCS degradation by avoiding high-power and aggressive load-change behavior
- reduce battery and EDU degradation by moderating power requests when SOH is low
- regulate SOC toward 0.5
- avoid very low FCS load when SOC is below target and the vehicle is moving
- avoid aggressive charging when SOC is already high

The rule does not reproduce the exact reward equations. It uses the reward structure only as behavioral guidance.

---

### 20. Important limitations

1. **Finite horizon limitation**

   The rule uses `H = 12` steps. It does not predict traffic signals beyond 12 s. If a candidate action would cause a stop-line crossing after the horizon, the rule reassesses at the next step.

2. **Simplified motion prediction**

   The rule uses a simplified discrete-time predictor with:

   - constant candidate acceleration over the horizon
   - trapezoidal distance approximation
   - ego no-reverse clipping
   - hard vehicle speed cap
   - conservative one-step ego distance margin

   SUMO’s exact within-step motion and integration may differ.

3. **Lead prediction approximation**

   The rule assumes the lead acceleration remains constant over the horizon. Actual lead acceleration may change.

4. **Constant candidate acceleration**

   Each candidate is evaluated as a constant acceleration over the whole horizon. This can make the rule conservative near speed limits because it may reject a small acceleration that would later require braking.

5. **Energy demand is approximate**

   The energy prior estimates mechanical/EDU-boundary power demand from speed and acceleration. It does not observe:

   - motor torque
   - EDU efficiency
   - inverter loss
   - gearbox loss
   - battery power
   - battery current/voltage limits
   - actual FCS delivered power
   - hydrogen consumption

   The estimate is explicitly approximate.

6. **No direct EDU/FCS efficiency model**

   The rule does not add detailed EDU, inverter, or gearbox losses. It uses a simplified vehicle-boundary power proxy. Actual environment feasibility may differ substantially.

7. **SOH caps are heuristic**

   The SOH-based power caps are simple degradation-avoidance heuristics. They are not exact aging-model calculations.

8. **Rule is soft, not authoritative**

   The fused MADDPG action may differ from the rule action. The rule may recommend an action that the environment later clips or changes.

9. **No validation claim**

   This is a static reasoning design. No claim is made that this program has been executed, tested, or validated in SUMO.

---

### 21. CLI selection

To select this program in normalized-action mode:

```bash
--prior-program-name fchev_follow_power_prior
--maddpg-rule-output-mode normalized_action
--maddpg-rule-lambda <value>
```

where:

```text
<value> in [0, 1]
```

Examples:

```bash
--prior-program-name fchev_follow_power_prior --maddpg-rule-output-mode normalized_action --maddpg-rule-lambda 0.2
```

or:

```bash
--prior-program-name fchev_follow_power_prior --maddpg-rule-output-mode normalized_action --maddpg-rule-lambda 0.5
```

For a fresh run, the first 1000 steps use prior-free random warm-up. For a resumed run, that warm-up is skipped according to the experiment configuration. Evaluation does not add exploration noise.