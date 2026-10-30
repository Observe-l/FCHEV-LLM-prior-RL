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