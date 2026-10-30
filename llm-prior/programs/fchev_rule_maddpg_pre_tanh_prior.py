# llm-prior/programs/fchev_follow_energy_prior.py
"""
Deterministic pre-tanh latent prior for the FCHEV MADDPG driving and energy actors.

The rule receives the complete raw 15-dimensional observation and returns one
latent prior value per actor per observation row. It does not add noise, does
not return std, and does not bypass the environment's physical constraints.
"""

import numpy as np


DT = 1.0
HORIZON = 10                 # within the required 5-20 step horizon
HARD_CAP = 70.0 / 3.6        # m/s, separate vehicle speed hard cap

# Vehicle / road parameters used only for approximate traction-power estimation.
VEHICLE_MASS = 7000.0
ROTATIONAL_MASS_COEFF = 1.05
ROLLING_RESISTANCE_COEFF = 0.0085
FRONTAL_AREA = 5.8
DRAG_COEFF = 0.55
AIR_DENSITY = 1.225
GRAVITY = 9.8

M_EFF = VEHICLE_MASS * ROTATIONAL_MASS_COEFF
ROLL_FORCE = ROLLING_RESISTANCE_COEFF * VEHICLE_MASS * GRAVITY
DRAG_COEF = 0.5 * AIR_DENSITY * DRAG_COEFF * FRONTAL_AREA

# Candidate requested accelerations for the driving prior.
# Includes 0 m/s^2 and both action endpoints.
CANDIDATES = np.linspace(-1.5, 0.7, 45, dtype=np.float32)


def _is_green_phase(state, remaining_time, tau):
    """
    Approximate whether the next signal is green at time tau seconds from now.

    Only the current phase state and remaining time are observed. Future phases
    are approximated using the documented 90 s cycle:
        green 40 s, yellow 3 s, red 47 s.

    state:
        0 red, 1 green, 2 yellow
    remaining_time:
        remaining time in the current phase, seconds
    tau:
        predicted time to an event, seconds
    """
    tau = float(max(0.0, tau))

    current_green = (state == 1) & (tau < remaining_time)

    future_mask = tau >= remaining_time
    elapsed = np.mod(np.maximum(tau - remaining_time, 0.0), 90.0)

    future_green = np.zeros_like(state, dtype=bool)

    # If current phase is green, after current green ends:
    # yellow 3 s, red 47 s, then green 40 s.
    green_now = (state == 1)
    future_green |= green_now & future_mask & (elapsed >= 50.0)

    # If current phase is yellow, after current yellow ends:
    # red 47 s, then green 40 s.
    yellow_now = (state == 2)
    future_green |= yellow_now & future_mask & (elapsed >= 47.0) & (elapsed < 87.0)

    # If current phase is red, after current red ends:
    # green 40 s, then yellow 3 s.
    red_now = (state == 0)
    future_green |= red_now & future_mask & (elapsed < 40.0)

    return current_green | future_green


def predict(observation):
    """
    Return deterministic pre-tanh latent priors for both MADDPG actors.

    Input:
        observation: raw unnormalized NumPy array with shape (15,) or (batch, 15).

    Output:
        {
            "driving": {"prior": (batch, 1) float32 latent prior,
                        "active": (batch, 1) bool},
            "energy":  {"prior": (batch, 1) float32 latent prior,
                        "active": (batch, 1) bool}
        }
    """
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15 or not np.all(np.isfinite(x)):
        raise ValueError("observation must be finite with shape (15,) or (batch, 15)")

    n = x.shape[0]

    if n == 0:
        empty_prior = np.zeros((0, 1), dtype=np.float32)
        active = np.ones((0, 1), dtype=bool)
        return {
            "driving": {"prior": empty_prior, "active": active},
            "energy": {"prior": empty_prior, "active": active},
        }

    # ------------------------------------------------------------------
    # Raw observation extraction and conservative sanitization.
    # The rule always receives raw units, not normalized actor inputs.
    # ------------------------------------------------------------------
    soc = np.clip(x[:, 0], 0.0, 1.0)
    soh_battery = np.clip(x[:, 1], 0.0, 1.0)
    soh_fcs = np.clip(x[:, 2], 0.0, 1.0)
    soh_edu = np.clip(x[:, 3], 0.0, 1.0)

    ego_acc = np.clip(x[:, 4], -1.5, 0.7)
    ego_v = np.maximum(x[:, 5], 0.0)

    lead_acc = x[:, 6]
    lead_v = np.maximum(x[:, 7], 0.0)

    gap = x[:, 8]

    signal_state = np.clip(x[:, 9], 0.0, 2.0).astype(np.int8)
    stop_dist = np.maximum(x[:, 10], 0.0)
    phase_rem = np.maximum(x[:, 11], 0.0)

    limit_cur = np.where(x[:, 12] > 0.0, x[:, 12], 16.6667)
    limit_next_raw = x[:, 13]
    limit_next = np.where(limit_next_raw > 0.0, limit_next_raw, limit_cur)

    entry_dist = np.maximum(x[:, 14], 0.0)

    # ------------------------------------------------------------------
    # Driving prior: candidate-action longitudinal planning.
    # ------------------------------------------------------------------
    c = CANDIDATES.shape[0]

    v_ego = np.repeat(ego_v[:, None], c, axis=1)
    v_lead = np.repeat(lead_v[:, None], c, axis=1)

    ego_dist = np.zeros((n, c), dtype=np.float32)
    lead_dist = np.zeros((n, c), dtype=np.float32)

    cost = np.zeros((n, c), dtype=np.float32)
    crossed = np.zeros((n, c), dtype=bool)
    prev_ego_cons = np.zeros((n, c), dtype=np.float32)

    gap0 = gap[:, None]
    gap_valid = gap0 < 29999.9

    stop_b = stop_dist[:, None]
    stop_valid = stop_b < 29999.9

    entry_b = entry_dist[:, None]
    entry_valid = entry_b < 29999.9

    limit_cur_b = limit_cur[:, None]
    limit_next_b = limit_next[:, None]

    signal_state_b = signal_state[:, None]

    for k in range(1, HORIZON + 1):
        v_ego_old = v_ego.copy()
        v_lead_old = v_lead.copy()

        # Active posted limit for this step.  The conservative predicted ego
        # position from the previous step is used to decide whether the next
        # speed region has been entered.
        active_limit = np.where(
            entry_valid & (prev_ego_cons >= entry_b),
            limit_next_b,
            limit_cur_b,
        )
        active_limit = np.where(active_limit > 0.0, active_limit, HARD_CAP)

        # Candidate requested acceleration is applied as a planning maneuver,
        # but predicted ego speed is constrained by the hard vehicle cap and,
        # when currently within the posted limit, by the posted limit.
        v_req = v_ego + CANDIDATES[None, :] * DT
        v_req = np.minimum(v_req, HARD_CAP)

        within_limit = v_ego <= active_limit + 1.0e-3
        v_req = np.where(
            within_limit,
            np.minimum(v_req, active_limit),
            v_req,
        )

        v_ego = np.maximum(0.0, v_req)
        v_lead = np.maximum(0.0, v_lead + lead_acc[:, None] * DT)

        # Integrate distance using average speed, then add the required
        # conservative one-step uncertainty margin.
        ego_dist += 0.5 * (v_ego_old + v_ego) * DT
        lead_dist += 0.5 * (v_lead_old + v_lead) * DT

        v_extra = np.maximum(v_ego_old, v_ego)
        ego_cons = ego_dist + v_extra * DT

        gap_cons = gap0 + lead_dist - ego_cons

        # Following objective.
        d_safe = 2.0 + 0.5 * v_ego + 0.0625 * v_ego * v_ego
        d_max = 10.0 + v_ego + 0.0825 * v_ego * v_ego

        unsafe = np.maximum(0.0, d_safe - gap_cons)
        over = np.maximum(0.0, gap_cons - d_max)

        unsafe_ratio = unsafe / np.maximum(d_safe, 1.0)
        over_ratio = over / np.maximum(d_max, 1.0)

        unsafe_ratio = np.minimum(unsafe_ratio, 20.0)
        over_ratio = np.minimum(over_ratio, 100.0)

        cost += np.where(
            gap_valid,
            10000.0 * unsafe_ratio + 500.0 * over_ratio,
            0.0,
        )

        # Collision / overlap is treated as a very strong penalty.
        cost += np.where(gap_valid & (gap_cons <= 0.0), 1.0e6, 0.0)

        # Speed-limit and hard-speed penalties.
        speed_excess = np.maximum(0.0, v_ego - active_limit)
        cost += 100000.0 * speed_excess / np.maximum(active_limit, 1.0)
        cost += np.where(v_ego > HARD_CAP, 1.0e6, 0.0)

        # Approaching a lower posted-speed region.
        entered = (
            entry_valid
            & (ego_cons >= entry_b)
            & (prev_ego_cons < entry_b)
            & (limit_next_b < limit_cur_b)
        )
        entry_error = np.maximum(0.0, v_ego - limit_next_b) / np.maximum(
            limit_next_b, 1.0
        )
        cost += 10000.0 * entry_error * entered

        prev_ego_cons = ego_cons

        # Traffic-light crossing feasibility.
        # Conservative crossing time is one step earlier than the nominal step.
        # The nominal step time is also checked to avoid crossing when the
        # signal may change between conservative and nominal arrival.
        tau_cons = float(max(0, k - 1))
        tau_nom = float(k)

        phase_ok_cons = _is_green_phase(signal_state, phase_rem, tau_cons)[:, None]
        phase_ok_nom = _is_green_phase(signal_state, phase_rem, tau_nom)[:, None]
        phase_ok = phase_ok_cons & phase_ok_nom

        cross_now = (
            (~crossed)
            & stop_valid
            & (ego_cons >= stop_b)
            & (v_extra > 1.0e-6)
        )

        non_green_cross = cross_now & (~phase_ok)
        green_cross = cross_now & phase_ok

        cost += np.where(non_green_cross, 1.0e6, 0.0)
        cost += np.where(green_cross, -20.0, 0.0)

        crossed = crossed | cross_now

        # Proximity penalty when approaching a red/yellow signal.
        dist_to_line = stop_b - ego_cons
        proximity = (
            (~crossed)
            & stop_valid
            & (signal_state_b != 1)
            & (dist_to_line >= 0.0)
            & (dist_to_line < 20.0)
            & (v_extra > 0.5)
        )
        proximity_score = np.maximum(0.0, 1.0 - dist_to_line / 20.0)
        cost += 1000.0 * proximity * proximity_score

        # Small penalty for requesting positive acceleration while already at
        # the active posted limit.  This is not a physical clamp; it only makes
        # the latent prior less likely to ask for pointless positive requests.
        positive_at_limit = (
            (CANDIDATES[None, :] > 1.0e-6)
            & (v_ego_old >= active_limit - 1.0e-3)
        )
        cost += 10.0 * positive_at_limit

    # Driving comfort prior: prefer accelerations close to the currently
    # observed executed acceleration.
    cost += 50.0 * np.abs(CANDIDATES[None, :] - ego_acc[:, None]) / 2.2

    cost = np.nan_to_num(cost, nan=1.0e9, posinf=1.0e9, neginf=-1.0e9)

    best_idx = np.argmin(cost, axis=1)
    best_cost = cost[np.arange(n), best_idx]

    zero_idx = int(np.argmin(np.abs(CANDIDATES)))
    best_idx = np.where(best_cost >= 1.0e9, zero_idx, best_idx)

    a_desired = CANDIDATES[best_idx].astype(np.float32)

    # Convert desired acceleration to normalized driving action u_d, then to
    # pre-tanh latent z_rule.
    u_d = np.where(
        a_desired < 0.0,
        a_desired / 1.5,
        a_desired / 0.7,
    )
    u_d = np.clip(u_d, -0.9999, 0.9999)
    z_drive = np.arctanh(u_d).astype(np.float32)
    z_drive = np.nan_to_num(
        z_drive,
        nan=0.0,
        posinf=4.605,
        neginf=-4.605,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Energy prior: approximate traction demand, SOC regulation, and SOH-aware
    # FCS power request.
    # ------------------------------------------------------------------
    entry_1d = entry_dist < 29999.9
    limit_next_safe = np.where(limit_next_raw > 0.0, limit_next_raw, limit_cur)

    v_prev = ego_v.copy()
    ego_dist_energy = np.zeros(n, dtype=np.float32)
    prev_cons_energy = np.zeros(n, dtype=np.float32)

    demand_kw = np.zeros((n, HORIZON), dtype=np.float32)

    for h in range(HORIZON):
        active_limit_e = np.where(
            entry_1d & (prev_cons_energy >= entry_dist),
            limit_next_safe,
            limit_cur,
        )
        active_limit_e = np.where(active_limit_e > 0.0, active_limit_e, HARD_CAP)

        v_req_e = v_prev + a_desired * DT
        v_req_e = np.minimum(v_req_e, HARD_CAP)

        within_limit_e = v_prev <= active_limit_e + 1.0e-3
        v_req_e = np.where(
            within_limit_e,
            np.minimum(v_req_e, active_limit_e),
            v_req_e,
        )

        v_next = np.maximum(0.0, v_req_e)
        a_eff = (v_next - v_prev) / DT

        drag_force = DRAG_COEF * v_next * v_next
        p_road_kw = ((M_EFF * a_eff + ROLL_FORCE + drag_force) * v_next) / 1000.0

        # Negative road power corresponds to braking/regeneration.  The FCS
        # requested output power is kept nonnegative; battery recovery is not
        # directly observable and is not modeled by this rule.
        demand_kw[:, h] = np.maximum(0.0, p_road_kw)

        ego_dist_energy += 0.5 * (v_prev + v_next) * DT
        v_extra_e = np.maximum(v_prev, v_next)
        prev_cons_energy = ego_dist_energy + v_extra_e * DT

        v_prev = v_next

    # Blend immediate predicted demand with short-horizon average demand.
    p_now = demand_kw[:, 0]
    p_avg = np.mean(demand_kw[:, :min(HORIZON, 5)], axis=1)
    p_demand = 0.6 * p_now + 0.4 * p_avg

    # Approximate previous traction demand from the current observed ego
    # acceleration and speed.  This is only a proxy for load-change smoothing.
    drag_prev = DRAG_COEF * ego_v * ego_v
    p_prev_est = np.maximum(
        0.0,
        ((M_EFF * ego_acc + ROLL_FORCE + drag_prev) * ego_v) / 1000.0,
    )

    # Smooth toward the previous estimated traction demand to reduce abrupt
    # FCS load changes.  This is a heuristic, not a measured FCS power.
    p_smooth = 0.85 * p_demand + 0.15 * p_prev_est

    # SOC regulation toward 0.5.  Positive correction when SOC is low,
    # negative correction when SOC is high.
    soc_error = soc - 0.5
    p_soc = -25.0 * soc_error

    p_target = p_smooth + p_soc

    # SOH-aware upper bound on requested FCS power.  This is a conservative
    # degradation-management heuristic; the environment may further limit power.
    soh_min = np.minimum(np.minimum(soh_battery, soh_fcs), soh_edu)
    p_max_allowed = 50.0 * np.clip(0.5 + 0.5 * soh_min, 0.5, 1.0)

    p_req = np.clip(p_target, 0.0, p_max_allowed)
    p_req = np.nan_to_num(
        p_req,
        nan=0.0,
        posinf=50.0,
        neginf=0.0,
    ).astype(np.float32)

    # Convert requested FCS power to normalized energy action u_e, then to
    # pre-tanh latent z_rule.
    u_e = 2.0 * p_req / 50.0 - 1.0
    u_e = np.clip(u_e, -0.9999, 0.9999)
    z_energy = np.arctanh(u_e).astype(np.float32)
    z_energy = np.nan_to_num(
        z_energy,
        nan=0.0,
        posinf=4.605,
        neginf=-4.605,
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # Output.
    # ------------------------------------------------------------------
    z_drive = z_drive.reshape(n, 1).astype(np.float32)
    z_energy = z_energy.reshape(n, 1).astype(np.float32)

    active = np.ones((n, 1), dtype=bool)

    return {
        "driving": {"prior": z_drive, "active": active},
        "energy": {"prior": z_energy, "active": active},
    }