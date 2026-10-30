## (A) Suggested program file

Create the module as:

```text
llm-prior/programs/fchev_follow_energy_prior.py
```

Complete module:

```python
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
```

---

## (B) Precise explanation of the proposed rule

### 1. Program interface and output mode

The program is intended to be selected as a pre-`tanh` latent prior:

```bash
--prior-program-name fchev_follow_energy_prior
--maddpg-rule-output-mode pre_tanh
--maddpg-rule-lambda <lambda>
```

The rule receives the complete raw 15-dimensional observation, not the normalized actor observation. It supports both:

```python
observation.shape == (15,)
observation.shape == (batch, 15)
```

For a single observation, it returns arrays with shape `(1, 1)`. For a batch, it returns arrays with shape `(batch, 1)`.

Required output structure:

```python
{
    "driving": {"prior": driving_latent_prior, "active": driving_mask},
    "energy":  {"prior": energy_latent_prior,  "active": energy_mask},
}
```

Both masks are always `True`:

```python
active = np.ones((batch, 1), dtype=bool)
```

The rule does not return `std`, does not add exploration noise, does not call SUMO, does not change rewards or configuration, and does not bypass physical constraints.

---

### 2. Pre-`tanh` conversion and fusion

The rule first computes a desired physical action internally:

- Driving prior: desired requested acceleration `a_desired` in `[-1.5, 0.7] m/s²`.
- Energy prior: desired requested FCS power `p_req` in `[0, 50] kW`.

These are converted to normalized actions.

For driving:

```python
if a_desired < 0:
    u_d = a_desired / 1.5
else:
    u_d = a_desired / 0.7
```

For energy:

```python
u_e = 2 * p_req / 50 - 1
```

Then both normalized actions are clipped to avoid infinite latent values:

```python
u = clip(u, -0.9999, 0.9999)
z_rule = arctanh(u)
```

The fusion is assumed to occur externally as:

```python
z_fused = (1 - lambda) * z_MADDPG + lambda * z_rule
u_fused = tanh(z_fused)
```

The environment then maps:

```python
u_d -> requested acceleration
u_e -> requested FCS power
```

The rule does not hard-code `lambda`.

---

## 3. Driving prior behavior

### 3.1 Decision structure

At every 1 s step, the driving prior evaluates 45 candidate requested accelerations:

```python
CANDIDATES = linspace(-1.5, 0.7, 45)
```

The candidate spacing is 0.05 m/s².

For each candidate acceleration, the rule predicts a finite horizon of:

```python
HORIZON = 10
```

This is within the required 5–20 step horizon.

The candidate acceleration is used as a simplified planning maneuver over the horizon. However, predicted ego speed is constrained by:

1. No reverse motion.
2. The separate hard ego speed cap:

```python
70 km/h = 19.4444 m/s
```

3. The active posted speed limit, when the predicted ego speed is currently within the posted limit.

This prevents the rule from choosing an acceleration that would obviously violate the posted speed limit over the prediction horizon, while still allowing acceleration up to the posted limit for catch-up.

The rule does not clamp a negative acceleration request to zero merely because the ego speed is low. The physical no-reverse behavior is handled by the environment. The rule only uses `max(0, speed)` inside its own simplified prediction so that its horizon prediction does not predict reverse motion.

---

### 3.2 Simplified ego and lead prediction

The lead vehicle is predicted using its currently observed speed and acceleration:

```python
v_lead(t + 1) = max(0, v_lead(t) + a_lead * 1 s)
```

The ego vehicle is predicted using the candidate acceleration with the speed constraints described above.

Ego distance is integrated using the average of old and new speeds:

```python
ego_dist += 0.5 * (v_old + v_new) * 1 s
```

This is a simplified predictor. It is not SUMO's exact integration.

A conservative one-step uncertainty margin is included:

```python
ego_cons = ego_dist + max(v_old, v_new) * 1 s
```

This represents the possibility that the ego vehicle travels up to one simulation step farther than nominally predicted. This conservative margin is used for:

- Following gap evaluation.
- Stop-line crossing evaluation.
- Speed-region entry evaluation.

The rule does not extrapolate beyond the 10-step horizon. At the next control step, the priors are recomputed from the new observation.

---

### 3.3 Following objective

The primary driving objective is following the lead vehicle.

The rule evaluates predicted gaps using the same following-distance definitions as the reward background:

```python
D_safe = 2 + 0.5 v + 0.0625 v²
D_max  = 10 + v + 0.0825 v²
```

where `v` is the predicted ego speed at each horizon step.

For each candidate action, the rule penalizes:

1. Gap below `D_safe`.
2. Gap above `D_max`.
3. Gap at or below zero, interpreted as overlap/collision risk.

The unsafe-gap penalty is stronger than the excessive-gap penalty. Collision or overlap is given a very large penalty.

If the gap is too large, the rule gives candidates that reduce the gap a lower cost, provided they remain feasible with respect to speed limits and traffic signals.

If the gap is already excessive but the vehicle is constrained by a speed limit or traffic signal, the rule does not treat the excessive gap as a reason to remain stopped. It chooses the feasible candidate that best preserves following and makes progress toward the lead.

---

### 3.4 Speed-limit handling

The rule uses the raw observation values:

- Index 12: current-region speed limit.
- Index 13: next-region speed limit.
- Index 14: distance to the next speed-region entrance.

The active predicted limit is:

```python
current limit if conservative ego position has not reached the next-region entrance
next limit otherwise
```

The rule penalizes predicted ego speed above the active posted limit.

It also applies an entry-shaping penalty when approaching a lower-limit region. If the conservative predicted ego position enters the lower-limit region and the predicted entry speed is above the next limit, the candidate is penalized.

The hard ego speed cap is separate from the posted limit and is treated as a stronger constraint.

The rule does not assume that a red traffic light changes the posted speed limit to zero.

---

### 3.5 Traffic-light handling

The rule uses:

- Index 9: next traffic-light state.
- Index 10: distance to the next stop line.
- Index 11: remaining time in the current signal phase.

The documented signal cycle is used:

```text
green: 40 s
yellow: 3 s
red: 47 s
total: 90 s
```

The rule does not know future signal offsets directly. It only propagates the currently observed phase and remaining time forward within the 10-step horizon.

A sentinel distance of 30000 m is treated as “no visible next traffic light.” A distance of 0 m is still treated as a valid stop-line distance.

For each candidate action, the rule checks whether the conservative predicted ego position reaches the stop line within the horizon.

If a candidate is predicted to cross the stop line within the horizon, the rule checks the signal phase at both:

1. The conservative earlier arrival time:

```python
tau_cons = k - 1
```

2. The nominal step arrival time:

```python
tau_nom = k
```

The candidate is allowed to cross only if the signal is predicted green at both times. This is a conservative way to avoid crossing when the phase may change between the earlier and nominal arrival predictions.

Candidates predicted to cross on red or yellow receive a very large penalty. Candidates predicted to cross on green receive a small negative cost, i.e. a small bonus.

The rule also applies a proximity penalty when approaching a red or yellow signal but not yet crossing. This encourages stopping before the stop line when the signal is non-green and the stop line is relatively close.

The rule does not predict signal phases beyond the finite horizon. At the next step, the signal state and remaining phase time are re-observed and the prior is recomputed.

---

### 3.6 Comfort

The driving prior includes a comfort term:

```python
comfort_cost = 50 * |candidate_acceleration - current_ego_acceleration| / 2.2
```

The divisor 2.2 matches the full requested acceleration range:

```python
0.7 - (-1.5) = 2.2
```

This term is secondary. It helps avoid abrupt requested acceleration changes but does not override safety, following, speed-limit, or signal constraints.

---

### 3.7 Driving candidate selection

For each observation row, the rule selects the candidate acceleration with the lowest total cost:

```python
best_idx = argmin(cost)
```

If all candidate costs are invalid or numerically broken, the rule falls back to zero acceleration.

The selected candidate acceleration is then converted to the driving latent prior:

```python
u_d = a_desired / 1.5 if a_desired < 0
u_d = a_desired / 0.7 if a_desired >= 0

z_drive = arctanh(clip(u_d, -0.9999, 0.9999))
```

For example:

```python
a_desired = -0.75 m/s²
u_d = -0.5
z_drive = arctanh(-0.5) ≈ -0.5493
```

---

## 4. Energy prior behavior

### 4.1 Available information and approximations

The raw observation does not directly provide:

- Motor torque.
- Instantaneous EDU loss.
- Requested FCS power.
- Battery power.
- Hydrogen consumption.
- Exact EDU operating point.
- Exact battery charge/discharge power.

Therefore, the energy prior uses approximations. These approximations are clearly labeled as estimates and are not guaranteed to match the environment’s internal FCS/battery/EDU dispatch.

---

### 4.2 Approximate traction-demand estimate

The rule estimates road traction power from vehicle parameters:

```python
M_EFF = 7000 kg * 1.05 = 7350 kg

Rolling force:
F_roll = 0.0085 * 7000 * 9.8

Drag force:
F_drag = 0.5 * 1.225 * 0.55 * 5.8 * v²

Road power:
P_road = (M_EFF * a + F_roll + F_drag) * v
```

This is converted to kW.

If the predicted road power is negative, corresponding to braking or regenerative braking, the rule clamps the estimated FCS output demand to zero. The rule does not directly model battery recovery because battery power is not observed.

The energy prior recomputes a constrained ego trajectory over the same 10-step horizon using the selected driving acceleration `a_desired`. This trajectory is also speed-constrained by the hard cap and posted limits.

---

### 4.3 Demand blending

The rule blends immediate predicted demand with short-horizon average demand:

```python
p_demand = 0.6 * p_now + 0.4 * mean(p_demand over first 5 steps)
```

This is intended to make the energy prior responsive to the current driving action while still considering the near-term predicted outcome.

---

### 4.4 Load-change smoothing

The rule estimates a previous traction demand from the currently observed ego acceleration and speed:

```python
p_prev_est = max(0, estimated road power using current ego_v and ego_acc)
```

Then it smooths the new demand toward that estimate:

```python
p_smooth = 0.85 * p_demand + 0.15 * p_prev_est
```

This is a heuristic attempt to reduce abrupt FCS load changes. It is not a measured previous FCS power and does not directly model the FCS aging model.

---

### 4.5 SOC regulation

The energy actor’s SOC objective is to keep SOC near 0.5.

The rule computes:

```python
soc_error = SOC - 0.5
p_soc = -25 * soc_error
```

Meaning:

- If SOC is below 0.5, the rule increases requested FCS power to help recharge the battery.
- If SOC is above 0.5, the rule reduces requested FCS power and allows battery discharge to meet traction demand.

The coefficient 25 kW per unit SOC error is a heuristic. It is not derived from a directly observed battery power balance.

---

### 4.6 SOH-aware power cap

The rule uses battery SOH, FCS SOH, and EDU SOH to form a conservative maximum requested FCS power:

```python
soh_min = min(battery_SOH, FCS_SOH, EDU_SOH)
p_max_allowed = 50 * clip(0.5 + 0.5 * soh_min, 0.5, 1.0)
```

This reduces requested FCS power when components are degraded. It is a soft degradation-management heuristic. The environment may still impose stricter FCS, EDU, battery, or powertrain feasibility limits.

---

### 4.7 Energy action conversion

The requested FCS power is clipped to:

```python
0 <= p_req <= min(50 kW, p_max_allowed)
```

Then converted to normalized energy action:

```python
u_e = 2 * p_req / 50 - 1
```

and to latent prior:

```python
z_energy = arctanh(clip(u_e, -0.9999, 0.9999))
```

Examples:

```python
p_req = 25 kW  -> u_e = 0.0  -> z_energy = 0.0
p_req = 0 kW   -> u_e ≈ -1.0 -> z_energy ≈ -4.605 after clipping
p_req = 50 kW  -> u_e ≈ +1.0 -> z_energy ≈ +4.605 after clipping
```

---

## 5. Relationship to the task objectives

### Driving actor objectives

The driving prior is designed to support:

1. Primary following behavior.
2. Maintaining gap above `D_safe`.
3. Avoiding excessive gap beyond `D_max` when feasible.
4. Closing excessive gaps when constrained only by speed limit or traffic signal.
5. Obeying the current posted speed limit.
6. Crossing stop lines only when the signal is predicted green.
7. Avoiding red and yellow crossings.
8. Approaching lower-speed regions with reasonable entry speed.
9. Reducing abrupt acceleration changes through a comfort prior.

The rule treats following as the dominant driving objective, but it gives very large penalties to collision/overlap and non-green crossing. Speed-limit violations are also penalized strongly so that the prior supports road-rule compliance.

---

### Energy actor objectives

The energy prior is designed to support:

1. Reducing unnecessary hydrogen use by requesting FCS power only when estimated traction demand or SOC regulation requires it.
2. Keeping SOC near 0.5 using a SOC correction term.
3. Reducing abrupt FCS load changes using demand smoothing.
4. Reducing high-power operation when SOH is degraded using an SOH-aware maximum request.
5. Avoiding FCS power requests during braking when estimated road power is negative.

The rule does not directly optimize hydrogen consumption, FCS degradation, battery SOH loss, or EDU degradation because those quantities are not directly observable in the raw observation. It uses approximate physical demand and component SOH as proxies.

---

## 6. Masks and determinism

Both actors are active on every observation:

```python
driving.active = True
energy.active = True
```

The rule never uses an inactive mask to skip fusion.

The rule is deterministic:

- No random numbers.
- No noise.
- No policy std.
- No sampling.
- No online service call.
- No direct SUMO access.
- No reward/configuration modification.

---

## 7. Physical and environment constraints respected

The rule does not bypass:

- Driving action range `[-1.5, 0.7] m/s²`.
- Energy action range `[0, 50] kW`.
- Hard ego speed cap `70 km/h`.
- No-reverse behavior.
- Battery, FCS, EDU, or powertrain feasibility limits.
- Environment clipping of requested actions.

The rule only produces latent priors. The final executed action is still determined by:

```python
z_fused = (1 - lambda) * z_MADDPG + lambda * z_rule
u_fused = tanh(z_fused)
```

and then by the environment’s physical feasibility checks.

---

## 8. Important limitations

1. **Simplified longitudinal predictor.**  
   The rule uses a simplified 10-step constant-acceleration planning model with average-speed integration. SUMO and the environment may integrate motion differently.

2. **Constant lead acceleration assumption.**  
   The lead vehicle’s future acceleration is approximated as constant over the horizon using the currently observed lead acceleration. The actual lead motion may vary.

3. **Candidate persistence is a planning heuristic.**  
   The rule evaluates candidate accelerations as if they influence the ego speed over the horizon, with speed-limit capping. The actual MADDPG actor will recompute actions each step, so the candidate is not a guaranteed future maneuver.

4. **Conservative uncertainty margin is approximate.**  
   The rule adds a one-step conservative travel margin to predicted ego position. This is intended to protect against earlier arrival, but it is not a full stochastic uncertainty model.

5. **Signal prediction is approximate.**  
   Future signal phases are inferred only from the current signal state and remaining phase time using the documented 90 s cycle. The rule does not know exact future signal offsets beyond the observed phase.

6. **Dual arrival-time signal check is more conservative than strictly required.**  
   The rule checks both conservative and nominal arrival times. This may make the prior more conservative near phase boundaries.

7. **Energy power demand is estimated, not observed.**  
   The rule estimates traction power from vehicle mass, rolling resistance, drag, speed, and acceleration. It does not observe motor torque, EDU loss, battery power, hydrogen consumption, or actual FCS output.

8. **SOC correction is heuristic.**  
   The SOC correction is a linear heuristic toward SOC 0.5. It does not model the actual battery charge/discharge efficiency, voltage/current limits, or electrothermal dynamics.

9. **SOH-based power cap is heuristic.**  
   Reducing requested FCS power when SOH is low is a degradation-management approximation. It is not derived from the exact aging model.

10. **Rule fusion is soft.**  
   For `lambda = 0`, the actor action dominates. For `lambda = 1`, the rule prior dominates after `tanh`. In practice, `lambda` should be selected externally and may need to be small if the learned actor has already developed strong behavior.

11. **No validation claim.**  
   This explanation is a static design description only. The code has not been claimed to be executed, tested, or validated in SUMO.