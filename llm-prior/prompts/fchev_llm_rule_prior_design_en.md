You are an engineer experienced in reinforcement learning, Soft Actor-Critic (SAC), longitudinal vehicle control, and fuel-cell hybrid electric vehicles (FCHEVs). Design an executable Python rule-prior program for the FCHEV setting below. The program proposes Gaussian action-distribution priors for two independent SAC agents. It is not a standalone controller and must not bypass the environment's physical constraints.

### 1. Project Background and Objectives

This project develops driving and energy-management strategies for a fuel-cell hybrid electric vehicle (FCHEV). SUMO simulates the road, lead vehicle, traffic lights, and vehicle motion. The Python environment couples vehicle longitudinal dynamics with the fuel-cell system (FCS), battery, and electric-drive-unit (EDU) models.

The controller has two independent stochastic SAC agents with separate observations and rewards:

- **Driving agent:** Its level-one, dominant task is to follow the lead vehicle. It uses the live ego and lead speeds and accelerations, their distance, and the traffic-light and speed-limit observations described below to choose ego acceleration. It should maintain a safe following gap of at least `D_safe` and, whenever feasible, avoid allowing the gap to exceed the prescribed maximum `D_max`; if the gap is too large, it should act to close it. A gap below `D_safe` is too close and risks collision or overlap. Following remains the primary driving objective, and it must be carried out without violating the current road-region speed limit. The ego vehicle may pass through a signalized intersection only on green: it must cross the stop line only while the signal is green. Red and yellow both prohibit crossing the stop line.
- **Energy agent:** It selects the requested FCS output power. Its objective is to reduce hydrogen consumption and degradation of the FCS, battery, and EDU, while keeping battery SOC close to the target value of 0.5.

Each agent optimizes its corresponding reward independently. The driving agent's dominant objective is following within the safe and prescribed distance range, while obeying speed limits and traffic signals; the energy agent is responsible for power management, component health, and SOC regulation. Their rewards are not combined into a shared team reward.

The control loop runs at 1 s per step. At every step, both agents make a new decision from that step's current state. The energy agent's observation contains the currently measured ego acceleration and ego speed; it does not contain the driving agent's newly requested acceleration for the same step. The rule program is evaluated once on the complete raw observation to produce both priors. The decisions are applied in the same environment step and recomputed from the newly observed state at the next step.

SOC deviation is an energy-management objective, not a battery degradation metric. The current observation does not directly include motor torque, instantaneous EDU loss, requested FCS power, battery power, or hydrogen consumption. Do not assume these quantities are directly available to the rule. If a rule needs to estimate them, clearly label the estimate as an approximation and explain its limitations.

### 2. Current Simulation Environment and Exact Values

**Road and task:**

- The route has 25 signalized intersections spaced 1000 m apart, with 24 road segments between adjacent intersections. An episode is limited to 6000 s, and each control step is 1 s.
- Both ego and lead vehicles start at 0 m/s with an initial gap of 10 m. The lead vehicle's speed and acceleration vary with the scenario and it stops at the route's terminal stop.
- **Primary driving task — following:** The driving agent's main task is to follow the lead vehicle. Use the current ego and lead speeds and accelerations and the observed gap to keep the ego between the safe following distance `D_safe` and the prescribed maximum distance `D_max` whenever feasible. If the gap is below `D_safe`, reduce closing speed and protect the minimum safe gap. If traffic lights, speed limits, or other constraints cause the gap to exceed `D_max`, treat the situation as catch-up: choose feasible actions that actively reduce the gap toward the `D_safe`–`D_max` range, while continuing to obey speed limits and the green-only crossing requirement. Keep making progress toward the lead when constrained from closing the gap; do not treat an already excessive gap as a reason to remain stopped.
- **Decision frequency and prediction horizon:** Evaluate the rule once at every 1 s step using that step's current observation. For each decision, use a finite horizon `H` of 5–20 steps (5–20 s) to predict likely ego, lead, traffic-signal, speed-limit, and approximate power-demand outcomes. Choose both the driving acceleration prior and energy power prior with reference to the predicted outcomes over this same horizon. Because SUMO's exact within-step motion and integration may differ from a simplified predictor, include a conservative uncertainty margin of one simulation step (1 s) in driving predictions: account for the ego traveling up to one step farther than nominally predicted, and use this earlier-arrival case when assessing following gaps, stop-line crossings, and speed-region entries. Do not extrapolate beyond `H`; recompute the priors from the new observation at the next step.
- **Action ranges and physical capability:** The driving agent requests acceleration in `[-1.5, +0.7] m/s²` (maximum requested braking magnitude 1.5 m/s² and maximum positive acceleration 0.7 m/s²). The energy agent requests FCS output power in `[0, 50] kW`. Ego speed has a separate hard cap of 70 km/h, and the environment prevents reverse motion. Battery, EDU, and powertrain feasibility limits may further restrict actual acceleration or delivered power.
- **Speed limits:** The ordinary-road posted limit is 60 km/h (16.6667 m/s). The ego's posted limit is 30 km/h (8.3333 m/s) beginning 100 m before each stop line, continuing through the intersection, and returning to 60 km/h after it leaves the intersection zone. The observation provides the current limit, next limit, and distance to the next speed-region entrance. A traffic-light red state does not change the speed-limit observation to zero.
- **Traffic signals:** Each signal has a 90 s cycle: 40 s green, 3 s yellow, and 47 s red. Signal phases are offset across intersections. The observation provides the next signal's state, distance to its stop line, and remaining time in its current phase. The ego may cross the stop line only when the signal is green; red and yellow prohibit crossing. For any candidate action predicted to reach the stop line within `H`, use the predicted vehicle motion and signal phase at the crossing time to decide whether that action is allowed. Do not make unbounded future signal predictions beyond `H`; reassess on the next step.
- **Balancing following with road rules:** Following is the primary driving objective. Evaluate speed-limit and signal requirements after following, using the same prediction horizon. When predicted outcomes show a conflict, choose the action that preserves following as much as feasible while respecting the posted speed limit and green-only crossing requirement. If maintaining the desired gap is not feasible under those constraints, obey the road and signal rules and resume following when feasible.
- The episode normally ends when the lead reaches its terminal stop and is nearly stationary; completion of the ego route also ends the episode.

**Vehicle and action bounds:**

- Vehicle mass: 7000 kg; length: 8 m; wheel radius: 0.4 m; frontal area: 5.8 m²; drag coefficient: 0.55; rolling-resistance coefficient: 0.0085; rotational-mass coefficient: 1.05; air density: 1.225 kg/m³; gravity: 9.8 m/s²; road slope: 0°.
- A normalized driving action `u_d ∈ [-1, 1]` maps to requested acceleration as follows: if `u_d < 0`, `a_req = 1.5 u_d` ([-1,0] maps to [-1.5,0] m/s²); if `u_d ≥ 0`, `a_req = 0.7 u_d` ([0,1] maps to [0,0.7] m/s²).
- The environment enforces nonnegative speed: with a 1 s step it clips acceleration no lower than `−v/Δt`, so a negative request at standstill becomes 0 m/s² and a moving vehicle can decelerate to rest without reversing. Do not add a rule-level low-speed threshold that changes a negative acceleration request to zero; let the environment handle the no-reverse constraint.
- The ego vehicle's SUMO speed mode is set to 0, and the environment directly sets its speed. SUMO therefore does not supervise ego following, speed-limit compliance, or red-light stopping. Road limits are handled through rewards/termination logic; the separate 70 km/h vehicle cap is a hard constraint.
- A normalized energy action `u_e ∈ [-1, 1]` maps to requested FCS power as `P_req = 25(u_e+1) kW`, i.e. [0,50] kW.
- Maximum FCS output is 50 kW; DC/DC efficiency is 0.95. Actual FCS power may be limited by energy dispatch, EDU demand, and battery physical limits, so it is not guaranteed to equal the request.
- The battery model represents A123 26650 LiFePO₄ cells in a 106-series, 134-parallel pack. Nominal cell voltage is 3.3 V and capacity is 2.3 Ah. Initial and target SOC are both 0.5; SOC bounds are [0.15, 0.9]; maximum discharge and charge rates are 2C and 1C; cell-voltage range is 2.0–3.6 V; initial core, surface, and coolant temperatures are all 25°C. The battery uses a second-order RC electrothermal model and lookup data.
- The EDU uses the complete EPA Generic IPM 150 kW EDU map, which represents the combined motor, inverter, and gearbox boundary; it is not a motor-only map. Gear ratio is 9.5. The dataset does not provide rated efficiency; do not label peak efficiency as rated efficiency. The environment uses the total EDU efficiency/loss map and valid operating region, so do not add inverter or gearbox losses a second time.
- The FCS hydrogen-consumption table comes from the original 50 kW FCS polynomial in Lin and Zheng (2011), without scaling to another rated power. The aging model accounts for startup, low-power, high-power, and load-change degradation. EDU aging uses a reference point of 75 kW mechanical power, 156 Nm torque, and 4591 rpm, with a reference lifetime of 30000 h. Battery SOH is updated by the current electrothermal and aging model.
- There is no following or traffic-light supervisor controlling the ego vehicle. SUMO does not automatically follow the lead or stop the ego at red lights. Battery/FCS/EDU feasibility limits, the no-reverse rule, and the hard maximum-speed limit remain active. Physical constraints may make the executed action differ from the requested action.

### 3. Observation, Agent Inputs, and Units

The rule program receives the **complete, raw, unnormalized 15-dimensional observation** in the exact order below. When no next traffic light is available, the environment encodes its state as green (1), distance as 30000 m, and remaining phase time as 0 s. The value 30000 m is a sentinel for “no visible target/outside the effective range.” A traffic-light distance of 0 m is still a valid observation: it means the ego is at the stop line or has reached it; it does not mean that no traffic light is present. Do not infer signal absence from a zero or negative distance unless the interface explicitly defines that value as invalid. The next traffic light may be visible far ahead; its phase relevance is evaluated each step from estimated arrival time versus remaining phase time, independently of the 100 m posted-speed zone.

| Index | Name | Unit / encoding |
|---:|---|---|
| 0 | Battery SOC | Fraction, normally [0.15, 0.9] |
| 1 | Battery SOH | Fraction, initial value 1 |
| 2 | FCS SOH | Fraction, initial value 1 |
| 3 | EDU SOH | Fraction, initial value 1 |
| 4 | Current ego acceleration | m/s² |
| 5 | Current ego speed | m/s |
| 6 | Current lead-vehicle acceleration | m/s² |
| 7 | Current lead-vehicle speed | m/s |
| 8 | Ego-to-lead distance | m, maximum observed value 30000 |
| 9 | State of the next traffic light | Red 0, green 1, yellow 2 |
| 10 | Distance to the next traffic-light stop line | m, maximum 30000; 30000 if none |
| 11 | Remaining time in the current signal phase | s, normally [0, 100]; 0 if none |
| 12 | Speed limit in the current region | m/s; 16.6667 on the main road, 8.3333 in the intersection zone |
| 13 | Speed limit in the next speed region | m/s |
| 14 | Distance to the entrance of the next speed region | m, maximum 30000 |

In the current SAC experiment configuration, observation normalization is enabled. Before each actor's input is selected, SOH indices 1–3 become `(1−SOH)×1000`; acceleration indices 4 and 6 are divided by 1.5; speed indices 5 and 7 by 20; gap and distance indices 8, 10, and 14 by 1000; signal state index 9 by 2; phase time index 11 by 100; and speed-limit indices 12 and 13 by 20. The driving agent receives normalized indices 4–14 (11 values). The energy agent receives normalized indices [4,5,0,1,2,3] (6 values): current ego acceleration, ego speed, SOC, battery SOH, FCS SOH, and EDU SOH. At a given step both agent inputs come from the same pre-action environment observation; the energy agent does not receive the driving agent's newly selected action. The rule program always receives the complete raw, unnormalized 15-dimensional observation.

### 4. SAC Actions and Physical Execution

Each independent SAC actor represents a one-dimensional Gaussian policy in latent space. The actor predicts a mean and log standard deviation; `log_std` is clamped to [-20, 2], and `std = exp(log_std)`. During stochastic policy sampling, `z = mean + std × ε`, where `ε ~ Normal(0,1)`, followed by `u = tanh(z)`. Deterministic evaluation uses `u = tanh(mean)`.

- Driving actor: `u_d ∈ [-1,1]`, then the piecewise mapping in Section 2 converts it to requested acceleration in [-1.5, 0.7] m/s².
- Energy actor: `u_e ∈ [-1,1]`, then `P_req = 25(u_e+1) kW` converts it to requested FCS output power in [0,50] kW.
- These are independent actors with separate observations and rewards. Neither actor receives the other actor's current selected action as part of its input.
- The environment then checks vehicle dynamics and powertrain feasibility. Maximum speed, no-reverse handling, battery power/SOC/current/voltage limits, EDU torque-speed/efficiency validity, and FCS limits can clip or change a request. A rule prior cannot bypass or modify these constraints.

### 5. Current Reward Definition

The independent SAC agents keep reward channels separate. The values below match `config/experiments/maddpg_individual_reward_g0999_500ep.json`, which is the current individual-reward reference configuration; they are not guaranteed to match another selected configuration. The driving agent optimizes the driving reward and the energy agent optimizes the energy reward. They do not receive a shared team reward.

| Coefficient | Current value | Main purpose |
|---|---:|---|
| β₀ | 1 | Following |
| β₁ | 0.2 | Hydrogen consumption |
| β₂ | 5000 | FCS degradation increment |
| β₃ | 100000 | Battery SOH loss |
| β₄ | 100000 | EDU degradation increment |
| β₅ | 2 | Driving comfort |
| β₆ | 10 | SOC deviation |
| β₇ | 1 | Traffic-light prediction reward |
| β₈ | 500 | Speed shaping before entering a lower-limit region |

These reward functions and coefficients are training signals that explain the behavior encouraged by RL. The generated rule is a reference prior, not an implementation of the reward function. It does not need to reproduce every term, coefficient, or instantaneous reward calculation, and it should not mechanically optimize the equations step by step. Use them to understand task priorities and tradeoffs. The driving prior should support following with a safe, not excessively large gap while obeying speed limits and signals. The energy prior should support FCS power management, lower hydrogen use and component degradation, and SOC near its target. Keep the two agents' objectives and rewards distinct.

The driving reward is the sum of following, comfort, traffic-light, speeding, green-light crossing bonus, speed-limit-entry shaping, zero-speed negative-acceleration penalty, positive-acceleration-at-limit penalty, and terminal-penalty contributions. The energy reward is the sum of hydrogen, FCS degradation, battery SOH loss, EDU degradation, and SOC contributions.

**Following term.** Use post-action executed ego speed `v_(t+1)` in m/s and post-action gap `d_(t+1)` in m:

- `D_safe = 2 + 0.5v + 0.0625v²`; `D_max = 10 + v + 0.0825v²`; `d_ref = 200 m`.
- For `0 ≤ d < D_safe`: `c_follow = C₀ − C₁(1 − d/D_safe)³`, with `C₀=5` and `C₁=500`.
- For `D_safe ≤ d ≤ D_max`: `c_follow = C₀`.
- For `D_max < d < d_ref`: `c_follow = C₀ − C₂(d−D_max)/(d_ref−D_max)`, with `C₂=5`.
- For `d ≥ d_ref`: `c_follow = C₃ × (ego forward distance during this step / current-region speed limit in m/s)`, with `C₃=5`. Forward distance is in meters, so the ratio has units of seconds. This term is 0 when the lead gap is invalid.
- The final following reward is `β₀ c_follow`.

**Comfort.** `c_comfort = −|a_exec,t−a_exec,t−1| / 2.2`, using actual executed acceleration in m/s². The reward contribution is `β₅ c_comfort`. The divisor 2.2 comes from the requested-acceleration range `0.7−(−1.5)`.

**Energy and degradation.** The implementation penalizes per-step increments: hydrogen term = `−β₁ × hydrogen consumed during this step (g)`; FCS term = `−β₂ × 100 × FCS degradation increment`; battery term = `−β₃ × 100 × battery SOH-loss increment`; EDU term = `−β₄ × 100 × EDU CLR increment`. The factor 100 converts fractional increments to percentage points. The SOC term is `−β₆ × |SOC−0.5|`.

**Traffic lights.** The reward is evaluated every step from pre-action speed, signal distance, current phase, and remaining phase time; it is not restricted to the 100 m posted-speed zone. Estimate time to the stop line as `t₁ = distance/current speed` and compare it with remaining time `t₂` in the current phase. A reward is active only when speed > 0 and `t₁ < t₂`. If the current phase is green, the raw reward is `+C_green`; if yellow or red, it is `−C_non-green`. Current values are `C_green=C_non-green=5`, multiplied by `β₇=1`. If `t₁ ≥ t₂`, no traffic-light reward is active for that step; recompute on the next step. Crossing the stop line on green also gives a one-time +50 bonus. The simulator terminates the episode for a red crossing; a yellow crossing currently does not trigger this termination condition. This reward uses the current observed phase; the rule's decision should consider the predicted phase at crossing within the stated finite horizon.

**Speed limits.** The active posted limit is 30 or 60 km/h. The continuous speeding term uses post-action executed speed: `−100 × max(v_(t+1)−v_limit,0)/v_limit`. The vehicle's 70 km/h hard cap is separate from the posted limit. Before entering a lower-limit zone, the environment predicts entry speed using an assumed braking magnitude of 0.75 m/s²: `v_entry = sqrt(max(0, v²−2×0.75×d_entry))`; `e = max(0,(v_entry−v_next_limit)/v_next_limit)`. Entry shaping is `β₈×(e_t−e_(t+1))`, with `β₈=500`. It applies only when approaching a lower-limit region; a red light is not treated as a zero speed limit. The implementation suppresses spurious positive boundary bonuses when the speed-region context changes or on a collision-termination step.

The extra penalty for negative acceleration at zero speed, the extra penalty for requesting positive acceleration at the speed limit, and all extra abnormal-termination penalties are currently 0.

### 6. Episode Termination Conditions

- A collision reported by SUMO or a front gap ≤ 0 (vehicle overlap) terminates the episode early; the current additional terminal penalty is 0.
- The simulator terminates early when the signal state at the crossing instant is red; a yellow crossing does not trigger this termination condition, and the current additional penalty is 0.
- The episode terminates early when ego speed reaches or exceeds 120% of the active posted limit; the current additional penalty is 0. The threshold is 36 km/h in a 30 km/h zone and 72 km/h on an ordinary 60 km/h road. The 70 km/h vehicle hard cap makes the ordinary-road 72 km/h termination threshold unreachable. The continuous speeding reward still applies below the termination threshold.
- The episode ends normally when the lead reaches its terminal stop and stops, or when the ego completes its route.
- The 6000 s time limit produces a truncation.
- The 300 m gap diagnostic threshold and 1000 m monitoring threshold are logging-only and do not terminate episodes. There is no long-stationary termination condition.
- Collision, red-light, and speeding terminations currently have no extra negative terminal reward; their main effect is ending the episode.

### 7. SAC Gaussian Rule Prior and Fusion

The rule program returns a distribution prior for each actor in the actor's latent space before `tanh`; it does not return physical acceleration, kW, or a sampled action. For actor `i` (driving or energy), let the SAC actor predict `μ_actor^i` and `σ_actor^i = exp(log_std_actor^i)`. The rule returns `μ_prior^i` and nonnegative `σ_prior^i`. The integration wrapper fuses mean and standard deviation independently:

`μ_f^i = (1−λ_μ^i) μ_actor^i + λ_μ^i μ_prior^i`

`σ_f^i = (1−λ_σ^i) σ_actor^i + λ_σ^i σ_prior^i`

The mean and standard-deviation weights are external configuration values, independent of each other and of the rule program. They may also be set separately for driving and energy. Use 0.5 for each weight only as an illustrative default; never hard-code the weights into the rule. `active=False` bypasses fusion for that actor on that observation. If `σ_prior=0` and `λ_σ<1`, the fused policy remains stochastic because the SAC actor still contributes standard deviation.

For stochastic SAC action selection, sample the fused distribution as `z = μ_f + σ_f ε`, where `ε ~ Normal(0,1)`, then compute `u=tanh(z)` and apply the physical mapping in Section 4. Deterministic action selection uses `u=tanh(μ_f)`. The rule changes the actor distribution but does not directly override the sampled or mapped action. Evaluate the rule from the current observation at each 1 s decision step.

### 8. Rule Program Interface

Design a Python module exposing `predict(observation)`. The program receives the full raw observation as a NumPy array shaped `(15,)` or `(batch,15)`, even though the SAC actors receive their separate normalized views. It must convert inputs to `float32`, validate shape and finite values, support single and batch inputs, and deterministically return both agent entries. Use one scalar per row; `(batch,1)` arrays are recommended:

```python
{
    "driving": {"mean": driving_prior_mean, "std": driving_prior_std, "active": driving_active},
    "energy":  {"mean": energy_prior_mean,  "std": energy_prior_std,  "active": energy_active},
}
```

- Each actor has one action dimension. `mean` must be finite and expressed in pre-`tanh` latent Gaussian units. `std` must be finite and nonnegative; it is a standard deviation, not `log_std` or variance.
- `active` is an optional Boolean or batch-shaped Boolean mask; if omitted it defaults to `True`. A conditional rule may set `active=False` when it has no justified prior for that actor/state. The program is still evaluated once each environment step.
- The program may provide a prior for driving, energy, or both. It must not make either actor's decisions depend on an unobserved current action from the other actor.
- Return the same distribution parameters deterministically for the same observation. Do not sample actions inside `predict` or add exploration noise.
- NumPy is allowed. Do not use network services, API keys, online LLM calls, file I/O, environment globals, or SAC internals. The program returns prior parameters only; fusion, SAC sampling, and physical action mapping follow the definitions above.
- Any new thresholds or coefficients must be clearly named rule hyperparameters with an explanation. Do not silently change the simulation environment, rewards, powertrain parameters, or hard constraints. Explicitly identify unavailable physical signals; do not fabricate them.

### 9. Minimal Interface Example: Recommend Braking Above the Current Limit

This example demonstrates the prior interface and latent-mean definition; it is not a complete driving policy and does not handle following, traffic lights, or energy management. The target normalized action is `−0.999`, represented by prior mean `atanh(−0.999)≈−3.8002`. Its prior standard deviation is 0.05. When ego speed does not exceed the current limit, the driving prior is inactive. The energy prior is always inactive.

```python
import numpy as np

TARGET_U = -0.999
TARGET_MEAN = float(np.arctanh(TARGET_U))

def predict(observation):
    x = np.asarray(observation, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != 15:
        raise ValueError("observation must have shape (15,) or (batch, 15)")
    if not np.all(np.isfinite(x)):
        raise ValueError("observation must contain only finite values")

    batch = x.shape[0]
    speed = x[:, 5]
    current_limit = x[:, 12]
    active_drive = (current_limit > 0.0) & (speed > current_limit + 1e-6)
    drive_mean = np.zeros((batch, 1), dtype=np.float32)
    drive_mean[active_drive, 0] = TARGET_MEAN
    drive_std = np.full((batch, 1), 0.05, dtype=np.float32)
    zeros = np.zeros((batch, 1), dtype=np.float32)
    inactive = np.zeros((batch, 1), dtype=bool)
    return {
        "driving": {
            "mean": drive_mean,
            "std": drive_std,
            "active": active_drive.reshape(batch, 1),
        },
        "energy": {"mean": zeros.copy(), "std": zeros.copy(), "active": inactive},
    }
```

For example, if `observation[5]=18 m/s` and `observation[12]=16.6667 m/s`, the driving prior is active and proposes a latent mean near −3.8002; the energy prior is inactive. The final stochastic action still depends on the externally configured fusion weights and SAC actor distribution.

### 10. Required Response

Return (A) the complete Python module with a suggested filename and (B) a concise, precise explanation of its task logic, observations, prior triggers, `active` masks, mean/std values, and limits. Explain how the Gaussian prior would affect the SAC actor after external fusion and `tanh`, and distinguish the prior from the final executed action. Identify assumptions, unavailable signals, and any added rule hyperparameters. Do not provide a test suite, request code execution, or claim that the rule was executed or validated in SUMO.
