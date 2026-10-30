You are an engineer experienced in reinforcement learning, MADDPG, longitudinal vehicle control, and fuel-cell hybrid electric vehicles (FCHEVs). Based on the project interface below, design an executable Python rule-prior program for the FCHEV environment. The program will be fused with the project's learned MADDPG actors. It is not a standalone controller and must not bypass the environment's physical constraints.

### 1. Project Background and Objectives

This project develops driving and energy-management strategies for a fuel-cell hybrid electric vehicle (FCHEV). SUMO simulates the road, lead vehicle, traffic lights, and vehicle motion. The Python environment couples vehicle longitudinal dynamics with the fuel-cell system (FCS), battery, and electric-drive-unit (EDU) models.

The controller has two independent deterministic MADDPG actors with separate rewards and centralized critics:

- **Driving actor:** Its level-one, dominant task is to follow the lead vehicle. It uses the live ego and lead speeds and accelerations, their distance, and the traffic-light and speed-limit observations described below to choose ego acceleration. It should maintain a safe following gap of at least `D_safe` and, whenever feasible, avoid allowing the gap to exceed the prescribed maximum `D_max`; if the gap is too large, it should act to close it. A gap below `D_safe` is too close and risks collision or overlap. Following remains the primary driving objective, and it must be carried out without violating the current road-region speed limit. The ego vehicle may pass through a signalized intersection only on green: it must cross the stop line only while the signal is green. Red and yellow both prohibit crossing the stop line.
- **Energy actor:** It selects the requested FCS output power. Its objective is to reduce hydrogen consumption and degradation of the FCS, battery, and EDU, while keeping battery SOC close to the target value of 0.5.

Each actor learns from its corresponding reward. The driving actor's dominant objective is following within the safe and prescribed distance range, while obeying speed limits and traffic signals; the energy actor is responsible for power management, component health, and SOC regulation.

The control loop runs at 1 s per step. At every step, each actor makes a new decision using that step's current observation, and the rule program is evaluated once on the same observation to produce its prior(s). Each decision applies to the current step and is recomputed from the newly observed state at the next step.

SOC deviation is an energy-management objective, not a battery degradation metric. The current observation does not directly include motor torque, instantaneous EDU loss, requested FCS power, battery power, or hydrogen consumption. Do not assume these quantities are directly available to the rule. If a rule needs to estimate them, clearly label the estimate as an approximation and explain its limitations.

### 2. Current Simulation Environment and Exact Values

**Road and task:**

- The route has 25 signalized intersections, spaced 1000 m apart.
- An episode is limited to 6000 s; each RL step is 1 s.
- Both the ego vehicle and lead vehicle start at 0 m/s, with an initial gap of 10 m.
- **Following task:** the ego vehicle follows a lead vehicle simulated by SUMO. The lead vehicle's speed and acceleration can change during the trip, including slowing down and stopping as part of the scenario. At every 1 s decision step, the observation provides its current speed and acceleration, along with the ego speed, acceleration, and inter-vehicle gap. Evaluate the rule at every step and return an active prior for both actors at every step. Use these live observations to choose the current action, maintaining a safe following gap and avoiding an unnecessarily large gap. Do not assume a fixed lead-vehicle speed or driving profile.
- There are 24 road segments between adjacent signalized intersections. Scenario details, including the lead vehicle's driving behavior, may vary with the configured scenario and seed.
- **Prediction horizon and physical capability:** Each control step is 1 s. From the current observation, the rule may forecast likely vehicle and powertrain states over the next few steps and use the expected consequences to choose the current driving and energy priors. Balance following distance and safety, speed limits, traffic lights, and energy objectives rather than optimizing one term in isolation. When a predicted risk can be mitigated within the available vehicle and powertrain capabilities, keep following as the primary driving objective: choose a feasible action that addresses the risk while maintaining or closing the safe gap whenever road and signal rules allow. Do not abandon following or default to stopping merely because a manageable risk is predicted. This priority never permits crossing the stop line on red or yellow or exceeding the active speed limit. Account for the ego acceleration request range of −1.5 to +0.7 m/s², the no-reverse constraint, the 70 km/h vehicle speed cap, the FCS maximum output of 50 kW, and battery/EDU operating limits; requested acceleration and power may not be fully delivered. This is a look-ahead for the current decision, not an open-loop action held for multiple steps: recompute the prior at the next 1 s step using the new observation. Any predicted state or power demand not directly observed is an estimate.
- Each traffic light has a 90 s cycle: 40 s green, 3 s yellow, and 47 s red. Initial phases differ across lights, so they are not synchronized.
- The ego vehicle's ordinary-road speed limit is 60 km/h (16.6667 m/s). For intersection safety, its posted speed limit becomes 30 km/h (8.3333 m/s) starting 100 m before the stop line; this zone continues through the intersection and returns to 60 km/h after the vehicle leaves it. In either region, the ego vehicle must not exceed that region's maximum permitted speed. The observation provides the current and next speed limits and the distance to the next speed-region entrance; the rule may use these inputs according to its own design.
- Speed-limit and traffic-light information are represented separately in the observation: a red light does not turn the speed-limit observation into zero. The 100 m intersection approach zone defines the ego's 30 km/h posted speed limit; it is not a gate for considering the traffic-light phase. At every decision step, estimate time to the stop line as t₁ = distance/current ego speed and compare it with the remaining time t₂ in the current phase. This comparison indicates whether arrival is before or after the current phase boundary. In addition, whenever t₁ < 90 s (one complete signal cycle), predict the signal state at the estimated arrival time: if t₁ < t₂, it is the currently observed state; otherwise, advance from the end of the current phase through the configured repeating sequence (green 40 s, yellow 3 s, red 47 s; 90 s total) until reaching the estimated arrival time. Use both the current-phase comparison and the predicted arrival state when choosing the action. The ego may cross the stop line only when the signal state at the actual crossing instant is green. Crossing on red or yellow is forbidden. If the predicted state at crossing is red or yellow, choose an action that keeps the vehicle before the stop line, then reevaluate each step and proceed only when the crossing state is green. In particular, t₁ ≥ t₂ alone does not mean the vehicle may accelerate through the intersection. Reevaluate every step with the new observation, and handle zero or near-zero speed without division by zero. This describes the information to consider, not a fixed braking formula. Once inside the 30 km/h zone, the next speed region is the 60 km/h road after the intersection.
- The vehicle's maximum speed is 70 km/h (19.4444 m/s). This is a hard cap: the environment clips acceleration if the requested acceleration would take the vehicle above it.
- The lead vehicle stops at the route's terminal stop. An episode ends normally when the lead vehicle reaches that stop and is nearly stationary; completion of the ego route also ends the episode.

**Vehicle and action bounds:**

- Vehicle mass: 7000 kg; length: 8 m; wheel radius: 0.4 m; frontal area: 5.8 m²; drag coefficient: 0.55; rolling-resistance coefficient: 0.0085; rotational-mass coefficient: 1.05; air density: 1.225 kg/m³; gravity: 9.8 m/s²; road slope: 0°.
- After tanh, the driving action is u_d ∈ [-1, 1]. It maps to requested acceleration as follows: if u_d < 0, a_req = 1.5 u_d ([-1,0] maps to [-1.5,0] m/s²); if u_d ≥ 0, a_req = 0.7 u_d ([0,1] maps to [0,0.7] m/s²).
- The environment enforces nonnegative speed: with a 1 s step it clips acceleration no lower than −v/Δt, so a negative request at standstill becomes 0 m/s² and a moving vehicle can decelerate to rest without reversing. Do not add a rule-level low-speed threshold that changes a negative acceleration request to zero; let the environment handle the no-reverse constraint.
- The ego vehicle's SUMO speed mode is set to 0, and the environment directly sets its speed. SUMO therefore does not supervise ego following, speed-limit compliance, or red-light stopping. Road limits are handled through rewards/termination logic; the separate 70 km/h vehicle cap is a hard constraint.
- After tanh, the energy action is u_e ∈ [-1, 1], mapped to requested FCS power as P_req = 25(u_e+1) kW, i.e. [0,50] kW.
- Maximum FCS output is 50 kW; DC/DC efficiency is 0.95. Actual FCS power may be limited by energy dispatch, EDU demand, and battery physical limits, so it is not guaranteed to equal the request.
- The battery model represents A123 26650 LiFePO₄ cells in a 106-series, 134-parallel pack. Nominal cell voltage is 3.3 V and capacity is 2.3 Ah. Initial and target SOC are both 0.5; SOC bounds are [0.15, 0.9]; maximum discharge and charge rates are 2C and 1C; cell-voltage range is 2.0–3.6 V; initial core, surface, and coolant temperatures are all 25°C. The battery uses a second-order RC electrothermal model and lookup data.
- The EDU uses the complete EPA Generic IPM 150 kW EDU map, which represents the combined motor, inverter, and gearbox boundary; it is not a motor-only map. Gear ratio is 9.5. The dataset does not provide rated efficiency; do not label peak efficiency as rated efficiency. The environment uses the total EDU efficiency/loss map and valid operating region, so do not add inverter or gearbox losses a second time.
- The FCS hydrogen-consumption table comes from the original 50 kW FCS polynomial in Lin and Zheng (2011), without scaling to another rated power. The aging model accounts for startup, low-power, high-power, and load-change degradation. EDU aging uses a reference point of 75 kW mechanical power, 156 Nm torque, and 4591 rpm, with a reference lifetime of 30000 h. Battery SOH is updated by the current electrothermal and aging model.
- There is no following or traffic-light supervisor controlling the ego vehicle. SUMO does not automatically follow the lead or stop the ego at red lights. Battery/FCS/EDU feasibility limits, the no-reverse rule, and the hard maximum-speed limit remain active. Physical constraints may make the executed action differ from the requested action.

### 3. Observation, Actor Inputs, and Units

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

In the current MADDPG experiment configurations, observation normalization is enabled. Before the actor split, SOH indices 1–3 become (1−SOH)×1000; acceleration indices 4 and 6 are divided by 1.5; speed indices 5 and 7 by 20; gap and distance indices 8, 10, and 14 by 1000; signal state index 9 by 2; phase time index 11 by 100; and speed-limit indices 12 and 13 by 20. The driving actor receives normalized indices 4–14 (11 values). The energy actor receives normalized indices [4,5,0,1,2,3] (6 values): ego acceleration, ego speed, SOC, battery SOH, FCS SOH, and EDU SOH. The rule program always receives the complete raw, unnormalized 15-dimensional observation.

### 4. Physical Action Mapping and Execution

Each MADDPG actor produces one deterministic scalar latent value z. The actor applies tanh to obtain its normalized action u=tanh(z). MADDPG has no actor standard deviation. During training, exploration noise is added by the training program to the normalized actor action; the rule program itself must be deterministic and must not add noise:

- Driving actor: u_d = tanh(z_d), then mapped to requested acceleration [-1.5, 0.7] m/s² using the piecewise-linear mapping above;
- Energy actor: u_e = tanh(z_e), mapped to a 0–50 kW FCS power request.

These are independent deterministic outputs from two separate actors; there is no fixed one-to-one correspondence between them. The environment then checks vehicle dynamics and powertrain feasibility. Maximum speed, no-reverse handling, battery power/SOC/current/voltage limits, EDU torque-speed/efficiency validity, and FCS limits can clip or change a request. A rule prior cannot bypass or modify these constraints.

### 5. Current Reward Definition

MADDPG keeps reward channels separate. The values below match the saved reference run `maddpg_maddpg_separate_rewards_g099_500ep_20261028_134519_041524_seed7`; they are not the values in the repository's current `config/default.json`. If another experiment configuration is selected, use that run's saved configuration instead of assuming these coefficients.

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

The main terms are below. A rule may assist with these objectives, but should not assume any one component is equivalent to the total reward.

**How to use these rewards when designing the rule:** These reward functions and coefficients are training signals used by RL; they are provided to explain what behavior the learned actors are encouraged to develop. The generated rule is a reference prior, not an implementation of the reward function. It does not need to reproduce every term, coefficient, or instantaneous reward calculation, and it should not mechanically optimize the reward equations step by step. Use them to understand the task priorities and tradeoffs. The main purpose of the driving prior is to help the ego follow the lead vehicle while keeping a safe, not excessively large gap and obeying speed limits and traffic signals. The main purpose of the energy prior is to help manage FCS power while reducing hydrogen use and component degradation and keeping SOC near its target. Favor coherent behavior that supports these actor objectives; the rule remains a soft prior fused with the learned policy.

The driving critic receives the sum of following, comfort, traffic-light, speeding, green-light crossing bonus, speed-limit-entry shaping, zero-speed negative-acceleration penalty, positive-acceleration-at-limit penalty, and terminal-penalty contributions. The energy critic receives the sum of hydrogen, FCS degradation, battery SOH loss, EDU degradation, and SOC contributions. These are separate rewards for the two MADDPG critics; the trainer does not give both critics one shared sum.

**Following term.** Use the post-action executed ego speed v_(t+1) in m/s and post-action gap d_(t+1) in m:

- D_safe = 2 + 0.5v + 0.0625v²; D_max = 10 + v + 0.0825v²; d_ref = 200 m.
- For 0 ≤ d < D_safe: c_follow = C₀ − C₁(1 − d/D_safe)³, with C₀=5 and C₁=500.
- For D_safe ≤ d ≤ D_max: c_follow = C₀.
- For D_max < d < d_ref: c_follow = C₀ − C₂(d−D_max)/(d_ref−D_max), with C₂=5.
- For d ≥ d_ref: c_follow = C₃ × (ego forward distance during this step / current-region speed limit in m/s), with C₃=5. Forward distance is in meters, so the ratio has units of seconds. This term is 0 when the lead gap is invalid.
- The final following reward is β₀ c_follow.

**Comfort.** c_comfort = −|a_exec,t−a_exec,t−1| / 2.2, using actual executed acceleration in m/s². The reward contribution is β₅ c_comfort. The divisor 2.2 comes from the requested-acceleration range 0.7−(−1.5).

**Energy and degradation.** The implementation penalizes per-step increments: hydrogen term = −β₁ × hydrogen consumed during this step (g); FCS term = −β₂ × 100 × FCS degradation increment; battery term = −β₃ × 100 × battery SOH-loss increment; EDU term = −β₄ × 100 × EDU CLR increment. The factor 100 converts fractional increments to percentage points. The SOC term is −β₆ × |SOC−0.5|.

**Traffic lights.** This environment reward is evaluated every step from the pre-action speed, signal distance, current phase, and its remaining time; it is not restricted to the 100 m posted-speed zone. Estimate time to the stop line as t₁ = distance/current speed and compare it with remaining time t₂ in the current phase. A reward is active only when speed > 0 and t₁ < t₂. If the current phase is green, the raw reward is +C_green; if yellow or red, it is −C_non-green. Current values are C_green=C_non-green=5, multiplied by β₇=1. If t₁ ≥ t₂, no traffic-light reward is active for that step; recompute on the next step. Crossing the stop line on green also gives a one-time +50 bonus. The simulator terminates the episode for a red crossing; a yellow crossing currently does not trigger this termination condition. This reward definition uses the current observed phase; the rule's action decision must additionally use the arrival-state forecast described in the road/task information above.

**Speed limits.** The active posted limit is 30 or 60 km/h. The continuous speeding term uses post-action executed speed: −100 × max(v_(t+1)−v_limit,0)/v_limit. The vehicle's 70 km/h hard cap is separate from the posted limit. Before entering a lower-limit zone, the environment predicts entry speed using an assumed braking magnitude of 0.75 m/s²: v_entry = sqrt(max(0, v²−2×0.75×d_entry)); e = max(0,(v_entry−v_next_limit)/v_next_limit). Entry shaping is β₈×(e_t−e_(t+1)), with β₈=500. It applies only when approaching a lower-limit region; a red light is not treated as a zero speed limit. The implementation suppresses spurious positive boundary bonuses when the speed-region context changes or on a collision-termination step.

The extra penalty for negative acceleration at zero speed, the extra penalty for requesting positive acceleration at the speed limit, and all extra abnormal-termination penalties are currently 0.

### 6. Episode Termination Conditions

- A collision reported by SUMO or a front gap ≤ 0 (vehicle overlap) terminates the episode early; the current additional terminal penalty is 0.
- The simulator terminates early when the signal state at the crossing instant is red; a yellow crossing does not trigger this termination condition, and the current additional penalty is 0.
- The episode terminates early when ego speed reaches or exceeds 120% of the active posted limit; the current additional penalty is 0. The threshold is 36 km/h in a 30 km/h zone and 72 km/h on an ordinary 60 km/h road. The 70 km/h vehicle hard cap makes the ordinary-road 72 km/h termination threshold unreachable. The continuous speeding reward still applies below the termination threshold.
- The episode ends normally when the lead reaches its terminal stop and stops, or when the ego completes its route.
- The 6000 s time limit produces a truncation.
- The 300 m gap diagnostic threshold and 1000 m monitoring threshold are logging-only and do not terminate episodes. There is no long-stationary termination condition.
- Collision, red-light, and speeding terminations currently have no extra negative terminal reward; their main effect is ending the episode.
- Gamma is determined by the selected MADDPG experiment configuration (for example, 0.99 or 0.999) and is not set by this rule interface. Training begins with random replay warm-up; afterward the deterministic actor action receives trainer-side Gaussian exploration noise (configured standard deviation 0.1 in normalized action space). Rule fusion is not applied during the initial random warm-up.
- The environment distinguishes terminated from time-limit truncated. MADDPG stores `terminated` as the replay buffer's done flag; time-limit truncations do not set that mask.

**Rule-design scope.** Evaluate and return both actor priors at every step, using the documented observations and task objectives. The local model is free to design the decision logic; no exact acceleration or FCS-power formula is prescribed. Keep the generated program consistent with the documented observation units and encodings, the selected output mode, and the environment's physical action limits. The rule is a soft prior fused with MADDPG; it does not replace the environment or guarantee the final executed action.

### 7. Rule Output and Fusion: Physical Actions

This prompt requires the rule to return final physical action values:

- `driving.action`: requested acceleration in m/s², within [-1.5, 0.7]. This is not a target-speed command.
- `energy.action`: requested FCS output power in kW, within [0, 50]. This is not battery power.

The rule output is deterministic and contains no `std`. The adapter converts the MADDPG actor's normalized action to physical units, then blends the values independently for each actor:

`a_fused = (1 − λ) × a_MADDPG,physical + λ × a_rule,physical`

For driving, actor normalized action `u_d` maps to acceleration as `1.5u_d` when `u_d<0` and `0.7u_d` otherwise. For energy, `u_e` maps to FCS request `25(u_e+1)` kW. After fusion, the trainer maps the physical result back to the normalized joint action used by replay and centralized critics. `λ` is provided externally and must not be hard-coded in the rule.

### 8. Rule Program Interface

Create a module in `llm-prior/programs/<program_name>.py` with `predict(observation)`. The rule receives the complete raw observation as a NumPy array shaped `(15,)` or `(batch,15)`, validates shape and finite values, handles both single and batch inputs, and deterministically returns both actor entries. Use one scalar per row; `(batch,1)` float32 arrays are recommended. The required output structure is:

```python
{
    "driving": {"action": acceleration_m_s2, "active": driving_mask},
    "energy":  {"action": fcs_power_kw, "active": energy_mask},
}
```

Return `active=True` for both actors on every observation; do not use an inactive mask to skip per-step rule evaluation or fusion. Do not return `std`, sample noise, call an online service, access SUMO directly, change reward/configuration, or bypass physical constraints. Select the program with `--prior-program-name <program_name>`, the mode with `--maddpg-rule-output-mode physical_action`, and the blend with `--maddpg-rule-lambda <value>` where λ∈[0,1]. The first 1,000 steps of a fresh run use prior-free random warm-up; a resumed run skips that warm-up. Evaluation does not add exploration noise.

### 9. Minimal Example: Physical-Action Output

The example recommends −0.75 m/s² when ego speed exceeds the current posted limit. It illustrates only the interface; it is not a complete controller and does not handle following, traffic lights, or energy optimization.

```python
import numpy as np

def predict(observation):
    x = np.asarray(observation, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2 or x.shape[1] != 15 or not np.all(np.isfinite(x)):
        raise ValueError("observation must be finite with shape (15,) or (batch, 15)")
    overspeed = (x[:, 12] > 0.0) & (x[:, 5] > x[:, 12])
    accel = np.where(overspeed, -0.75, 0.0).astype(np.float32).reshape(-1, 1)
    power = np.zeros((x.shape[0], 1), dtype=np.float32)
    return {
        "driving": {"action": accel, "active": np.ones((x.shape[0], 1), dtype=bool)},
        "energy": {"action": power, "active": np.ones((x.shape[0], 1), dtype=bool)},
    }
```

### 10. Required Response

Return (A) the complete Python module with a suggested filename, and (B) a precise explanation of its behavior, observations, triggers, masks, physical-unit blending, limitations, and CLI selection. Explain how the proposed rule relates to the stated task objectives and observations, and describe important limitations or unavailable information. This is a static reasoning explanation only: do not add a test suite, claim that code was executed, or claim the rule has been validated in SUMO.
