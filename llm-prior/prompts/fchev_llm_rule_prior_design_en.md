You are an engineer experienced in reinforcement learning, SAC/HASAC, longitudinal vehicle control, and fuel-cell hybrid electric vehicles (FCHEVs). Based on the project interface below, design an executable Python rule-prior program for the FCHEV environment. The program will be fused with a learned HASAC actor's Gaussian policy. It is not a standalone controller and must not bypass the environment's physical constraints.

### 1. Project Background and Objectives

This project studies driving strategies for a fuel-cell hybrid electric vehicle. SUMO provides the road, lead vehicle, traffic lights, and vehicle-motion simulation. A Python environment connects vehicle longitudinal dynamics with fuel-cell system (FCS), battery, and electric-drive-unit (EDU) models. HASAC has two independent actors: the driving actor controls vehicle acceleration, and the energy actor requests FCS output power. The actors use a centralized critic and jointly optimize the same team reward in HASAC.

The objectives are to complete the driving task while:

- Following the lead vehicle correctly, avoiding rear-end collisions, and accelerating/decelerating smoothly;
- Respecting road and intersection speed limits, responding to traffic lights, and crossing intersections safely;
- Reducing hydrogen consumption, FCS degradation, battery SOH loss, and EDU losses;
- Keeping battery SOC close to its target of 0.5.

The control loop runs at 1 s per step. At every step, each actor makes a new decision using that step's current observation, and the rule program is evaluated once on the same observation to produce its prior(s). Each decision applies to the current step and is recomputed from the newly observed state at the next step.

SOC deviation is an energy-management objective, not a battery degradation metric. The current observation does not directly include motor torque, instantaneous EDU loss, requested FCS power, battery power, or hydrogen consumption. Do not assume these quantities are directly available to the rule. If a rule needs to estimate them, clearly label the estimate as an approximation and explain its limitations.

### 2. Current Simulation Environment and Exact Values

**Road and task:**

- The route has 25 signalized intersections, spaced 1000 m apart.
- An episode is limited to 6000 s; each RL step is 1 s.
- Both the ego vehicle and lead vehicle start at 0 m/s, with an initial gap of 10 m. The lead vehicle's target speed is 13 m/s (46.8 km/h).
- The lead vehicle uses SUMO's IDM car-following model. Its vType parameters are accel=0.7 m/s², decel=1.5 m/s², emergencyDecel=1.5 m/s², minGap=3 m, tau=1.5 s, sigma=0, and speedDev=0. Both lead braking limits match the ego action's maximum requested deceleration of 1.5 m/s². The ego SUMO vType still retains emergencyDecel=6.68 m/s²; its normal action mapping range is [-1.5, 0.7] m/s². The lead is speed-limited to 30 km/h within 200 m of a signal and inside the intersection.
- There are 24 road segments between adjacent signalized intersections. One random stop is assigned to the lead vehicle on each segment; each stop lasts 50–100 s, and the stop point is at least 100 m from either end of that segment. The random scenario is determined by the seed, so signal initial phases and the stop plan are reproducible for a fixed seed.
- Each traffic light has a 90 s cycle: 40 s green, 3 s yellow, and 47 s red. Initial phases differ across lights, so they are not synchronized.
- The ordinary-road speed limit is 60 km/h (16.6667 m/s). The enabled intersection speed zone is 30 km/h (8.3333 m/s), covering the 100 m approach before the stop line and the interior of the intersection. The limit returns to 60 km/h after leaving the intersection.
- Speed-limit and traffic-light information are represented separately in the observation: a red light does not turn the speed-limit observation into zero. Once inside the 30 km/h zone, the next speed region is the 60 km/h road after the intersection.
- The vehicle's maximum speed is 70 km/h (19.4444 m/s). This is a hard cap: the environment clips acceleration if the requested acceleration would take the vehicle above it.
- The lead vehicle stops at the route's terminal stop. An episode ends normally when the lead vehicle reaches that stop and is nearly stationary; completion of the ego route also ends the episode.

**Vehicle and action bounds:**

- Vehicle mass: 7000 kg; length: 8 m; wheel radius: 0.4 m; frontal area: 5.8 m²; drag coefficient: 0.55; rolling-resistance coefficient: 0.0085; rotational-mass coefficient: 1.05; air density: 1.225 kg/m³; gravity: 9.8 m/s²; road slope: 0°.
- After tanh, the driving action is u_d ∈ [-1, 1]. It maps to requested acceleration as follows: if u_d < 0, a_req = 1.5 u_d ([-1,0] maps to [-1.5,0] m/s²); if u_d ≥ 0, a_req = 0.7 u_d ([0,1] maps to [0,0.7] m/s²).
- Reverse motion is prohibited. A negative acceleration request while stopped is physically clipped and cannot make the vehicle move backward.
- The ego vehicle's SUMO speed mode is set to 0, and the environment directly sets its speed. SUMO therefore does not supervise ego following, speed-limit compliance, or red-light stopping. Road limits are handled through rewards/termination logic; the separate 70 km/h vehicle cap is a hard constraint.
- After tanh, the energy action is u_e ∈ [-1, 1], mapped to requested FCS power as P_req = 25(u_e+1) kW, i.e. [0,50] kW.
- Maximum FCS output is 50 kW; DC/DC efficiency is 0.95. Actual FCS power may be limited by energy dispatch, EDU demand, and battery physical limits, so it is not guaranteed to equal the request.
- The battery model represents A123 26650 LiFePO₄ cells in a 106-series, 134-parallel pack. Nominal cell voltage is 3.3 V and capacity is 2.3 Ah. Initial and target SOC are both 0.5; SOC bounds are [0.15, 0.9]; maximum discharge and charge rates are 2C and 1C; cell-voltage range is 2.0–3.6 V; initial core, surface, and coolant temperatures are all 25°C. The battery uses a second-order RC electrothermal model and lookup data.
- The EDU uses the complete EPA Generic IPM 150 kW EDU map, which represents the combined motor, inverter, and gearbox boundary; it is not a motor-only map. Gear ratio is 9.5. The dataset does not provide rated efficiency; do not label peak efficiency as rated efficiency. The environment uses the total EDU efficiency/loss map and valid operating region, so do not add inverter or gearbox losses a second time.
- The FCS hydrogen-consumption table comes from the original 50 kW FCS polynomial in Lin and Zheng (2011), without scaling to another rated power. The aging model accounts for startup, low-power, high-power, and load-change degradation. EDU aging uses a reference point of 75 kW mechanical power, 156 Nm torque, and 4591 rpm, with a reference lifetime of 30000 h. Battery SOH is updated by the current electrothermal and aging model.
- There is no following or traffic-light supervisor controlling the ego vehicle. SUMO does not automatically follow the lead or stop the ego at red lights. Battery/FCS/EDU feasibility limits, the no-reverse rule, and the hard maximum-speed limit remain active. Physical constraints may make the executed action differ from the requested action.

### 3. Observation, Actor Inputs, and Units

The rule program receives the **complete, raw, unnormalized 15-dimensional observation** in the exact order below. When no next traffic light is available, the environment encodes its state as green (1), distance as 30000 m, and remaining phase time as 0 s. The value 30000 m is a sentinel for “no visible target/outside the effective range.”

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

Inside HASAC, the driving actor receives indices 4–14 (11 dimensions); only the signal-state category is divided by 2, while all other values remain in the units above. The energy actor receives 5 values: [current ego acceleration, SOC, battery SOH, FCS SOH, EDU SOH]. The rule program, however, always receives the complete raw 15-dimensional observation, not either actor's split input.

### 4. Physical Action Mapping and Execution

Each actor outputs one scalar Gaussian policy, which is passed through tanh to produce a normalized action:

- Driving actor: u_d = tanh(z_d), then mapped to requested acceleration [-1.5, 0.7] m/s² using the piecewise-linear mapping above;
- Energy actor: u_e = tanh(z_e), mapped to a 0–50 kW FCS power request.

These are independent outputs from two separate actors; there is no fixed one-to-one correspondence between them. The environment then checks vehicle dynamics and powertrain feasibility. Maximum speed, no-reverse handling, battery power/SOC/current/voltage limits, EDU torque-speed/efficiency validity, and FCS limits can clip or change a request. A rule prior cannot bypass or modify these constraints.

### 5. Current Reward Definition

HASAC adds the driving reward and energy reward and gives both actors the **same shared team reward**. The coefficients below are from the current configuration:

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

**Following term.** Ego speed v is in m/s and gap d is in m:

- D_safe = 2 + 0.5v + 0.0625v²; D_max = 10 + v + 0.0825v²; d_ref = 200 m.
- For 0 ≤ d < D_safe: c_follow = C₀ − C₁(1 − d/D_safe)³, with C₀=5 and C₁=500.
- For D_safe ≤ d ≤ D_max: c_follow = C₀.
- For D_max < d < d_ref: c_follow = C₀ − C₂(d−D_max)/(d_ref−D_max), with C₂=5.
- For d ≥ d_ref: c_follow = C₃ × (ego forward distance during this step / current-region speed limit in m/s), with C₃=5. Forward distance is in meters, so the ratio has units of seconds. This term is 0 when the lead gap is invalid.
- The final following reward is β₀ c_follow.

**Comfort.** c_comfort = −|a_t−a_(t−1)| / 2.2, where acceleration change is in m/s². The reward contribution is β₅ c_comfort. The divisor 2.2 comes from the requested-acceleration range 0.7−(−1.5).

**Energy and degradation.** The implementation penalizes per-step increments: hydrogen term = −β₁ × hydrogen consumed during this step (g); FCS term = −β₂ × 100 × FCS degradation increment; battery term = −β₃ × 100 × battery SOH-loss increment; EDU term = −β₄ × 100 × EDU CLR increment. The factor 100 converts fractional increments to percentage points. The SOC term is −β₆ × |SOC−0.5|.

**Traffic lights.** Estimate time to the stop line as t₁ = distance/current speed. A reward is applied only when speed > 0 and t₁ is less than the remaining time t₂ in the current phase. If the current phase is green, the raw reward is +C_green; if yellow or red, it is −C_non-green. Current values are C_green=C_non-green=5, multiplied by β₇=1. Crossing the stop line on green also gives a one-time +50 bonus. Crossing on red still terminates the episode; yellow crossing does not trigger this red-light termination condition.

**Speed limits.** The current posted limit is 30 or 60 km/h. The continuous speeding term is −100 × max(v−v_limit,0)/v_limit. The vehicle's 70 km/h hard cap is separate from the posted limit. Before entering a lower-limit zone, the environment predicts entry speed using an assumed braking magnitude of 0.75 m/s²: v_entry = sqrt(max(0, v²−2×0.75×d_entry)); e = max(0,(v_entry−v_next_limit)/v_next_limit). Entry shaping is β₈×(e_t−e_(t+1)), with β₈=500. It applies only when approaching a lower-limit region; a red light is not treated as a zero speed limit. The implementation suppresses spurious positive boundary bonuses when the speed-region context changes or on a collision-termination step.

The extra penalty for negative acceleration at zero speed, the extra penalty for requesting positive acceleration at the speed limit, and all extra abnormal-termination penalties are currently 0.

### 6. Episode Termination Conditions

- A collision reported by SUMO or a front gap ≤ 0 (vehicle overlap) terminates the episode early; the current additional terminal penalty is 0.
- If the ego crosses a stop line while the signal was red before the crossing, the episode terminates early; the current additional penalty is 0. Yellow crossing does not trigger this condition.
- The episode terminates early when ego speed reaches or exceeds 120% of the active posted limit; the current additional penalty is 0. The threshold is 36 km/h in a 30 km/h zone and 72 km/h on an ordinary 60 km/h road. The 70 km/h vehicle hard cap makes the ordinary-road 72 km/h termination threshold unreachable. The continuous speeding reward still applies below the termination threshold.
- The episode ends normally when the lead reaches its terminal stop and stops, or when the ego completes its route.
- The 6000 s time limit produces a truncation.
- The 300 m gap diagnostic threshold and 1000 m monitoring threshold are logging-only and do not terminate episodes. There is no long-stationary termination condition.
- Collision, red-light, and speeding terminations currently have no extra negative terminal reward; their main effect is ending the episode.
- The current HASAC configuration uses discount factor gamma=0.99. Training uses random warm-up followed by Gaussian-policy sampling; the rule prior participates only in actor-policy stages.
- The environment distinguishes terminated from time-limit truncated; the current training code stores either case as the replay buffer's done flag.

### 7. Fusion of the Rule Prior and HASAC Actor

The rule program does not return physical acceleration or kW. It returns each actor's latent Gaussian parameters before tanh: prior mean μ_prior and standard deviation σ_prior. Each HASAC actor predicts mean and log_std; log_std is clamped to [-20, 2], then σ_actor=exp(log_std). Mean and standard deviation are fused independently:

- μ_f = (1−λ₀) μ_actor + λ₀ μ_prior
- σ_f = (1−λ₁) σ_actor + λ₁ σ_prior

The current values are λ₀=0.5 and λ₁=0.5. During training, the fused distribution is sampled as z=μ_f+σ_f ε, where ε~Normal(0,1); deterministic evaluation uses z=μ_f. Then u=tanh(z), followed by the physical action mapping in Section 4. Thus, the rule affects the action by changing the actor's mean and standard deviation; it does not overwrite the actor's action. σ_prior=0 means the rule recommends a zero-variance prior, but the fused or sampled action is not necessarily deterministic: when λ₁<1, part of the actor's standard deviation remains.

The first 1000 environment steps are replay warm-up, during which actions are sampled uniformly at random and the actor/prior fusion is not called. The prior is used in subsequent actor sampling and HASAC updates.

### 8. Rule Program Input/Output Interface

The module is selected by the external configuration field `llm_prior.program_name` and is loaded from `llm-prior/programs/<program_name>.py`. The command line can override it with `--prior-program-name <program_name>`. The prior must also be enabled in the configuration. The program must implement:

    def predict(observation):
        ...

`observation` is a NumPy array shaped `(15,)` for one state or `(batch, 15)` for a batch, containing the full raw observation described in Section 3. Convert it to float32 and process it internally as a batch; recommended outputs have shape `(batch, 1)`. Return a mapping containing both driving and energy:

    {
        "driving": {"mean": ..., "std": ..., "active": ...},
        "energy":  {"mean": ..., "std": ..., "active": ...},
    }

- Both actors have action dimension 1, so each mean/std is one finite scalar; `(batch,1)` arrays are recommended.
- Mean/std must be in the latent Gaussian space before tanh; std must be finite and nonnegative.
- `active` is an optional Boolean mask, recommended shape `(batch,1)`. If omitted, it defaults to True. `active=False` means the prior is not fused for that actor/state.
- One program can provide a prior for driving, energy, or both. Set `active=False` for an actor the rule does not intend to control.
- Support single and batch inputs; reject invalid dimensions, NaN, and Inf. Return the same distribution parameters deterministically for the same input. Do not sample actions inside `predict`.
- NumPy is allowed. Do not use network services, API keys, online LLM calls, file I/O, environment globals, or HASAC internals. The program only returns distribution parameters; existing HASAC code handles fusion, sampling, and physical mapping.
- Any new thresholds or coefficients must be clearly named rule hyperparameters with an explanation. Do not silently change the simulation environment, rewards, powertrain parameters, or hard constraints. Explicitly identify unavailable physical signals; do not fabricate them.

### 9. Minimal Interface Example: Recommend Braking Above the Current Limit

This minimal example demonstrates the interface and latent-mean definition; it is not a complete driving policy and does not handle following, traffic lights, or energy management. The target normalized action is −0.999, whose pre-tanh mean is atanh(−0.999)≈−3.8002. When ego speed is not above the current limit, the prior is inactive. When above the limit, only the driving prior is active. The energy prior is always inactive.

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
        zeros = np.zeros((batch, 1), dtype=np.float32)
        inactive = np.zeros((batch, 1), dtype=bool)
        return {
            "driving": {
                "mean": drive_mean,
                "std": zeros.copy(),
                "active": active_drive.reshape(batch, 1),
            },
            "energy": {
                "mean": zeros.copy(),
                "std": zeros.copy(),
                "active": inactive,
            },
        }

For example, if observation[5]=18 m/s and observation[12]=16.6667 m/s, the driving prior has mean≈−3.8002, std=0, and active=True. With λ₀=0.5, the fused mean is the average of the actor mean and −3.8002. Gaussian sampling still occurs during training, followed by tanh and physical action mapping. The environment may still clip the result due to maximum speed, no-reverse handling, or powertrain feasibility.

### 10. Required Output

Output only the following two sections, with enough detail for review and direct integration:

**A. Executable Python program**

Provide the complete source code, with a suggested filename `<program_name>.py`, which can be placed in `llm-prior/programs/` and imported by the existing HASAC loader. It must implement `predict(observation)` as specified above. If the rule is parameterized, it may also provide `build_prior(config)`, but prediction capability must still be available. Include input shape/finite-value checks, single and batch handling, outputs for both agents, and necessary comments. Do not provide pseudocode, ellipses, or fragments that depend on unavailable files.

**B. Explanation of the rule's behavior**

Explain the rule's objective, observation indices used, each trigger condition and priority, how mean/std/active are set for both actors, how λ fusion affects actions, how boundaries and missing targets are handled, and any new hyperparameters and their rationale. Include at least one 15-dimensional test observation and a summary of its expected output. Explain how to set `llm_prior.program_name` and run an interface smoke test. Be candid about what the rule can and cannot improve, especially that it cannot guarantee optimal unobserved motor losses or replace the policy learned by HASAC.

Do not claim that the example has already been trained or validated in SUMO. Do not change environment parameters, add observation fields, return physical actions, or let the program bypass episode termination or vehicle/battery/EDU physical constraints.
