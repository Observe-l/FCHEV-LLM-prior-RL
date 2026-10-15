# SUMO-FCHEV SAC / TD3 environment

This repository implements a one-second Gymnasium environment for a 7,000 kg fuel-cell hybrid bus following a lead bus through a deterministic SUMO route. The ego policy chooses requested longitudinal acceleration and 50 kW fuel-cell-system output. An independent plant layer applies the EPA EDU envelope, battery limits, traffic-light stopping, and following-distance constraints before the action is sent to SUMO.

The local source papers are kept under `papers/`. The primary reference is Peng et al. (2024), “Multiple Electric Components Health-Aware Eco-Driving Strategy for Fuel Cell Hybrid Electric Vehicle Based on Soft Actor-Critic Algorithm.” Battery, fuel-cell, and propulsion data are loaded from the existing `dataset/battery/`, `dataset/fcs/`, and `dataset/motor/` files; the original source files are not modified.

## Run

The current `default` Conda environment already contains NumPy, SciPy, PyTorch, Gymnasium, and SUMO 1.19.0 is installed on the machine. From this repository root, the following commands use that environment directly:

```bash
/home/lwh/anaconda3/envs/default/bin/python -m unittest discover -s tests -v
/home/lwh/anaconda3/envs/default/bin/python train.py --algorithm sac
/home/lwh/anaconda3/envs/default/bin/python train.py --algorithm td3
/home/lwh/anaconda3/envs/default/bin/python evaluate.py results/<run>/best.pt --episodes 10
```

For a short smoke run, which caps the episode at 64 steps and starts updates after the first eight transitions:

```bash
/home/lwh/anaconda3/envs/default/bin/python train.py --algorithm sac --episodes 1 --max-steps 64 --learning-starts 8 --batch-size 8
/home/lwh/anaconda3/envs/default/bin/python train.py --algorithm td3 --episodes 1 --max-steps 64 --learning-starts 8 --batch-size 8
```

Use `--gui` with either script for SUMO-GUI. Training defaults to headless SUMO. Use `--seed N` to set the SUMO, NumPy, and PyTorch seeds. Training runs are created under `results/<algorithm>_<timestamp>_seed<seed>/`; they include the resolved configuration, episode summaries, per-step trajectories with all reward terms and component states, losses, evaluations, and `best.pt` / `final.pt` checkpoints. `train.py` defaults to 200 episodes, hidden layers `[256, 128, 64]`, gamma 0.95, a 100,000-transition replay buffer, and batch size 64. SAC uses a squashed Gaussian policy and automatic entropy tuning; TD3 uses delayed actor updates, twin critics, target smoothing, and clipped exploration noise.

## Project layout

- `environment/fchev_env.py`: Gymnasium reset/step interface, action constraints, SUMO synchronization, power balance, and reward assembly.
- `environment/sumo_interface.py`, `environment/scenario/`: TraCI integration and reproducible single-lane road generation.
- `util/`: cached source-data loading, vehicle dynamics, battery electrical/thermal/aging model, FCS and EDU models, degradation, observation scaling, and reward functions.
- `algorithm/`: SAC, TD3, networks, and replay buffer.
- `config/default.json`: vehicle, pack, FCS, EDU, reward, SUMO, and training parameters.
- `tests/`: model checks, agent-update checks, and a live SUMO step/power-balance check.
- `results/`: timestamped runs and validation outputs.

## Data interpretation and physical models

The `dataset/battery/README.txt`, `dataset/motor/README.txt`, and `dataset/fcs/README.md` files define the actual file axes, units, matrix orientation, and source limits. These were read before implementation. Lookup tables are loaded once when the environment is created; no files are opened from `step()`.

The battery uses the 2.3 Ah nominal A123 26650 cell model. The 106-series / 134-parallel pack has 349.8 V nominal voltage, 308.2 Ah nominal capacity, 107.81 kWh nominal energy, and 14,204 cells. One representative cell is solved electrically and thermally; pack power, current, and voltage are obtained through the series/parallel scaling. Discharge current is positive. Charge and discharge select their separate source R/C tables. OCV comes from the supplied mean C/20 curve. SOC is limited to 0.15–0.90, current to 2C discharge / 1C charge, and terminal voltage to 2.0–3.6 V per cell. The quadratic pack-power equation is solved with the stable low-current root. The R/C branches use exact first-order discretization; the two-node thermal equations use backward Euler to remain stable at a one-second step. Core/surface capacitance and resistance stay at representative-cell scale. For (R_u), the implementation selects Lin et al. (2014) Table 1 value 3.19 K/W rather than the separate bundled `run_model.m` value of 15 K/W.

Entropy coefficient data are supplied in mV/K and converted to V/K. Lin/Peng Eq. (21) is used for reversible entropy heat, with temperature in Kelvin; ohmic heat remains nonnegative. The measured charge/discharge tables only cover their documented SOC and temperature axes. Queries outside those axes use the nearest boundary value and set `battery_parameter_table_clipped` in step information; the code does not silently extrapolate. A step is rejected if the resulting SOC, current, or terminal voltage cannot satisfy the configured limits.

Battery aging follows Peng et al. (2024), Eqs. (22)–(25), with (z=0.55), (R=8.314), (E_a=31700-370.3c), and the four Table II (B(c)) samples. Eq. (23) gives the one-way amp-hour throughput (Ah_{20}) at 20% capacity loss. In Eq. (25), (C_n) is represented in coulombs, so (N=3600Ah_{20}/C_n) is dimensionless and equals (Ah_{20}/Q_{cell,Ah}). Eq. (22) then uses current in A and elapsed time in seconds, with the factor of two accounting for charge and discharge throughput over a full cycle. `SOH_bat` is the Peng lifetime-health indicator: it decreases to zero at its modeled EOL throughput. The implied physical capacity is mapped from 100% to 80% over that health interval and feeds back into current/SOC limits. (B(c)) is linearly interpolated over the published C-rate samples and clamped below 0.5C (the configured operating current never exceeds 2C); the clamp is exposed as `battery_aging_c_rate_clipped`. Arrhenius temperature is always Kelvin. This is the explicit unit interpretation used by the implementation and tested in `tests/test_models.py`.

The FCS is the original 50 kW Lin and Zheng (2011) polynomial from `dataset/fcs/fcs_parameters.txt`. Hydrogen rate is interpolated directly from the source table and already includes its modeled purge loss. FCS OFF is handled separately with zero hydrogen consumption. A 0.95 DC/DC efficiency is applied between stack output and the DC bus. The first version does not enforce the paper's ramp-rate limit. Song et al. (2020) degradation values are converted from percent to fractions; startup is counted on OFF-to-ON only, low/high power time uses seconds-to-hours conversion, and load-change damage is applied once per power change.

The motor query uses the EPA Generic IPM 150 kW **complete EDU** map, with motor-shaft rpm and torque axes, 9.5 gear ratio, valid operating mask, electric-power map, and total-loss map. It is not a motor-only map: inverter and gearing losses are already included, and component losses cannot be separated. EPA provides no independent regen test data; its regen area mirrors motoring losses with factor 1.05. The implementation builds interpolators once from valid operating-map nodes and rejects queries outside the valid map hull, torque-speed envelope, mechanical power limit, or speed limit. It does not fill invalid cells or extrapolate. EDU electrical power comes from the native electrical-power lookup. EDU loss for CLR comes from the separate valid-region loss lookup and is constrained nonnegative, including regeneration.

EPA does not provide rated power, rated torque, rated efficiency, or a rated operating point. To make Peng's loss-based motor health metric executable, this project uses the user-selected proxy point (75\,kW, 156\,N\cdot m, 4591\,rpm). Its speed/torque product is 74.9999 kW, so the port definitions agree. The point is inside the EPA valid envelope. Interpolation at that point gives approximately 94.32% efficiency from the native electrical-power map, 79.52 kW electrical power, and 4.98 kW loss from the loss map. These are proxy-reference values, not EPA ratings. CLR is accumulated as (P_{loss}dt/(P_{loss,ref}\cdot 30000\,h)); the 30,000 h lifetime is the supplied Peng assumption.

The bus longitudinal dynamics use the configured SI parameters and the current speed/acceleration. Motor speed and torque are derived from wheel force, 0.40 m wheel radius, and the 9.5 ratio. Positive EDU electrical power means motoring; negative means regeneration. The DC-bus equation is enforced as (P_{EDU,e}=P_{battery}+0.95P_{FCS}), with auxiliary loads set to zero. If requested acceleration exceeds motor/battery capability, a bisection search reduces positive acceleration to the highest feasible value. During braking, the feasible regen torque is used and friction brakes provide the remainder; regen is reduced if battery charge acceptance is insufficient. Step information contains requested and executed acceleration, wheel force/power, motor speed/torque, EDU electrical power/loss, friction-brake force, battery current/power/voltage, FCS power/hydrogen, constraints, and the power-balance residual.

The ego's SUMO speed mode is set to manual so SUMO does not silently change the RL command. Before the step, the environment applies speed, leader-gap, red/yellow light, and physical powertrain limits. It sets the resulting speed and then advances SUMO exactly once; executed acceleration is measured from TraCI speeds. Lead and ego positions, velocities, accelerations, net gap, signal state, collision IDs, and SUMO intervention are recorded. SUMO's initial one-second call loads the two `depart=0` vehicles; it is an initialization/bootstrap step, not an RL transition. Every subsequent `env.step()` advances exactly one configured one-second SUMO tick. SUMO's position integration is semi-implicit; quasi-steady electrical power uses the mean speed over the transition for the kinetic-energy calculation.

The scenario is one straight, flat, single-lane road with 25 signalized intermediate nodes spaced 1 km apart. Signal cycles have differing durations and offsets and repeat identically across episodes. The lead bus uses SUMO IDM and obeys signals. Its speed limit is 69 km/h, `speedDev` is zero for reproducibility, and its SUMO speed factor sets the free-flow target to 13 m/s. The ego uses the 9-element Peng state ([SOC, SOH_{bat}, SOH_{FCS}, SOH_{EDU}, a_h, v_h, a_l, v_l, D_{h,l}]) in SI units. The environment returns those physical values; the trainer applies fixed feature scaling before neural-network input. Actions are ([a_h,P_{FCS}]) in ([-1.5,0.7]\,m/s^2\) and ([0,50]\,kW\).

Peng's empirical (D_{max}=0.0825v^2+v+10) relation is evaluated with speed in km/h, consistent with the source paper's vehicle speed convention. Its time-based safe-distance equation uses SI speed in m/s, seconds, and m/s². The literal paper equation (D_{safe}=v t_d+v^2/a_{emergency}+d_0) is retained. Comfort uses the one-second acceleration difference normalized by the requested acceleration range. Health reward uses positive single-step battery SOH loss and cumulative FCS degradation / EDU CLR, matching the paper's structure while preserving a cost sign for battery damage.

## Verification performed

Validation was run with the local `default` Conda environment and SUMO 1.19.0. All 10 unit/integration tests passed, covering source-data dimensions, battery 1C charge/discharge and pack power, thermal rest stability, battery aging units, FCS OFF and 50 kW reference values, EDU motoring/regen signs and limits, one SAC and TD3 update, and a live SUMO transition that advances exactly one second with near-zero DC-bus balance residual.

A full SUMO route completed all 25 signalized intersections in 2,397 RL steps (2,398 s SUMO time) without collision or safety violation. The baseline gap/speed controller used for this integration run consumed 457.29 g hydrogen; minimum net following gap was 41.97 m. The maximum absolute DC-bus balance residual was `7.28e-12 W`. The rollout took 4.40 s, or about 544.8 `env.step()` calls/s; this timing starts after model and scenario initialization and is a simulator throughput check, not an RL training speed. Detailed values are in `results/full_episode_validation.json`.

Both agents completed one short training episode of 32 steps with seed 37, replay updates beginning after eight transitions, and batch size eight. SAC ran at 21.8 environment steps/s including updates; TD3 ran at 24.7 steps/s including updates. Each checkpoint then completed an eight-step deterministic evaluation with seed 93, with zero collisions. These are end-to-end smoke tests only; one 32-step episode does not establish policy convergence or performance. The default commands above train for 200 episodes, and evaluation can be run for 10 episodes using the shown command.
