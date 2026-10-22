## A. Executable Python program

Suggested filename: `fchev_rule_prior.py`  
Suggested path: `llm-prior/programs/fchev_rule_prior.py`

```python
# fchev_rule_prior.py
#
# Rule prior for FCHEV + HASAC fusion.
#
# This program returns latent Gaussian parameters before tanh:
#   driving: mean/std/active for the driving actor latent z_d
#   energy:  mean/std/active for the energy actor latent z_e
#
# It does NOT return physical acceleration or kW.
# It does NOT bypass SUMO, vehicle dynamics, battery/FCS/EDU feasibility,
# no-reverse handling, speed limits, or episode termination logic.
#
# The rule is a soft prior. HASAC fuses:
#   mu_f = (1 - lambda0) * mu_actor + lambda0 * mu_prior
#   sigma_f = (1 - lambda1) * sigma_actor + lambda1 * sigma_prior
# with current lambda0 = lambda1 = 0.5 in the project configuration.

import numpy as np

# Physical action-mapping constants from the environment specification.
# These are not tunable rule hyperparameters; they match the environment mapping.
ACTION_A_MIN = -1.5      # u_d < 0 maps a_req = 1.5 * u_d, so u_d = a_req / 1.5
ACTION_A_MAX = 0.7       # u_d >= 0 maps a_req = 0.7 * u_d, so u_d = a_req / 0.7
HARD_SPEED_CAP = 19.4444 # 70 km/h hard vehicle speed cap, m/s
FCS_HARD_MAX_KW = 50.0   # FCS request mapping upper bound, kW

_SENTINEL = 30000.0
_EPS = 1e-6

# Rule hyperparameters. These are rule-side thresholds only.
# They do not modify the simulation environment or reward coefficients.
DEFAULT_PARAMS = {
    # Speed-limit and speed-shaping thresholds.
    "speed_limit_margin": 0.98,       # target speed = margin * posted limit
    "speed_trigger_margin": 0.05,     # activate driving prior if speed exceeds posted limit by this margin
    "entry_trigger_margin": 0.01,     # activate entry-shaping prior if predicted entry speed exceeds next limit
    "traffic_trigger_margin": 0.01,   # activate traffic-light prior if speed exceeds safe traffic target

    # Traffic-light and lower-speed-region braking assumptions.
    "brake_assumed": 0.75,            # matches environment entry-speed prediction assumption
    "stop_decel": 1.2,                # comfortable stopping deceleration used by the rule, m/s^2
    "red_arrival_margin": 1.0,        # safety margin when trying to arrive after a red phase
    "yellow_arrival_margin": 0.5,     # safety margin for yellow phase

    # Following rule.
    "gap_gain": 0.2,                  # proportional speed adjustment from gap error, (m/s)/m
    "lead_accel_gain": 0.25,          # anticipatory gain on lead acceleration
    "follow_trigger_margin": 5.0,     # gap margin for "too far" activation
    "follow_speed_margin": 0.5,       # speed mismatch margin for mild following activation
    "large_follow_decel_margin": 2.0, # if following requires > this speed reduction, treat as emergency
    "lead_stopped_speed_margin": 20.0, # margin for stopped-lead close activation
    "emergency_lead_stopped_margin": 50.0, # margin for emergency braking toward stopped lead
    "min_gap_to_lead": 2.0,           # minimum gap used in stopped-lead stopping-distance estimate

    # Comfort / jerk limiting.
    "max_jerk_normal": 0.5,           # normal acceleration-change limit, m/s^3 over 1 s step
    "stop_eps": 0.0,                  # only suppress negative acceleration when speed is exactly zero

    # Latent Gaussian prior shaping.
    "u_clip": 0.9999,                 # clip normalized tanh action before atanh
    "latent_clip": 5.0,               # clip latent mean magnitude
    "mean_gain": 1.0,                 # optional amplification of prior latent mean; 1.0 means direct target latent mean
    "driving_std": 0.15,              # prior std for driving when active
    "energy_std": 0.15,               # prior std for energy when active

    # Energy-management rule.
    "soc_target": 0.5,
    "soc_trigger": 0.02,              # activate energy prior if |SOC - target| exceeds this
    "soc_gain": 100.0,                # kW per unit SOC error; heuristic, not a battery-model coefficient
    "energy_trigger": 5.0,            # activate energy prior if estimated mechanical demand exceeds this, kW
    "fcs_max_rule": 45.0,             # rule-side FCS request cap, kW; environment hard max remains 50 kW

    # Known vehicle constants used only for an approximate road-load power estimate.
    # These are approximations, not observed EDU/battery/FCS signals.
    "mass": 7000.0,
    "rotational_mass_coeff": 1.05,
    "rolling_resistance_coeff": 0.0085,
    "drag_coeff": 0.55,
    "frontal_area": 5.8,
    "air_density": 1.225,
    "gravity": 9.8,
    "accessory_power_w": 0.0,
}


def _get_float(params, key, default):
    """Read a finite float hyperparameter; fall back to default on invalid values."""
    try:
        value = params.get(key, default)
        if value is None:
            return float(default)
        value = float(value)
        if not np.isfinite(value):
            return float(default)
        return value
    except (TypeError, ValueError):
        return float(default)


def _validate_observation(observation):
    """Validate and normalize observation to a 2-D batch array."""
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15:
        raise ValueError("observation must have shape (15,) or (batch, 15)")

    if not np.all(np.isfinite(x)):
        raise ValueError("observation must contain only finite values")

    return x


def _u_to_mean(u, u_clip, latent_clip, mean_gain):
    """
    Convert a target normalized action u in [-1, 1] to a latent mean before tanh.

    The returned mean is clipped to avoid infinite latent values.
    mean_gain = 1.0 gives the direct latent target atanh(u).
    Larger mean_gain makes the prior stronger after lambda fusion, but can
    overdrive if the actor mean already matches the target.
    """
    u = np.clip(u, -u_clip, u_clip)
    mean = np.arctanh(u) * mean_gain
    mean = np.clip(mean, -latent_clip, latent_clip)
    return mean.astype(np.float32)


def _predict(observation, params):
    """Core vectorized rule prior."""
    if params is None:
        params = DEFAULT_PARAMS

    x = _validate_observation(observation)
    batch = x.shape[0]
    y = x.astype(np.float64)

    # -------------------------------------------------------------------------
    # Sanitize rule hyperparameters.
    # -------------------------------------------------------------------------
    speed_limit_margin = min(max(_get_float(params, "speed_limit_margin", 0.98), 0.0), 1.0)
    speed_trigger_margin = min(max(_get_float(params, "speed_trigger_margin", 0.05), 0.0), 100.0)
    entry_trigger_margin = min(max(_get_float(params, "entry_trigger_margin", 0.01), 0.0), 100.0)
    traffic_trigger_margin = min(max(_get_float(params, "traffic_trigger_margin", 0.01), 0.0), 100.0)

    brake_assumed = min(max(_get_float(params, "brake_assumed", 0.75), 0.0), 1.5)
    stop_decel = min(max(_get_float(params, "stop_decel", 1.2), 0.0), 1.5)
    red_arrival_margin = min(max(_get_float(params, "red_arrival_margin", 1.0), 0.0), 1000.0)
    yellow_arrival_margin = min(max(_get_float(params, "yellow_arrival_margin", 0.5), 0.0), 1000.0)

    gap_gain = min(max(_get_float(params, "gap_gain", 0.2), 0.0), 10.0)
    lead_accel_gain = min(max(_get_float(params, "lead_accel_gain", 0.25), 0.0), 10.0)
    follow_trigger_margin = min(max(_get_float(params, "follow_trigger_margin", 5.0), 0.0), 1000.0)
    follow_speed_margin = min(max(_get_float(params, "follow_speed_margin", 0.5), 0.0), 100.0)
    large_follow_decel_margin = min(max(_get_float(params, "large_follow_decel_margin", 2.0), 0.0), 100.0)
    lead_stopped_speed_margin = min(max(_get_float(params, "lead_stopped_speed_margin", 20.0), 0.0), 1000.0)
    emergency_lead_stopped_margin = min(max(_get_float(params, "emergency_lead_stopped_margin", 50.0), 0.0), 1000.0)
    min_gap_to_lead = min(max(_get_float(params, "min_gap_to_lead", 2.0), 0.0), 100.0)

    max_jerk_normal = min(max(_get_float(params, "max_jerk_normal", 0.5), 0.0), 2.2)
    stop_eps = min(max(_get_float(params, "stop_eps", 0.0), 0.0), 10.0)

    u_clip = float(np.clip(_get_float(params, "u_clip", 0.9999), 0.0, 1.0 - 1e-6))
    latent_clip = min(max(_get_float(params, "latent_clip", 5.0), 0.0), 10.0)
    mean_gain = min(max(_get_float(params, "mean_gain", 1.0), 0.0), 3.0)
    driving_std = min(max(_get_float(params, "driving_std", 0.15), 0.0), 5.0)
    energy_std = min(max(_get_float(params, "energy_std", 0.15), 0.0), 5.0)

    soc_target = float(np.clip(_get_float(params, "soc_target", 0.5), 0.0, 1.0))
    soc_trigger = min(max(_get_float(params, "soc_trigger", 0.02), 0.0), 1.0)
    soc_gain = min(max(_get_float(params, "soc_gain", 100.0), 0.0), 1000.0)
    energy_trigger = min(max(_get_float(params, "energy_trigger", 5.0), 0.0), 1000.0)
    fcs_max_rule = min(max(_get_float(params, "fcs_max_rule", 45.0), 0.0), FCS_HARD_MAX_KW)

    mass = min(max(_get_float(params, "mass", 7000.0), 0.0), 1e6)
    rotational_mass_coeff = min(max(_get_float(params, "rotational_mass_coeff", 1.05), 0.0), 10.0)
    rolling_resistance_coeff = min(max(_get_float(params, "rolling_resistance_coeff", 0.0085), 0.0), 1.0)
    drag_coeff = min(max(_get_float(params, "drag_coeff", 0.55), 0.0), 10.0)
    frontal_area = min(max(_get_float(params, "frontal_area", 5.8), 0.0), 100.0)
    air_density = min(max(_get_float(params, "air_density", 1.225), 0.0), 10.0)
    gravity = min(max(_get_float(params, "gravity", 9.8), 0.0), 100.0)
    accessory_power_w = min(max(_get_float(params, "accessory_power_w", 0.0), 0.0), 1e6)

    # -------------------------------------------------------------------------
    # Extract raw observation indices.
    # -------------------------------------------------------------------------
    soc = y[:, 0]
    battery_soh = y[:, 1]
    fcs_soh = y[:, 2]
    edu_soh = y[:, 3]
    a_ego = y[:, 4]
    v = y[:, 5]
    a_lead = y[:, 6]
    v_lead = y[:, 7]
    gap = y[:, 8]
    light_state = y[:, 9]
    light_dist = y[:, 10]
    light_time = y[:, 11]
    limit_cur = y[:, 12]
    limit_next = y[:, 13]
    dist_next = y[:, 14]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # ---------------------------------------------------------------------
        # Basic safe clipping of raw values.
        # ---------------------------------------------------------------------
        v_safe = np.maximum(0.0, v)
        v_phys = np.clip(v_safe, 0.0, HARD_SPEED_CAP)

        v_lead_safe = np.clip(v_lead, 0.0, HARD_SPEED_CAP)
        a_lead_safe = np.clip(a_lead, -10.0, 10.0)

        limit_cur_safe = np.clip(np.maximum(0.0, limit_cur), 0.0, HARD_SPEED_CAP)
        limit_next_safe = np.clip(np.maximum(0.0, limit_next), 0.0, HARD_SPEED_CAP)

        soc_safe = np.clip(soc, 0.0, 1.0)

        # ---------------------------------------------------------------------
        # Driving target speed: speed limit, next lower speed region, traffic light.
        # ---------------------------------------------------------------------
        speed_target = limit_cur_safe * speed_limit_margin
        speed_target = np.minimum(speed_target, HARD_SPEED_CAP * 0.99)
        speed_target = np.maximum(0.0, speed_target)

        # Next lower speed region shaping.
        dist_next_safe = np.clip(np.maximum(0.0, dist_next), 0.0, _SENTINEL)
        lower_next = (
            (limit_next_safe > 0.0)
            & (limit_cur_safe > 0.0)
            & (limit_next_safe < limit_cur_safe - 0.1)
            & (dist_next >= 0.0)
            & (dist_next < _SENTINEL - 1.0)
        )

        # Environment entry-speed prediction uses assumed braking 0.75 m/s^2:
        # v_entry = sqrt(max(0, v^2 - 2 * 0.75 * d_entry)).
        # To enter at or below next limit, current speed should be <=
        # sqrt(next_limit^2 + 2 * brake_assumed * d_entry).
        v_allowed = np.sqrt(
            np.maximum(
                0.0,
                limit_next_safe * limit_next_safe + 2.0 * brake_assumed * dist_next_safe,
            )
        )
        speed_target = np.where(lower_next, np.minimum(speed_target, v_allowed), speed_target)

        # Traffic-light constraint for red/yellow.
        light_dist_safe = np.clip(np.maximum(0.0, light_dist), 0.0, _SENTINEL)
        light_time_safe = np.clip(light_time, 0.0, 100.0)
        light_valid = (light_dist >= 0.0) & (light_dist < _SENTINEL - 1.0)

        # Robust state classification for raw encoding: red 0, green 1, yellow 2.
        red = light_state < 0.5
        yellow = (light_state > 1.5) & (light_state < 2.5)

        # Speed from which the ego can still stop at the stop line with stop_decel.
        v_stop = np.sqrt(np.maximum(0.0, 2.0 * stop_decel * light_dist_safe))

        # Speed that would arrive after the current red/yellow phase, with margin.
        v_arrive_red = np.where(
            light_time_safe > 0.0,
            light_dist_safe / np.maximum(light_time_safe + red_arrival_margin, _EPS),
            np.inf,
        )
        v_arrive_yellow = np.where(
            light_time_safe > 0.0,
            light_dist_safe / np.maximum(light_time_safe + yellow_arrival_margin, _EPS),
            np.inf,
        )

        v_light = np.where(
            red,
            np.minimum(v_stop, v_arrive_red),
            np.where(yellow, np.minimum(v_stop, v_arrive_yellow), speed_target),
        )

        traffic_constraint = light_valid & (red | yellow)
        speed_target = np.where(traffic_constraint, np.minimum(speed_target, v_light), speed_target)
        speed_target = np.maximum(0.0, speed_target)

        # ---------------------------------------------------------------------
        # Following target speed.
        # Uses the same D_safe / D_max forms as the reward definition.
        # ---------------------------------------------------------------------
        gap_safe = np.clip(gap, 0.0, _SENTINEL)
        gap_valid = (gap >= 0.0) & (gap < _SENTINEL - 1.0)

        d_safe = 2.0 + 0.5 * v_phys + 0.0625 * v_phys * v_phys
        d_max = 10.0 + v_phys + 0.0825 * v_phys * v_phys

        # Target gap is the midpoint of the safe following band.
        # This avoids forcing the ego to always sit at D_max.
        gap_target = 0.5 * (d_safe + d_max)

        v_follow = (
            v_lead_safe
            + gap_gain * (gap_safe - gap_target)
            + lead_accel_gain * a_lead_safe
        )

        # If the lead is stopped, cap target speed by a stopping-distance estimate.
        lead_stopped = gap_valid & (v_lead_safe < 0.5)
        v_stop_to_lead = np.sqrt(
            np.maximum(
                0.0,
                2.0 * stop_decel * np.maximum(0.0, gap_safe - min_gap_to_lead),
            )
        )
        v_follow = np.where(lead_stopped, np.minimum(v_follow, v_stop_to_lead), v_follow)

        # If no valid lead gap is visible, do not impose a following speed.
        v_follow = np.where(gap_valid, v_follow, speed_target)

        # Collision/overlap gap: force target speed to zero.
        v_follow = np.where(gap <= 0.0, 0.0, v_follow)

        # Cap following target by the speed/traffic target and by nonnegative speed.
        v_follow = np.maximum(0.0, np.minimum(v_follow, speed_target))

        v_target = np.minimum(speed_target, v_follow)

        # ---------------------------------------------------------------------
        # Driving activation and emergency flags.
        # Priority: collision/hard cap/speed limit/entry/traffic/unsafe gap/large
        # following deceleration are emergency. Mild following adjustments are normal.
        # ---------------------------------------------------------------------
        collision_gap = gap <= 0.0
        unsafe_gap = gap_valid & (gap < d_safe)
        too_far = gap_valid & (gap > d_max + follow_trigger_margin)

        speed_over = v_safe > limit_cur_safe + speed_trigger_margin
        entry_over = lower_next & (v_safe > v_allowed + entry_trigger_margin)
        traffic_over = traffic_constraint & (v_safe > v_light + traffic_trigger_margin)
        hard_cap_over = v_safe > HARD_SPEED_CAP - 0.1

        stopped_lead_close = (
            gap_valid
            & (v_lead_safe < 0.5)
            & (gap < d_max + lead_stopped_speed_margin)
        )

        lead_mismatch = gap_valid & (
            (v_follow > v_safe + follow_speed_margin)
            | (v_follow < v_safe - follow_speed_margin)
        )

        large_follow_decel = gap_valid & (v_follow < v_safe - large_follow_decel_margin)

        active_drive = (
            unsafe_gap
            | too_far
            | speed_over
            | entry_over
            | traffic_over
            | hard_cap_over
            | stopped_lead_close
            | lead_mismatch
            | collision_gap
        )

        emergency = (
            unsafe_gap
            | speed_over
            | entry_over
            | traffic_over
            | hard_cap_over
            | collision_gap
            | large_follow_decel
            | (
                gap_valid
                & (v_lead_safe < 0.5)
                & (gap < d_max + emergency_lead_stopped_margin)
            )
        )

        active_drive_flat = np.asarray(active_drive, dtype=bool)

        # ---------------------------------------------------------------------
        # Convert target speed to target acceleration, then apply comfort limiting.
        # One RL step is 1 s, so v_target - v is a direct acceleration estimate.
        # ---------------------------------------------------------------------
        a_raw = np.where(active_drive_flat, v_target - v_safe, 0.0)

        # Normal mode limits acceleration change for comfort.
        a_cmd_normal = a_ego + np.clip(a_raw - a_ego, -max_jerk_normal, max_jerk_normal)

        # Emergency mode allows a larger acceleration command because safety has
        # priority over comfort. The environment still clips physical execution.
        a_cmd = np.where(emergency, a_raw, a_cmd_normal)

        # Respect the environment's requested-acceleration mapping bounds.
        a_cmd = np.clip(a_cmd, ACTION_A_MIN, ACTION_A_MAX)

        # No reverse: if already stopped, do not request negative acceleration.
        a_cmd = np.where((v_safe <= stop_eps) & (a_cmd < 0.0), 0.0, a_cmd)

        # Convert target acceleration to target normalized driving action u_d.
        u_drive = np.where(
            a_cmd >= 0.0,
            a_cmd / ACTION_A_MAX,
            a_cmd / ACTION_A_MIN,
        )

        mean_drive = _u_to_mean(u_drive, u_clip, latent_clip, mean_gain).reshape(batch, 1)
        std_drive = np.full((batch, 1), driving_std, dtype=np.float32)
        active_drive = active_drive_flat.reshape(batch, 1)

        # ---------------------------------------------------------------------
        # Energy target FCS power request.
        #
        # The observation does not include motor torque, EDU loss, battery power,
        # requested FCS power, or hydrogen consumption. The following mechanical
        # power is only an approximation from speed/acceleration and known vehicle
        # constants. It does not include EDU efficiency/loss map, battery losses,
        # FCS efficiency, DC/DC losses, accessory loads, or exact torque-speed limits.
        # ---------------------------------------------------------------------
        a_est = np.where(active_drive_flat, a_cmd, a_ego)
        a_est = np.clip(a_est, ACTION_A_MIN, ACTION_A_MAX)

        m_eff = mass * rotational_mass_coeff

        v_power = np.clip(v_safe, 0.0, HARD_SPEED_CAP)
        v_avg = np.clip(v_power + 0.5 * a_est, 0.0, HARD_SPEED_CAP)

        rolling_power_w = rolling_resistance_coeff * mass * gravity * v_power
        drag_power_w = (
            0.5
            * air_density
            * drag_coeff
            * frontal_area
            * v_power
            * v_power
            * v_power
        )
        accel_power_w = m_eff * v_avg * a_est

        p_roadload_w = rolling_power_w + drag_power_w + accel_power_w + accessory_power_w
        p_demand_kw = np.maximum(0.0, p_roadload_w) / 1000.0

        # SOC feedback: positive P_soc when SOC is below target, negative when above.
        soc_error = soc_target - soc_safe

        # Mild SOH derating. This is a heuristic to reduce high-power operation when
        # subsystem SOH is low. It does not directly model degradation increments.
        soh_factor = (
            np.clip(0.5 + 0.5 * battery_soh, 0.5, 1.0)
            * np.clip(0.5 + 0.5 * fcs_soh, 0.5, 1.0)
            * np.clip(0.5 + 0.5 * edu_soh, 0.5, 1.0)
        )

        p_soc_kw = soc_gain * soh_factor * soc_error
        p_soc_kw = np.clip(p_soc_kw, -fcs_max_rule, fcs_max_rule)

        p_max_rule = fcs_max_rule * soh_factor
        if fcs_max_rule > 0.0:
            # Keep a minimum usable power unless the rule cap is explicitly zero.
            p_max_rule = np.maximum(p_max_rule, min(10.0, fcs_max_rule))

            # If SOC is very low, allow the full rule cap even when SOH derating is active.
            p_max_rule = np.maximum(
                p_max_rule,
                np.where(soc_safe < 0.25, fcs_max_rule, 0.0),
            )

        p_max_rule = np.clip(p_max_rule, 0.0, FCS_HARD_MAX_KW)

        p_target_kw = p_demand_kw + p_soc_kw
        p_target_kw = np.clip(p_target_kw, 0.0, p_max_rule)

        active_energy_flat = (
            (np.abs(soc_error) > soc_trigger)
            | (p_demand_kw > energy_trigger)
        )

        # Energy action mapping: P_req = 25 * (u_e + 1), u_e in [-1, 1].
        u_energy = p_target_kw / 25.0 - 1.0

        mean_energy = _u_to_mean(u_energy, u_clip, latent_clip, mean_gain).reshape(batch, 1)
        std_energy = np.full((batch, 1), energy_std, dtype=np.float32)
        active_energy = np.asarray(active_energy_flat, dtype=bool).reshape(batch, 1)

    # ---------------------------------------------------------------------
    # Final finite/nonnegative masking.
    # ---------------------------------------------------------------------
    mean_drive = np.where(np.isfinite(mean_drive), mean_drive, 0.0).astype(np.float32)
    std_drive = np.where(
        np.isfinite(std_drive) & (std_drive >= 0.0),
        std_drive,
        0.0,
    ).astype(np.float32)
    active_drive = active_drive & np.isfinite(mean_drive) & np.isfinite(std_drive)
    mean_drive = np.where(active_drive, mean_drive, 0.0).astype(np.float32)
    std_drive = np.where(active_drive, std_drive, 0.0).astype(np.float32)

    mean_energy = np.where(np.isfinite(mean_energy), mean_energy, 0.0).astype(np.float32)
    std_energy = np.where(
        np.isfinite(std_energy) & (std_energy >= 0.0),
        std_energy,
        0.0,
    ).astype(np.float32)
    active_energy = active_energy & np.isfinite(mean_energy) & np.isfinite(std_energy)
    mean_energy = np.where(active_energy, mean_energy, 0.0).astype(np.float32)
    std_energy = np.where(active_energy, std_energy, 0.0).astype(np.float32)

    return {
        "driving": {
            "mean": mean_drive,
            "std": std_drive,
            "active": active_drive,
        },
        "energy": {
            "mean": mean_energy,
            "std": std_energy,
            "active": active_energy,
        },
    }


def predict(observation):
    """Required module-level prediction entry point."""
    return _predict(observation, DEFAULT_PARAMS)


def build_prior(config=None):
    """
    Optional parameterized prior builder.

    The module-level predict() remains available even if the loader does not
    use build_prior(). If build_prior() is used, it returns a callable prior
    with configurable rule hyperparameters. Unknown or invalid config keys are
    ignored or fall back to defaults.
    """
    cfg = config if isinstance(config, dict) else {}
    params = dict(DEFAULT_PARAMS)

    for key in list(params.keys()):
        if key in cfg:
            params[key] = _get_float(cfg, key, params[key])

    def _configured_predict(observation):
        return _predict(observation, params)

    # Some loaders may expect either a callable or an object with .predict.
    _configured_predict.predict = _configured_predict
    return _configured_predict
```

---

## B. Explanation of the rule's behavior

### 1. Overall objective

The rule is a soft prior for two HASAC actors:

- **Driving actor**: recommends a latent Gaussian mean/std for the driving action `z_d`, which is later transformed as `u_d = tanh(z_d)` and mapped to requested acceleration.
- **Energy actor**: recommends a latent Gaussian mean/std for the energy action `z_e`, which is later transformed as `u_e = tanh(z_e)` and mapped to requested FCS power.

The rule aims to assist HASAC with:

1. Safe lead-vehicle following and collision avoidance.
2. Speed-limit compliance and smooth entry into lower-speed regions.
3. Red/yellow traffic-light safety.
4. Driving comfort through normal-mode jerk limiting.
5. Battery SOC regulation near 0.5.
6. Conservative FCS power requests to reduce high-power operation and subsystem stress.

It is not a standalone controller. It does not guarantee optimal following, optimal energy management, optimal EDU loss, optimal hydrogen consumption, or guaranteed avoidance of termination. HASAC still learns the final policy, and the environment still clips or changes requested actions according to physical constraints.

---

### 2. Observation indices used

The program receives the raw 15-dimensional observation:

| Index | Name | Used for |
|---:|---|---|
| 0 | Battery SOC | Energy SOC feedback |
| 1 | Battery SOH | Energy derating heuristic |
| 2 | FCS SOH | Energy derating heuristic |
| 3 | EDU SOH | Energy derating heuristic |
| 4 | Current ego acceleration | Comfort limiting and approximate mechanical demand |
| 5 | Current ego speed | Speed target, following, traffic, energy estimate |
| 6 | Current lead acceleration | Following anticipation |
| 7 | Current lead speed | Following target speed |
| 8 | Ego-to-lead distance | Following safety and activation |
| 9 | Next traffic-light state | Red/yellow safety constraint |
| 10 | Distance to next stop line | Traffic-light stopping/arrival constraint |
| 11 | Remaining phase time | Traffic-light arrival constraint |
| 12 | Current speed limit | Speed-limit compliance |
| 13 | Next speed limit | Lower-speed-region entry shaping |
| 14 | Distance to next speed region | Entry-speed prediction |

The rule does not assume access to unobserved quantities such as motor torque, instantaneous EDU loss, actual battery power, actual FCS output, or hydrogen consumption. Where such quantities are needed, the rule uses clearly labeled approximations.

---

### 3. Driving rule behavior

#### 3.1 Speed-limit target

The rule first forms a conservative speed target:

```text
speed_target = speed_limit_margin * current_limit
speed_target = min(speed_target, 0.99 * hard_speed_cap)
```

`speed_limit_margin = 0.98` is a rule hyperparameter. It keeps the recommended speed below the posted limit to reduce continuous speeding penalty and avoid approaching the 120% termination threshold.

The driving prior is activated for speed-limit violation only when:

```text
ego_speed > current_limit + speed_trigger_margin
```

with `speed_trigger_margin = 0.05 m/s`. This avoids activating the prior for tiny speed differences below the posted limit.

#### 3.2 Next lower speed region

When the next speed region has a lower limit and the distance is valid, the rule predicts the maximum current speed that can enter the next region using the same assumed braking magnitude as the environment:

```text
v_allowed = sqrt(next_limit^2 + 2 * brake_assumed * distance_to_next_region)
```

where `brake_assumed = 0.75 m/s^2`.

If:

```text
ego_speed > v_allowed + entry_trigger_margin
```

the driving prior becomes active and emergency braking is allowed. This supports the β₈ speed-shaping objective before entering a 30 km/h intersection zone.

#### 3.3 Traffic-light rule

For red or yellow lights with a valid stop-line distance:

- The rule computes a stopping speed:

```text
v_stop = sqrt(2 * stop_decel * distance_to_stop_line)
```

- It also computes a speed that would arrive after the current phase:

```text
v_arrive = distance_to_stop_line / (remaining_phase_time + arrival_margin)
```

For red, `arrival_margin = 1.0 s`.  
For yellow, `arrival_margin = 0.5 s`.

The traffic target speed is:

```text
v_light = min(v_stop, v_arrive)
```

Then:

```text
speed_target = min(speed_target, v_light)
```

The prior activates if:

```text
ego_speed > v_light + traffic_trigger_margin
```

with `traffic_trigger_margin = 0.01 m/s`.

This rule tries to avoid arriving during red or yellow phases. It does not guarantee red-light safety because the prior is fused with the actor and the environment may clip the executed action.

For green lights, the rule does not impose a hard constraint. It may still accelerate due to following or speed-limit targets, but it does not specifically guarantee the green-crossing bonus.

#### 3.4 Following rule

The rule uses the same following-distance forms as the reward:

```text
D_safe = 2 + 0.5 v + 0.0625 v^2
D_max  = 10 + v + 0.0825 v^2
```

It uses the midpoint of the safe band as a target gap:

```text
gap_target = 0.5 * (D_safe + D_max)
```

A following target speed is computed as:

```text
v_follow = lead_speed
         + gap_gain * (gap - gap_target)
         + lead_accel_gain * lead_acceleration
```

with:

- `gap_gain = 0.2`
- `lead_accel_gain = 0.25`

If the lead is stopped, the rule additionally caps `v_follow` by a stopping-distance estimate:

```text
v_stop_to_lead = sqrt(2 * stop_decel * max(0, gap - min_gap_to_lead))
```

The final speed target is:

```text
v_target = min(speed_target, v_follow)
```

Activation cases:

- `gap < D_safe`: unsafe following, emergency braking.
- `gap > D_max + follow_trigger_margin`: too far, accelerate if safe.
- `v_follow` differs from ego speed by more than `follow_speed_margin`: mild following adjustment.
- `v_follow < ego_speed - large_follow_decel_margin`: emergency braking.
- Lead stopped and close: activation, with emergency margin controlled by `emergency_lead_stopped_margin`.
- `gap <= 0`: collision/overlap, emergency braking.

The rule does not guarantee collision avoidance. If the ego is already too close or too fast, the requested deceleration may be physically insufficient.

#### 3.5 Acceleration command and comfort

The raw acceleration command is:

```text
a_raw = v_target - ego_speed
```

because the RL step is 1 s.

In normal mode, the rule limits acceleration change:

```text
a_cmd = a_ego + clip(a_raw - a_ego, -max_jerk_normal, max_jerk_normal)
```

with `max_jerk_normal = 0.5 m/s^3`.

In emergency mode, the rule allows a larger acceleration change because safety has priority over comfort:

```text
a_cmd = a_raw
```

Then it clips to the environment action mapping bounds:

```text
a_cmd in [-1.5, 0.7] m/s^2
```

If the ego is exactly stopped and `a_cmd < 0`, the rule sets `a_cmd = 0` to respect the no-reverse rule.

#### 3.6 Driving latent mean

The target acceleration is converted to the normalized driving action:

```text
if a_cmd >= 0:
    u_drive = a_cmd / 0.7
else:
    u_drive = a_cmd / 1.5
```

Then:

```text
u_drive = clip(u_drive, -u_clip, u_clip)
mean_drive = atanh(u_drive) * mean_gain
mean_drive = clip(mean_drive, -latent_clip, latent_clip)
```

Default values:

- `u_clip = 0.9999`
- `latent_clip = 5.0`
- `mean_gain = 1.0`
- `driving_std = 0.15`

If `active_drive = False`, the program returns:

```text
mean_drive = 0
std_drive = 0
active_drive = False
```

so the prior is not fused for that state.

---

### 4. Energy rule behavior

#### 4.1 Approximate mechanical demand

The energy actor requests FCS output power, but the observation does not include EDU torque, EDU loss, battery power, or actual FCS output. The rule estimates mechanical road-load power from ego speed and acceleration using known vehicle constants:

```text
P_roadload =
    rolling resistance power
  + aerodynamic drag power
  + approximate acceleration power
```

The acceleration power uses an average-speed approximation:

```text
v_avg = clip(v + 0.5 * a_est, 0, hard_speed_cap)
P_accel = m_eff * v_avg * a_est
```

where:

```text
m_eff = mass * rotational_mass_coeff
```

This estimate:

- does not include EDU efficiency or loss map,
- does not include battery charge/discharge losses,
- does not include FCS efficiency or DC/DC losses,
- does not include accessory loads unless explicitly set,
- does not know the actual executed action after fusion and physical clipping.

It is only a rough proxy for driving power demand.

The positive demand used for FCS request is:

```text
P_demand_kw = max(0, P_roadload) / 1000
```

If the vehicle is decelerating, the estimated demand may be zero, allowing the rule to reduce FCS request and let the battery handle regen or discharge.

#### 4.2 SOC feedback

The rule uses SOC target 0.5:

```text
soc_error = soc_target - SOC
P_soc_kw = soc_gain * soh_factor * soc_error
```

If SOC is below target, `P_soc_kw` is positive and the rule requests more FCS power to charge the battery.  
If SOC is above target, `P_soc_kw` is negative and the rule requests less FCS power, allowing battery discharge.

`soc_gain = 100 kW per unit SOC error` is a heuristic. It is not a battery-model coefficient. It roughly corresponds to changing SOC by 0.05 in about 1 s with a 5 kW power flow, but the actual battery pack energy, efficiency, voltage, current, and temperature limits are not directly used.

#### 4.3 SOH derating

The rule applies a mild derating factor:

```text
soh_factor =
    clip(0.5 + 0.5 * battery_SOH, 0.5, 1.0)
  * clip(0.5 + 0.5 * FCS_SOH, 0.5, 1.0)
  * clip(0.5 + 0.5 * EDU_SOH, 0.5, 1.0)
```

This reduces both the SOC feedback magnitude and the maximum rule-side FCS request when subsystem SOH is low. This is a heuristic. It does not directly model degradation increments, startup degradation, low-power degradation, high-power degradation, or load-change degradation.

#### 4.4 FCS power target

The target FCS request is:

```text
P_target_kw = P_demand_kw + P_soc_kw
P_target_kw = clip(P_target_kw, 0, P_max_rule)
```

where:

```text
P_max_rule = fcs_max_rule * soh_factor
```

Default `fcs_max_rule = 45 kW`, below the environment hard maximum of 50 kW. This is intended to reduce high-power FCS operation. If SOC is very low, the rule allows the full rule cap even when SOH derating is active.

The rule does not guarantee that the requested FCS power will be physically achievable. Battery limits, EDU demand, FCS limits, and dispatch logic may clip the request.

#### 4.5 Energy activation

The energy prior is active when:

```text
|SOC - 0.5| > soc_trigger
```

or when:

```text
estimated mechanical demand > energy_trigger
```

Default:

- `soc_trigger = 0.02`
- `energy_trigger = 5 kW`

If SOC is near target and estimated demand is low, the energy prior is inactive, allowing the actor to decide.

If SOC is high and demand is low, the rule may request near-zero FCS power. If SOC is low, the rule requests charging power. If demand is high, the rule requests power to support driving demand, subject to the rule cap.

#### 4.6 Energy latent mean

The target FCS power is converted to normalized energy action:

```text
u_energy = P_target_kw / 25 - 1
```

because:

```text
P_req = 25 * (u_e + 1)
```

Then:

```text
u_energy = clip(u_energy, -u_clip, u_clip)
mean_energy = atanh(u_energy) * mean_gain
mean_energy = clip(mean_energy, -latent_clip, latent_clip)
```

Default:

- `energy_std = 0.15`

If `active_energy = False`, the program returns:

```text
mean_energy = 0
std_energy = 0
active_energy = False
```

---

### 5. Effect of HASAC λ fusion

The rule returns latent parameters before tanh. HASAC fuses them independently:

```text
mu_f = 0.5 * mu_actor + 0.5 * mu_prior
sigma_f = 0.5 * sigma_actor + 0.5 * sigma_prior
```

Then:

```text
z = mu_f + sigma_f * epsilon
u = tanh(z)
```

during training, and:

```text
z = mu_f
u = tanh(z)
```

during deterministic evaluation.

Therefore:

- The prior does not overwrite the actor.
- The prior mean is blended with the actor mean.
- The prior std is blended with the actor std.
- If `active = False`, the prior is not fused for that actor/state.
- If `active = True` but `mean = 0`, the prior biases the fused mean toward zero latent action.
- If `std = 0.15`, the prior adds some exploration variance even if the actor std is small.

Because λ₀ = 0.5, a prior mean equal to the direct latent target does not necessarily produce the target physical action if the actor mean is zero. For example, a prior mean of `-1.126` fused with actor mean `0` gives fused mean `-0.563`, not `-1.126`. The rule is intentionally soft. The optional `mean_gain` hyperparameter can amplify the prior if stronger guidance is desired, but too large a gain can overdrive when the actor already matches the target.

---

### 6. Boundary and sentinel handling

The rule treats the following as sentinels or invalid targets:

- `30000 m` for no next traffic light.
- `30000 m` for no next speed region.
- `30000 m` for invalid or maximum observed lead gap.
- Negative distances are clipped or treated as invalid.
- Negative ego speed is clipped to zero for control calculations.
- Speed limits are clipped to `[0, hard_speed_cap]`.
- SOC is clipped to `[0, 1]` for rule calculations.
- SOH values are clipped through the derating factor.
- NaN or Inf observations raise `ValueError`.
- Invalid observation dimensions raise `ValueError`.
- Batch and single-state inputs are both supported.
- Outputs are always shaped `(batch, 1)` for mean, std, and active.

If no next traffic light is visible, the traffic-light constraint is inactive.  
If no next speed region is visible, entry shaping is inactive.  
If no valid lead gap is visible, the following constraint is inactive.

---

### 7. New rule hyperparameters and rationale

Important rule hyperparameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `speed_limit_margin` | 0.98 | Keep target speed below posted limit. |
| `speed_trigger_margin` | 0.05 m/s | Activate speed-limit prior only for meaningful exceedance. |
| `entry_trigger_margin` | 0.01 m/s | Activate lower-speed-region entry prior. |
| `traffic_trigger_margin` | 0.01 m/s | Activate traffic-light prior. |
| `brake_assumed` | 0.75 m/s² | Matches environment entry-speed prediction. |
| `stop_decel` | 1.2 m/s² | Comfortable stopping estimate for traffic and stopped lead. |
| `red_arrival_margin` | 1.0 s | Safety margin for arriving after red. |
| `yellow_arrival_margin` | 0.5 s | Safety margin for yellow. |
| `gap_gain` | 0.2 | Proportional following speed adjustment. |
| `lead_accel_gain` | 0.25 | Anticipate lead acceleration. |
| `follow_trigger_margin` | 5 m | Activate when gap is too far. |
| `follow_speed_margin` | 0.5 m/s | Activate mild following mismatch. |
| `large_follow_decel_margin` | 2.0 m/s | Treat large following deceleration need as emergency. |
| `lead_stopped_speed_margin` | 20 m | Activate stopped-lead close rule. |
| `emergency_lead_stopped_margin` | 50 m | Emergency braking margin for stopped lead. |
| `min_gap_to_lead` | 2 m | Minimum gap in stopped-lead stopping estimate. |
| `max_jerk_normal` | 0.5 m/s³ | Comfort limiting in normal mode. |
| `stop_eps` | 0.0 m/s | Suppress negative acceleration only when exactly stopped. |
| `u_clip` | 0.9999 | Avoid infinite latent means from atanh. |
| `latent_clip` | 5.0 | Limit latent mean magnitude. |
| `mean_gain` | 1.0 | Optional prior amplification after fusion. |
| `driving_std` | 0.15 | Prior std for driving when active. |
| `energy_std` | 0.15 | Prior std for energy when active. |
| `soc_target` | 0.5 | SOC objective. |
| `soc_trigger` | 0.02 | SOC activation threshold. |
| `soc_gain` | 100 kW per SOC error | Heuristic SOC feedback gain. |
| `energy_trigger` | 5 kW | Activate energy prior for estimated demand. |
| `fcs_max_rule` | 45 kW | Rule-side FCS request cap below 50 kW hard max. |

Vehicle constants such as mass, drag coefficient, frontal area, rolling resistance, air density, and rotational-mass coefficient are used only for the approximate road-load power estimate. They are not used to modify the environment.

---

### 8. Example 15-dimensional test observation

Example observation:

```python
obs = np.array(
    [
        0.45,      # 0: SOC below target
        0.98,      # 1: battery SOH
        0.95,      # 2: FCS SOH
        0.97,      # 3: EDU SOH
        0.2,       # 4: ego acceleration
        12.0,      # 5: ego speed
        0.0,       # 6: lead acceleration
        10.0,      # 7: lead speed
        80.0,      # 8: gap
        0.0,       # 9: next light red
        120.0,     # 10: distance to stop line
        20.0,      # 11: remaining red time
        16.6667,   # 12: current speed limit
        8.3333,    # 13: next speed limit
        150.0,     # 14: distance to next speed region
    ],
    dtype=np.float32,
)
```

Interpretation:

- SOC is 0.45, below target 0.5.
- Ego speed is 12 m/s.
- Lead speed is 10 m/s.
- Gap is 80 m.
- Next light is red, 120 m ahead, with 20 s remaining.
- Current limit is 60 km/h.
- Next region is 30 km/h, 150 m ahead.

Expected qualitative behavior:

- The red-light constraint dominates the driving target.
- The rule computes a low target speed to avoid arriving during the red phase.
- Because ego speed is much higher than the traffic target, the driving prior becomes active and emergency braking is recommended.
- The target acceleration is clipped to the environment maximum deceleration, so the driving latent mean is near the maximum braking latent value.
- The estimated mechanical demand is non-positive because the rule recommends hard deceleration, so the energy rule mainly charges the battery due to low SOC.
- The energy prior is active because SOC deviation exceeds 0.02.

Approximate expected output with default parameters:

```python
{
    "driving": {
        "mean": [[-4.9517]],
        "std": [[0.15]],
        "active": [[True]],
    },
    "energy": {
        "mean": [[-1.126]],
        "std": [[0.15]],
        "active": [[True]],
    },
}
```

The exact numbers may vary slightly due to floating-point precision, but the qualitative result should be:

- Driving prior active with a large negative mean, recommending strong braking.
- Energy prior active with a negative mean, requesting a modest FCS power request to charge the battery while decelerating.

With λ₀ = 0.5 and an actor mean of zero, the fused driving mean would be approximately:

```text
mu_f_drive ≈ 0.5 * 0 + 0.5 * (-4.9517) = -2.4759
```

Then:

```text
u_d = tanh(-2.4759) ≈ -0.9866
a_req = 1.5 * u_d ≈ -1.48 m/s^2
```

For energy, with actor mean zero:

```text
mu_f_energy ≈ 0.5 * 0 + 0.5 * (-1.126) = -0.563
u_e = tanh(-0.563) ≈ -0.510
P_req = 25 * (u_e + 1) ≈ 12.25 kW
```

This illustrates that the prior is soft: it biases the action but does not force the exact target physical action.

---

### 9. Configuration and smoke test

Place the file at:

```text
llm-prior/programs/fchev_rule_prior.py
```

Set the configuration field:

```yaml
llm_prior:
  program_name: fchev_rule_prior
  enabled: true
```

or use the command-line override:

```bash
--prior-program-name fchev_rule_prior
```

Because the directory name contains a hyphen, a standalone smoke test can load the file by path:

```bash
python - <<'PY'
import numpy as np
import importlib.util

spec = importlib.util.spec_from_file_location(
    "prior",
    "llm-prior/programs/fchev_rule_prior.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

obs = np.array(
    [
        0.45, 0.98, 0.95, 0.97,
        0.2, 12.0, 0.0, 10.0, 80.0,
        0.0, 120.0, 20.0,
        16.6667, 8.3333, 150.0,
    ],
    dtype=np.float32,
)

out = mod.predict(obs)

print("driving mean:", out["driving"]["mean"])
print("driving std:", out["driving"]["std"])
print("driving active:", out["driving"]["active"])
print("energy mean:", out["energy"]["mean"])
print("energy std:", out["energy"]["std"])
print("energy active:", out["energy"]["active"])

# Batch test
obs_batch = np.vstack([obs, obs])
out_batch = mod.predict(obs_batch)
assert out_batch["driving"]["mean"].shape == (2, 1)
assert out_batch["energy"]["mean"].shape == (2, 1)
assert out_batch["driving"]["active"].shape == (2, 1)
assert out_batch["energy"]["active"].shape == (2, 1)

# Optional parameterized prior test
prior = mod.build_prior({"driving_std": 0.1, "energy_std": 0.1})
out2 = prior(obs)
print("configured driving std:", out2["driving"]["std"])
print("configured energy std:", out2["energy"]["std"])
PY
```

If the existing HASAC loader can import the program directly, the same test can be written as:

```python
import numpy as np
from llm_prior.programs.fchev_rule_prior import predict

obs = np.array([...], dtype=np.float32)
out = predict(obs)
```

depending on the actual package/path structure used by the loader.

---

### 10. What this rule can and cannot improve

The rule can plausibly help with:

- Reducing obvious speed-limit violations.
- Improving entry-speed shaping before lower-limit regions.
- Reducing red/yellow arrival risk.
- Encouraging safe following when gaps are unsafe or too large.
- Reducing abrupt acceleration changes in normal mode.
- Pulling SOC back toward 0.5.
- Avoiding unnecessarily high FCS requests when SOC is high.
- Providing a structured prior during early HASAC training.

The rule cannot guarantee:

- Optimal following reward.
- Optimal comfort reward.
- Optimal hydrogen consumption.
- Optimal FCS degradation.
- Optimal battery SOH preservation.
- Optimal EDU degradation.
- Optimal EDU loss because EDU torque, speed, and loss map are not observed.
- Exact battery charge/discharge feasibility because battery current, voltage, temperature, and internal limits are not directly observed.
- Exact FCS output because actual dispatch may clip the request.
- Collision avoidance if the ego is already too close or too fast.
- Red-light safety if the ego is already too close to the stop line.
- Speeding termination avoidance if the actor overrides the prior.
- Optimal traffic-light crossing bonuses.
- Replacement of the learned HASAC policy.

The rule is a prior, not a supervisor. It changes the distribution used by HASAC but does not bypass physical constraints, reward logic, or episode termination conditions.