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
            a_cmd / abs(ACTION_A_MIN),
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
