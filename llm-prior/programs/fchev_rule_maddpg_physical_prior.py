# llm-prior/programs/fchev_rule_prior.py
# Deterministic FCHEV rule prior for fusion with MADDPG physical actions.
# Output mode: physical_action.
# This program is a soft prior only. It does not bypass environment clipping,
# battery/FCS/EDU feasibility limits, SUMO dynamics, or the hard 70 km/h cap.

import numpy as np


# ---------------------------------------------------------------------------
# Action and environment constants
# ---------------------------------------------------------------------------
DT = 1.0

# Physical driving action bounds after the project's tanh/action mapping.
A_POS = 0.7      # maximum requested acceleration, m/s^2
A_NEG = -1.5     # minimum requested acceleration, m/s^2

# Hard vehicle speed cap: 70 km/h = 19.4444 m/s.
V_HARD_MAX = 19.4444

# Main-road posted limit: 60 km/h = 16.6667 m/s.
V_MAIN_LIMIT = 16.6667

# SOC target and energy limits.
SOC_TARGET = 0.5
FCS_MAX_KW = 50.0

# EDU aging reference point from the project description: 75 kW mechanical power.
EDU_REF_POWER_W = 75000.0

# Traffic-light approach zone: the signal is used for immediate stop/proceed
# only from 100 m before the stop line through the stop line.
TRAFFIC_APPROACH = 100.0
YELLOW_DURATION = 3.0

# Sentinel used by the observation for "no visible target / outside range".
SENTINEL = 30000.0

# Safety / comfort margins.
SAFE_MARGIN = 2.0          # added to D_safe to keep a margin against the reward threshold
STOP_MARGIN = 1.0          # time margin for traffic-light decisions
ENTRY_MARGIN = 0.5         # speed margin before entering a lower-limit region
COMFORT_DELTA = 0.75       # maximum non-emergency acceleration change from current acceleration

EPS = 1e-6

# Vehicle constants from the project background.
MASS = 7000.0
ROT_MASS_COEFF = 1.05
M_EFF = MASS * ROT_MASS_COEFF

DRAG_K = 0.5 * 1.225 * 0.55 * 5.8
ROLL_FORCE = 0.0085 * MASS * 9.8


def _safe_gap(v):
    """Project following reward's D_safe, m."""
    return 2.0 + 0.5 * v + 0.0625 * v * v


def _max_gap(v):
    """Project following reward's D_max, m."""
    return 10.0 + v + 0.0825 * v * v


def predict(observation):
    """
    Deterministic FCHEV rule prior.

    Input
    -----
    observation : np.ndarray
        Raw, unnormalized 15-dimensional observation or batch of observations.
        Shapes: (15,) or (batch, 15).

    Output
    ------
    dict
        {
            "driving": {"action": acceleration_m_s2, "active": driving_mask},
            "energy":  {"action": fcs_power_kw,      "active": energy_mask},
        }

        action arrays are float32 with shape (batch, 1).
        active masks are bool with shape (batch, 1).
    """
    x = np.asarray(observation, dtype=np.float32)

    # Accept single observation or batch.
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if x.ndim != 2 or x.shape[1] != 15 or not np.all(np.isfinite(x)):
        raise ValueError("observation must be finite with shape (15,) or (batch, 15)")

    n = x.shape[0]

    # -----------------------------------------------------------------------
    # Raw observation unpacking.
    # -----------------------------------------------------------------------
    soc = x[:, 0]
    battery_soh = x[:, 1]
    fcs_soh = x[:, 2]
    edu_soh = x[:, 3]

    a_current = np.clip(x[:, 4], A_NEG, A_POS)
    v = np.clip(x[:, 5], 0.0, V_HARD_MAX)

    # Lead acceleration is used only for short-horizon prediction. Clip it to a
    # reasonable range to avoid numerical extremes; the environment remains the
    # authority on actual motion.
    a_lead = np.clip(x[:, 6], -2.0, 2.0)
    v_lead = np.clip(x[:, 7], 0.0, V_HARD_MAX)

    gap = x[:, 8]

    signal_state = x[:, 9]
    dist_light = x[:, 10]
    phase_time = x[:, 11]

    v_limit = x[:, 12]
    next_limit = x[:, 13]
    d_next = x[:, 14]

    # Speed-limit fallback. The observation normally provides 16.6667 or 8.3333.
    v_limit = np.where((v_limit > 0.0) & np.isfinite(v_limit), v_limit, V_MAIN_LIMIT)
    v_limit = np.clip(v_limit, 1.0, V_HARD_MAX)

    next_limit_valid = (
        (next_limit > 0.0)
        & np.isfinite(next_limit)
        & (d_next >= 0.0)
        & (d_next < SENTINEL)
    )

    # -----------------------------------------------------------------------
    # 1. Following constraint.
    #
    # One-step constant-acceleration prediction:
    #   d_pred = gap + (v_lead - v) * dt + 0.5 * (a_lead - a_req) * dt^2
    #
    # Solve for acceleration bounds that keep d_pred between D_safe and D_max.
    # This is a short-horizon approximation; actual lead behavior may change.
    # -----------------------------------------------------------------------
    v_ref = v  # Following reward thresholds use ego speed.
    d_safe = _safe_gap(v_ref) + SAFE_MARGIN
    d_max = _max_gap(v_ref)
    d_max = np.maximum(d_max, d_safe + 5.0)

    delta_v = v_lead - v

    a_follow_min = np.clip(
        a_lead + 2.0 * (gap + delta_v - d_max),
        A_NEG,
        A_POS,
    )
    a_follow_max = np.clip(
        a_lead + 2.0 * (gap + delta_v - d_safe),
        A_NEG,
        A_POS,
    )

    # Gap too small: predicted gap would violate safe following, or overlap is already present.
    gap_too_small = ((a_follow_max < -0.1) | (gap < 0.0)) & (gap < SENTINEL)

    # Gap too large: predicted gap would exceed D_max, or the gap is at the sentinel maximum.
    gap_too_large = (
        ((a_follow_min > 0.1) & (gap < SENTINEL))
        | (gap >= SENTINEL - 1.0)
    )

    # -----------------------------------------------------------------------
    # 2. Current posted speed-limit constraint.
    # -----------------------------------------------------------------------
    a_speed_max = np.clip((v_limit - v) / DT, A_NEG, A_POS)
    overspeed = v > v_limit + 0.1
    speed_limit_enforce = v > (v_limit - 0.25)

    # -----------------------------------------------------------------------
    # 3. Lower-limit entry shaping.
    #
    # The project's entry-shaping reward predicts entry speed using 0.75 m/s^2
    # braking. The rule uses the same prediction to request braking before a
    # lower-limit region if needed.
    # -----------------------------------------------------------------------
    approach_lower = next_limit_valid & (v_limit > next_limit + 0.1)

    v_entry = np.sqrt(np.maximum(0.0, v * v - 2.0 * 0.75 * d_next))
    entry_excess = approach_lower & (v_entry > next_limit + ENTRY_MARGIN)

    a_entry_required = np.clip(
        (next_limit * next_limit - v * v)
        / (2.0 * np.maximum(d_next, 1.0)),
        A_NEG,
        A_POS,
    )

    a_speed_region_max = np.where(
        entry_excess,
        np.minimum(a_speed_max, a_entry_required),
        a_speed_max,
    )
    speed_region_brake = entry_excess

    # -----------------------------------------------------------------------
    # 4. EDU degradation soft cap.
    #
    # Approximation: estimate traction mechanical power from vehicle dynamics
    # and limit acceleration so estimated mechanical power remains near the
    # EDU aging reference point of 75 kW, scaled by EDU SOH.
    #
    # This is not EDU electrical loss, inverter loss, gearbox loss, or motor
    # torque. The EDU map already represents a combined boundary in the project.
    # -----------------------------------------------------------------------
    f_resistance = DRAG_K * v * v + ROLL_FORCE
    edu_ref_power_w = EDU_REF_POWER_W * np.clip(edu_soh, 0.8, 1.0)

    a_edu_max = np.where(
        v > 0.1,
        (edu_ref_power_w / v - f_resistance) / M_EFF,
        A_POS,
    )
    a_edu_max = np.clip(a_edu_max, A_NEG, A_POS)
    edu_cap_binds = a_edu_max < (A_POS - 0.05)

    # -----------------------------------------------------------------------
    # 5. Traffic-light immediate decision.
    #
    # Signal phase is used only when the ego is in the 100 m approach zone or
    # at the stop line. Traffic-light distance 0 is valid. Sentinel 30000 m is
    # ignored.
    #
    # Cycle order:
    #   green -> yellow -> red -> green
    #
    # Red crossing terminates. Yellow crossing does not terminate but receives
    # negative traffic prediction reward in the project reward definition.
    # -----------------------------------------------------------------------
    traffic_approach = (
        (dist_light >= -1e-6)
        & (dist_light <= TRAFFIC_APPROACH)
        & (dist_light < SENTINEL)
        & (signal_state >= 0.0)
        & (signal_state <= 2.0)
        & (phase_time >= 0.0)
    )

    dist_safe = np.maximum(dist_light, 0.0)

    arrival_current = np.where(
        v > 0.1,
        dist_safe / np.maximum(v, EPS),
        np.inf,
    )

    # If stopped, use a conservative minimum arrival time assuming maximum
    # positive acceleration 0.7 m/s^2 from rest.
    min_arrival = np.where(
        dist_safe > 1e-3,
        np.sqrt(2.0 * dist_safe / A_POS),
        0.0,
    )

    arrival_decision = np.where(v > 0.1, arrival_current, min_arrival)

    stopping_dist = v * v / (2.0 * 0.75)
    can_stop_comfort = (
        (dist_safe > STOP_MARGIN)
        & (stopping_dist <= dist_safe - STOP_MARGIN)
    )

    red = signal_state < 0.5
    green = (signal_state >= 0.5) & (signal_state < 1.5)
    yellow = signal_state >= 1.5

    # Red: after red ends, the next phase is green. Stop if the ego would cross
    # while the signal is still red.
    red_cross_red = arrival_decision <= phase_time + STOP_MARGIN
    red_stop = traffic_approach & red & (red_cross_red | (dist_safe <= 1e-3))

    # Yellow: after yellow ends, the next phase is red. Stop if the ego cannot
    # safely cross before yellow ends. If it can cross yellow safely, stop only
    # when comfortable stopping is possible, to avoid negative yellow reward.
    yellow_must = traffic_approach & yellow & (arrival_decision > phase_time - STOP_MARGIN)
    yellow_optional = (
        traffic_approach
        & yellow
        & (arrival_decision <= phase_time - STOP_MARGIN)
        & can_stop_comfort
    )
    yellow_stop = yellow_must | yellow_optional

    # Green: after green ends, yellow then red. Stop if crossing would occur in
    # red. If crossing would occur in yellow, stop if comfortable.
    green_must = traffic_approach & green & (
        arrival_decision > phase_time + YELLOW_DURATION + STOP_MARGIN
    )
    green_optional = (
        traffic_approach
        & green
        & (arrival_decision > phase_time - STOP_MARGIN)
        & (arrival_decision <= phase_time + YELLOW_DURATION + STOP_MARGIN)
        & can_stop_comfort
    )
    green_stop = green_must | green_optional

    traffic_stop = red_stop | yellow_stop | green_stop

    a_stop_required = np.where(
        traffic_stop,
        np.clip(
            -v * v / (2.0 * np.maximum(dist_safe, 1.0)),
            A_NEG,
            0.0,
        ),
        A_POS,
    )

    # -----------------------------------------------------------------------
    # 6. Combine driving constraints.
    #
    # Safety, speed-limit, traffic-light, entry-shaping, and EDU cap constraints
    # are treated as upper bounds on acceleration. Following-gap closure is a
    # lower bound used only when higher-priority constraints are not binding.
    # -----------------------------------------------------------------------
    raw = np.full(n, A_POS, dtype=np.float32)

    raw = np.minimum(raw, a_follow_max)
    raw = np.minimum(raw, a_speed_max)
    raw = np.minimum(raw, a_speed_region_max)
    raw = np.minimum(raw, a_edu_max)
    raw = np.minimum(raw, a_stop_required)

    # If overlap is already present, request maximum braking. If stopped, the
    # no-reverse handling below will convert this to zero.
    raw = np.where(gap < 0.0, A_NEG, raw)

    speed_limit_binds = a_speed_max < a_follow_min

    priority_block = (
        gap_too_small
        | overspeed
        | traffic_stop
        | speed_region_brake
        | edu_cap_binds
        | speed_limit_binds
    )

    # Close excessive gap only if no higher-priority constraint prevents it.
    raw = np.where(
        (a_follow_min > raw) & (~priority_block),
        np.maximum(raw, a_follow_min),
        raw,
    )

    # -----------------------------------------------------------------------
    # 7. Comfort smoothing.
    #
    # The current ego acceleration is used as a proxy for the previous executed
    # acceleration. This is approximate because the executed acceleration may
    # differ from the previous requested action due to environment clipping.
    # -----------------------------------------------------------------------
    comfort_block = priority_block | (gap < 0.0)

    raw_before = raw.copy()
    comfort_min = a_current - COMFORT_DELTA
    comfort_max = a_current + COMFORT_DELTA
    raw_comfort = np.clip(raw, comfort_min, comfort_max)

    raw = np.where(comfort_block, raw, raw_comfort)
    comfort_limit_applied = (
        (~comfort_block)
        & (np.abs(raw - raw_before) > 1e-6)
    )

    # No reverse request when stopped. The environment also clips this, but the
    # rule avoids requesting negative acceleration at zero speed.
    raw = np.where((v <= 0.1) & (raw < 0.0), 0.0, raw)

    driving_action = np.clip(raw, A_NEG, A_POS).astype(np.float32).reshape(-1, 1)

    driving_active = (
        gap_too_small
        | overspeed
        | traffic_stop
        | speed_region_brake
        | edu_cap_binds
        | speed_limit_binds
        | speed_limit_enforce
        | comfort_limit_applied
        | (gap_too_large & (v < v_limit - 0.5))
    ).astype(bool).reshape(-1, 1)

    # -----------------------------------------------------------------------
    # 8. Energy rule: approximate FCS power request.
    #
    # The observation does not provide battery power, FCS actual power, EDU
    # loss, hydrogen consumption, or motor torque. The rule estimates traction
    # mechanical power from ego speed and a driving acceleration estimate.
    #
    # If the driving rule is active, use the rule's requested acceleration as
    # the demand estimate. If inactive, use the current observed acceleration.
    # This is an approximation; the fused driving action may differ.
    # -----------------------------------------------------------------------
    a_for_energy = np.where(driving_active, raw, a_current)

    traction_kw = (M_EFF * a_for_energy + f_resistance) * v / 1000.0
    p_demand_kw = np.maximum(0.0, traction_kw)

    braking = a_for_energy < -0.1

    # SOC-based FCS request bands. These are design choices intended to keep
    # SOC near 0.5 while avoiding very low FCS power when the FCS is on.
    base = np.zeros(n, dtype=np.float32)

    base = np.where(soc < 0.20, 50.0, base)
    base = np.where((soc >= 0.20) & (soc < 0.35), 40.0, base)
    base = np.where((soc >= 0.35) & (soc < 0.45), 30.0, base)
    base = np.where((soc >= 0.45) & (soc < 0.50), 20.0, base)

    # During braking, do not request FCS unless SOC is critically low.
    # This reduces hydrogen use and avoids FCS load changes when regen may be
    # available, but it relies on the environment's actual dispatch behavior.
    base = np.where(braking & (soc >= 0.30), 0.0, base)

    # If battery SOH is low and estimated traction demand is high, request more
    # FCS power to reduce battery discharge. This is a proxy for battery SOH
    # preservation; the exact battery aging increment is not observed.
    add = np.zeros(n, dtype=np.float32)
    add = np.where(
        (battery_soh < 0.98) & (p_demand_kw > 30.0) & (soc < 0.60),
        20.0,
        add,
    )
    add = np.where(
        (battery_soh < 0.90) & (p_demand_kw > 30.0) & (soc < 0.60),
        30.0,
        add,
    )

    power = base + add

    # Avoid very low positive FCS power when SOC is not critically low. This is
    # a heuristic against low-power FCS degradation, but the exact low-power
    # threshold is not provided by the observation.
    low_power_mask = (
        (power > 0.0)
        & (power < 15.0)
        & (soc >= 0.45)
        & (soc >= 0.30)
    )
    power = np.where(low_power_mask, 0.0, power)

    # FCS SOH-based power caps. These are heuristics for high-power degradation.
    cap = np.full(n, FCS_MAX_KW, dtype=np.float32)
    cap = np.where(fcs_soh < 0.98, np.minimum(cap, 35.0), cap)
    cap = np.where(fcs_soh < 0.95, np.minimum(cap, 25.0), cap)
    cap = np.where(fcs_soh < 0.90, np.minimum(cap, 15.0), cap)

    # Critical SOC override: if SOC is very low, allow full FCS request despite
    # FCS SOH caps. The environment may still limit the actual dispatch.
    cap = np.where(soc < 0.30, FCS_MAX_KW, cap)

    power = np.clip(power, 0.0, cap)
    power = np.clip(power, 0.0, FCS_MAX_KW).astype(np.float32).reshape(-1, 1)

    energy_active = (
        (soc < 0.45)
        | (soc > 0.55)
        | (power > 1e-6)
        | ((battery_soh < 0.98) & (p_demand_kw > 30.0) & (soc < 0.60))
        | (fcs_soh < 0.95)
    ).astype(bool).reshape(-1, 1)

    return {
        "driving": {
            "action": driving_action,
            "active": driving_active,
        },
        "energy": {
            "action": power,
            "active": energy_active,
        },
    }