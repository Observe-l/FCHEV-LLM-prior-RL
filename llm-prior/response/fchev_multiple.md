Below is a single importable Python rule-prior module. It uses only the raw 15-D observation, returns batch-shaped `(batch,1)` mean/std/active entries for both independent SAC agents, and does not issue actions, sample noise, store hidden episode counters, or override environment constraints.

```python
import numpy as np

# -----------------------------------------------------------------------------
# Module constants
# -----------------------------------------------------------------------------

OBS_DIM = 15

DT = 1.0
HORIZON = 15.0                     # 5-20 s horizon; chosen in range.
UNCERTAIN_TIME = 1.0               # one-step conservative earlier-arrival margin.

ACCEL_MIN = -1.5
ACCEL_MAX = 0.7
BRAKE_SCALE = 1.5
DRIVE_SCALE = 0.7
POWER_SCALE = 25.0
MAX_FCS_KW = 50.0

# Sentinel thresholds.  The environment uses 30000 m as a sentinel for
# "no visible target/outside effective range" for gap/signal/speed distances.
GAP_NO_TARGET = 29000.0
NO_SIGNAL_MIN = 29000.0
NO_LIMIT_MIN = 29000.0

GAP_MIN_VALID = 1e-3
STOP_EPS = 1e-4
STOP_ACCEL_EPS = 1e-6
STOP_MARGIN = 1.0
STOP_LINE_MARGIN = 1.0

SIGNAL_RED = 0
SIGNAL_GREEN = 1
SIGNAL_YELLOW = 2

GREEN_DURATION = 40.0
YELLOW_DURATION = 3.0
RED_DURATION = 47.0

_SIGNAL_DURATION = {
    SIGNAL_GREEN: GREEN_DURATION,
    SIGNAL_YELLOW: YELLOW_DURATION,
    SIGNAL_RED: RED_DURATION,
}
_SIGNAL_NEXT = {
    SIGNAL_GREEN: SIGNAL_YELLOW,
    SIGNAL_YELLOW: SIGNAL_RED,
    SIGNAL_RED: SIGNAL_GREEN,
}

SIGNAL_WATCH_MIN = 120.0
SIGNAL_RISK_TOL = 1e-4

SPEED_WATCH_MIN = 150.0
SPEED_MARGIN = 1.0
SPEED_ENTRY_MARGIN = 0.5
SPEED_COMPARISON_EPS = 1e-3
SPEED_ENTRY_LINE_MARGIN = 1.0

GAP_CATCH_MARGIN = 25.0
CATCHUP_MIN_ACCEL = 0.2
FOLLOW_CLOSE_MARGIN = 1.0
FOLLOW_WIDE_MARGIN = 10.0
SAFE_EXTRA = 0.5

K_GAP = 0.04
K_REL = 0.35
CLOSE_GAIN = 0.15

# Driving prior standard deviations in latent space.
FOLLOW_STD_CLOSE = 0.08
FOLLOW_STD_STABLE = 0.14
FOLLOW_STD_WIDE = 0.22
CATCH_STD = 0.22
SPEED_STD = 0.12
SIGNAL_STD_MEDIUM = 0.12
SIGNAL_STD_STRONG = 0.05
FOLLOW_MILD_STD = 0.30
FOLLOW_MILD_STD_CLOSE = 0.12
FOLLOW_MILD_STD_WIDE = 0.22

# Energy prior standard deviations in latent space.
ENERGY_STD_NORMAL = 0.12
ENERGY_STD_STRONG = 0.08
ENERGY_EXTREME_LOW_KW = 2.0
ENERGY_EXTREME_HIGH_KW = 45.0

MIN_PRIOR_STD = 0.02
MAX_PRIOR_STD = 0.50

# Used only to keep latent Gaussian means finite.  This is slightly inside
# the tanh action bound.  The physical action is still determined by SAC
# sampling and environment feasibility.
U_LIMIT = 0.9975

LEAD_ACCEL_CLIP = 5.0

# Vehicle/FCHEV simplified power model constants.  Used only for an
# approximate mechanical demand estimate.  The environment owns the true
# EDU/FCS/battery feasibility constraints.
MASS = 7000.0
ROT_MASS_COEFF = 1.05
MASS_ROT = MASS * ROT_MASS_COEFF
FRONT_AREA = 5.8
DRAG_CD = 0.55
AIR_DENSITY = 1.225
CRR = 0.0085
GRAVITY = 9.8

DRAG_COEFF = 0.5 * AIR_DENSITY * FRONT_AREA * DRAG_CD
ROLLING_FORCE = CRR * MASS * GRAVITY

TARGET_SOC = 0.5
SOC_GAIN_BASE = 60.0                # kW per full SOC error.
SOC_GAIN_DEADBAND = 18.0
SOC_DEADBAND = 0.04
SOC_TARGET_BAND = 0.03

IDLE_POWER_THRESHOLD_KW = 2.0
REGEN_THRESHOLD_KW = -2.0

IDLE_FCS_VERY_LOW = 30.0
IDLE_FCS_LOW = 20.0
IDLE_FCS_NORMAL = 12.0
SOC_IDLE_VERY_LOW = 0.20
SOC_IDLE_LOW = 0.35

FCS_SOH_SOFT_LIMIT = 0.85
FCS_SOH_HARD_LIMIT = 0.70
FCS_CAP_SOFT = 42.0
FCS_CAP_HARD = 30.0

BATTERY_SOH_SOFT_LIMIT = 0.90
BATTERY_FLOOR_POWER_KW = 5.0
BATTERY_FLOOR_FRACTION = 0.85

EDU_SOH_SOFT_LIMIT = 0.85
EDU_CAP_SOFT = 45.0

SPEED_CAP_HARD = 70.0 / 3.6

# Selector lambda values.
LAMBDA_SIGNAL = 0.80
LAMBDA_SPEED = 0.75
LAMBDA_CATCH = 0.75
LAMBDA_FOLLOW_CLOSE = 0.70
LAMBDA_FOLLOW_WIDE = 0.55
LAMBDA_FOLLOW_STABLE = 0.45

ENERGY_SOC_LOW = 0.22
ENERGY_SOC_HIGH = 0.78
LAMBDA_ENERGY_SOC = 0.62
LAMBDA_ENERGY_IDLE = 0.40


# -----------------------------------------------------------------------------
# Input validation helpers
# -----------------------------------------------------------------------------

def _as_batch_observation(observation):
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim == 1:
        if x.size != OBS_DIM:
            raise ValueError("1D observation must have shape (15,)")
        x = x.reshape(1, OBS_DIM)

    if x.ndim != 2 or x.shape[1] != OBS_DIM:
        raise ValueError("observation must have shape (15,) or (batch, 15)")

    if not np.all(np.isfinite(x)):
        raise ValueError("observation must contain only finite values")

    return x


def _as_single_observation(observation):
    x = np.asarray(observation, dtype=np.float32)

    if x.ndim != 1 or x.size != OBS_DIM:
        raise ValueError("selector observation must have shape (15,)")

    if not np.all(np.isfinite(x)):
        raise ValueError("selector observation must contain only finite values")

    return x


# -----------------------------------------------------------------------------
# Longitudinal following helpers
# -----------------------------------------------------------------------------

def _safe_gap(v):
    v = np.maximum(v, 0.0)
    return 2.0 + 0.5 * v + 0.0625 * v * v


def _max_gap(v):
    v = np.maximum(v, 0.0)
    return 10.0 + v + 0.0825 * v * v


def _initial_follow_accel(x):
    """
    Simple car-following acceleration proposal from raw observations.

    This is a rule proposal, not the actual ego action.  It uses:
      - current gap,
      - ego/lead speed,
      - ego/lead acceleration,
      - a conservative one-step earlier-arrival margin by increasing the
        safe-gap target by approximately v*1 s.

    If the gap sentinel is observed, driving prior is marked inactive in the
    rule output.
    """
    v = np.maximum(x[:, 5], 0.0)
    gap = x[:, 8]
    a_lead = np.clip(x[:, 6], -LEAD_ACCEL_CLIP, LEAD_ACCEL_CLIP)
    v_lead = np.maximum(x[:, 7], 0.0)

    valid_gap = (gap > GAP_MIN_VALID) & (gap < GAP_NO_TARGET)

    # Conservative safe gap: ego may travel one extra step farther.
    safe_eval = _safe_gap(v) + UNCERTAIN_TIME * v + SAFE_EXTRA
    max_eval = _max_gap(v)
    high_gap = np.maximum(safe_eval, max_eval)

    target_gap = np.clip(gap, safe_eval, high_gap)
    gap_err = gap - target_gap
    rel_v = v_lead - v

    raw = a_lead + K_GAP * gap_err + K_REL * rel_v

    # Extra closing protection when already below conservative safe gap.
    gap_short = np.maximum(0.0, safe_eval - gap)
    raw = np.minimum(raw, a_lead - CLOSE_GAIN * gap_short)

    raw = np.where(valid_gap, raw, 0.0)
    raw = np.clip(raw, ACCEL_MIN, ACCEL_MAX).astype(np.float32)

    return raw, valid_gap


# -----------------------------------------------------------------------------
# Speed-limit region helpers
# -----------------------------------------------------------------------------

def _speed_transition_mask(x):
    """
    True when a lower speed limit is being approached within the finite
    horizon.  A sentinel distance >= 29000 is treated as no next region.
    """
    v = np.maximum(x[:, 5], 0.0)
    cur_limit = x[:, 12]
    next_limit = x[:, 13]
    dist_next = x[:, 14]

    watch = np.maximum(SPEED_WATCH_MIN, v * (HORIZON + UNCERTAIN_TIME))

    valid_region = (dist_next < NO_LIMIT_MIN) & (next_limit > 0.0)
    lower_limit = next_limit < cur_limit - SPEED_COMPARISON_EPS
    within_watch = dist_next <= watch

    speed_too_high = v > next_limit + SPEED_MARGIN
    at_entry = dist_next <= SPEED_ENTRY_LINE_MARGIN

    mask = valid_region & lower_limit & within_watch & (speed_too_high | at_entry)
    return mask


def _speed_transition_cap(x):
    """
    Acceleration cap for lower-limit entry shaping.

    Conservative margin: if ego can travel one extra step farther, use
    d_eff = d - v*1 s for the braking-distance estimate.
    """
    mask = _speed_transition_mask(x)
    batch = x.shape[0]

    if not np.any(mask):
        return np.full(batch, ACCEL_MAX, dtype=np.float32)

    v = np.maximum(x[:, 5], 0.0)
    next_limit = x[:, 13]
    dist_next = x[:, 14]

    d_eff = np.maximum(dist_next - UNCERTAIN_TIME * v, 1e-3)
    v_target = np.maximum(0.0, next_limit - SPEED_ENTRY_MARGIN)

    a_needed = (v_target * v_target - v * v) / (2.0 * d_eff)
    cap = np.where(mask, a_needed, ACCEL_MAX)
    cap = np.clip(cap, ACCEL_MIN, ACCEL_MAX).astype(np.float32)

    return cap


# -----------------------------------------------------------------------------
# Traffic-signal prediction helpers
# -----------------------------------------------------------------------------

def _signal_state_at_scalar(t, state, remaining):
    """
    Predict the next-light state at time t using only the observed current
    phase and remaining time.  No unbounded future planning is used; callers
    pass only horizon-limited times.
    """
    state = int(round(float(state)))

    if state not in (SIGNAL_RED, SIGNAL_GREEN, SIGNAL_YELLOW):
        return SIGNAL_RED

    t = max(0.0, float(t))
    remaining = float(remaining)

    # Conservative: if remaining is 0, assume current phase has just ended.
    if remaining <= 0.0:
        state = _SIGNAL_NEXT[state]
        remaining = _SIGNAL_DURATION[state]

    for _ in range(30):
        if remaining > 1e-6:
            if t < remaining:
                return state
            t -= remaining

        state = _SIGNAL_NEXT[state]
        remaining = _SIGNAL_DURATION[state]

    return state


def _crossing_time_scalar(d, v, a):
    """
    Time for ego to reach a stop line under constant requested acceleration a.
    No reverse motion is modeled explicitly here; the environment later clips
    requests and execution.
    """
    d = float(d)
    v = max(0.0, float(v))
    a = float(a)

    if d <= 0.0:
        return 0.0

    if v <= STOP_EPS and a <= STOP_ACCEL_EPS:
        return np.inf

    # If the vehicle can stop before the stop line, allow a one-step
    # conservative extra travel margin before declaring safe no-crossing.
    if a < -STOP_ACCEL_EPS:
        if v > STOP_EPS:
            stop_distance = (v * v) / (2.0 * (-a))
        else:
            stop_distance = np.inf

        if stop_distance < d - STOP_MARGIN:
            return np.inf

    disc = v * v + 2.0 * a * d
    if disc < 0.0:
        return np.inf

    if abs(a) <= STOP_ACCEL_EPS:
        if v <= STOP_EPS:
            return np.inf
        return d / v

    t = (-v + np.sqrt(max(0.0, disc))) / a

    if t <= 0.0:
        return np.inf

    return t


def _apply_signal_constraint(x, a_target):
    """
    Adjust a proposed acceleration so that candidate crossings predicted
    within the finite horizon do not occur during yellow/red.

    Conservative earlier-arrival margin:
      t_cons = max(0, t_nominal - 1 s)

    Conservative stopping distance:
      d_stop_eff = d - stop_margin - v*1 s
    """
    a_arr = np.asarray(a_target, dtype=np.float32)
    orig_shape = a_arr.shape
    flat = a_arr.reshape(-1).copy()
    batch = x.shape[0]

    if flat.size != batch:
        raise ValueError("acceleration batch size mismatch")

    for i in range(batch):
        d_stop = float(x[i, 10])

        if d_stop >= NO_SIGNAL_MIN:
            continue

        state = int(round(float(x[i, 9])))
        remaining = float(x[i, 11])
        v = max(0.0, float(x[i, 5]))
        a = flat[i]

        # At/very near stop line.
        if d_stop <= STOP_LINE_MARGIN:
            non_green = (state != SIGNAL_GREEN) or (remaining <= 0.0)
            if non_green:
                if v > STOP_EPS:
                    flat[i] = min(flat[i], ACCEL_MIN)
                else:
                    # Keep negative requests as-is; environment handles
                    # no-reverse clipping.  Positive requests are suppressed.
                    flat[i] = min(flat[i], 0.0)
            continue

        if v <= STOP_EPS and a <= STOP_ACCEL_EPS:
            continue

        t_nom = _crossing_time_scalar(d_stop, v, a)
        if not np.isfinite(t_nom):
            continue

        t_cons = max(0.0, t_nom - UNCERTAIN_TIME)
        if t_cons > HORIZON + 1e-6:
            continue

        predicted_state = _signal_state_at_scalar(t_cons, state, remaining)

        if predicted_state == SIGNAL_GREEN:
            continue

        # Yellow/red crossing prohibited: request a stopping or hold action.
        effective_stop_distance = max(
            d_stop - STOP_MARGIN - UNCERTAIN_TIME * v,
            1e-3,
        )

        if v > STOP_EPS:
            a_stop = -(v * v) / (2.0 * effective_stop_distance)
        else:
            a_stop = 0.0

        flat[i] = min(flat[i], a_stop)
        flat[i] = float(np.clip(flat[i], ACCEL_MIN, ACCEL_MAX))

    return flat.reshape(orig_shape)


def _signal_brake_mask(x):
    """
    Rule-only estimate for energy management: braking is expected when the
    currently observed signal context is non-green within the finite horizon,
    or when coasting at current speed would arrive during a non-green phase.

    This uses observed traffic-light state only, not the driving agent's new
    action.
    """
    batch = x.shape[0]
    out = np.zeros(batch, dtype=bool)

    for i in range(batch):
        d_stop = float(x[i, 10])
        if d_stop >= NO_SIGNAL_MIN:
            continue

        state = int(round(float(x[i, 9])))
        remaining = float(x[i, 11])
        v = max(0.0, float(x[i, 5]))

        watch = max(SIGNAL_WATCH_MIN, v * (HORIZON + UNCERTAIN_TIME))

        if d_stop <= STOP_LINE_MARGIN:
            if state != SIGNAL_GREEN or remaining <= 0.0:
                out[i] = True
            continue

        if d_stop <= watch:
            if state != SIGNAL_GREEN:
                out[i] = True
            else:
                # Coasting-only green-ending check.
                t_nom = _crossing_time_scalar(d_stop, v, 0.0)
                if np.isfinite(t_nom):
                    t_cons = max(0.0, t_nom - UNCERTAIN_TIME)
                    if t_cons <= HORIZON + 1e-6:
                        pred = _signal_state_at_scalar(t_cons, state, remaining)
                        if pred != SIGNAL_GREEN:
                            out[i] = True

    return out


# -----------------------------------------------------------------------------
# Acceleration-to-latent mapping
# -----------------------------------------------------------------------------

def _u_from_accel(a):
    a = np.clip(a, ACCEL_MIN, ACCEL_MAX)

    # Physical mapping:
    #   a < 0: a = 1.5 u  => u = a/1.5, negative.
    #   a >=0: a = 0.7 u  => u = a/0.7, positive.
    u = np.where(a < 0.0, a / BRAKE_SCALE, a / DRIVE_SCALE)
    return np.clip(u, -U_LIMIT, U_LIMIT).astype(np.float32)


def _latent_mean_from_u(u):
    u = np.clip(u, -U_LIMIT, U_LIMIT)
    return np.arctanh(u).astype(np.float32)


# -----------------------------------------------------------------------------
# Driving prior helper
# -----------------------------------------------------------------------------

def _driving_accel_for_mode(x, mode):
    """
    Returns proposed requested acceleration [batch] and valid-gap mask [batch].
    The proposal applies speed, speed-region, and signal constraints after the
    primary following proposal.
    """
    a_follow, valid_gap = _initial_follow_accel(x)
    a = a_follow.copy()

    if mode == "catch_up":
        if np.any(valid_gap):
            gap = x[:, 8]
            v = np.maximum(x[:, 5], 0.0)
            max_gap = _max_gap(v)
            catch_mask = valid_gap & (gap > max_gap + GAP_CATCH_MARGIN)
            a = np.where(catch_mask, np.maximum(a, CATCHUP_MIN_ACCEL), a)

    # Posted current speed limit shaping.
    current_limit_cap = np.clip(x[:, 12] - x[:, 5], ACCEL_MIN, ACCEL_MAX)
    a = np.minimum(a, current_limit_cap)

    # Approaching a lower posted limit.
    transition_cap = _speed_transition_cap(x)
    a = np.minimum(a, transition_cap)

    # Green-only crossing safety.
    a = _apply_signal_constraint(x, a)

    a = np.asarray(a, dtype=np.float32)
    return a, valid_gap


def _driving_std(x, mode, valid_gap):
    batch = x.shape[0]
    v = np.maximum(x[:, 5], 0.0)
    gap = x[:, 8]

    if mode == "signal":
        sig_state = x[:, 9]
        sig_dist = x[:, 10]
        watch = np.maximum(SIGNAL_WATCH_MIN, v * (HORIZON + UNCERTAIN_TIME))
        signal_watch = (sig_dist < NO_SIGNAL_MIN) & (sig_dist <= watch) & (sig_state != SIGNAL_GREEN)
        std = np.where(signal_watch, SIGNAL_STD_STRONG, SIGNAL_STD_MEDIUM)

    elif mode == "speed":
        std = np.full(batch, SPEED_STD, dtype=np.float32)

    elif mode == "catch_up":
        std = np.full(batch, CATCH_STD, dtype=np.float32)

    elif mode in ("follow", "follow_mild"):
        safe_eval = _safe_gap(v) + UNCERTAIN_TIME * v + SAFE_EXTRA
        max_gap = _max_gap(v)

        valid_flat = valid_gap.ravel()
        gap_flat = gap.ravel()

        close = valid_flat & (gap_flat < safe_eval + FOLLOW_CLOSE_MARGIN)
        wide = valid_flat & (gap_flat > max_gap + FOLLOW_WIDE_MARGIN)

        if mode == "follow":
            std = np.where(close, FOLLOW_STD_CLOSE,
                    np.where(wide, FOLLOW_STD_WIDE, FOLLOW_STD_STABLE))
        else:
            std = np.where(close, FOLLOW_MILD_STD_CLOSE,
                    np.where(wide, FOLLOW_MILD_STD_WIDE, FOLLOW_MILD_STD))

    else:
        std = np.zeros(batch, dtype=np.float32)

    std = np.clip(std, MIN_PRIOR_STD, MAX_PRIOR_STD).astype(np.float32)
    return std.reshape(batch, 1)


# -----------------------------------------------------------------------------
# Approximate mechanical demand estimate and energy prior helper
# -----------------------------------------------------------------------------

def _mech_power_kw(v, a):
    """
    Approximate mechanical EDU/vehicle traction power in kW.

    Important limitations:
      - Uses measured current ego acceleration, not the newly requested
        driving action.
      - No observed torque, EDU map region, battery power, hydrogen flow, or
        actual FCS output is available.
      - EDU efficiency/loss map, valid EDU torque-speed region, and powertrain
        feasibility are not reconstructed by this rule.
      - Negative values represent estimated regen/braking demand, but the true
        EDU regeneration limit is environment-specific.
    """
    v = np.maximum(v, 0.0)
    a = np.clip(a, ACCEL_MIN, ACCEL_MAX)

    force = ROLLING_FORCE + DRAG_COEFF * v * v + MASS_ROT * a
    return (force * v) / 1000.0


def _u_from_power(p_kw):
    p_kw = np.clip(p_kw, 0.0, MAX_FCS_KW)
    u = (p_kw / POWER_SCALE) - 1.0
    return np.clip(u, -U_LIMIT, U_LIMIT).astype(np.float32)


def _energy_prior(x, strength="normal"):
    """
    Gaussian prior for requested FCS output power.

    Approximation strategy:
      - Estimate short-term traction power from measured ego speed/accel.
      - If a speed-region or signal braking context is observed, bias the
        estimated demand toward zero/regeneration for the FCS request.
      - Use SOC error to choose charging/assist power.
      - Apply conservative degradation-aware caps for FCS/battery/EDU SOH.
      - This does not attempt to reproduce the reward terms exactly.
    """
    batch = x.shape[0]

    v = np.maximum(x[:, 5], 0.0)
    a = np.clip(x[:, 4], ACCEL_MIN, ACCEL_MAX)

    soc = np.clip(x[:, 0], 0.0, 1.0)
    soh_batt = np.clip(x[:, 1], 0.0, 1.0)
    soh_fcs = np.clip(x[:, 2], 0.0, 1.0)
    soh_edu = np.clip(x[:, 3], 0.0, 1.0)

    current_limit = np.minimum(np.maximum(x[:, 12], 0.0), SPEED_CAP_HARD)

    # Approximate horizon demand by averaging current and one-step predicted
    # mechanical demand.  This is only an estimate.
    p0 = _mech_power_kw(v, a)
    v1 = np.maximum(0.0, np.minimum(v + a * DT, current_limit))
    p1 = _mech_power_kw(v1, a)
    p_est = 0.5 * (p0 + p1)

    # Observed driving-constraint context used only as an approximation.
    speed_brake = _speed_transition_mask(x)
    signal_brake = _signal_brake_mask(x)
    brake_expected = speed_brake | signal_brake

    # If braking is likely, prefer not to request FCS power for positive
    # traction demand; battery/regen handling is environment-determined.
    p_for_fcs = np.where(brake_expected, np.minimum(p_est, 0.0), p_est)

    soc_err = TARGET_SOC - soc
    soc_gain = np.where(
        np.abs(soc_err) > SOC_DEADBAND,
        SOC_GAIN_BASE,
        SOC_GAIN_DEADBAND,
    )
    soc_term = soc_gain * soc_err

    regen_mask = p_for_fcs < REGEN_THRESHOLD_KW
    demand_pos = np.maximum(0.0, p_for_fcs)

    # FCS request target in kW.
    p_target = np.where(regen_mask, 0.0, demand_pos + soc_term)

    # Low-demand idle handling: allow SOC recovery, but avoid unnecessary
    # hydrogen use/startup-heavy power.
    low_demand_mask = demand_pos < IDLE_POWER_THRESHOLD_KW
    p_target = np.where(
        low_demand_mask,
        np.maximum(0.0, soc_term),
        p_target,
    )
    p_target = np.maximum(0.0, p_target)

    # Degradation-aware FCS power caps.
    max_fcs = np.full(batch, MAX_FCS_KW, dtype=np.float32)

    max_fcs = np.where(
        soh_fcs < FCS_SOH_SOFT_LIMIT,
        np.minimum(max_fcs, FCS_CAP_SOFT),
        max_fcs,
    )
    max_fcs = np.where(
        soh_fcs < FCS_SOH_HARD_LIMIT,
        np.minimum(max_fcs, FCS_CAP_HARD),
        max_fcs,
    )

    # If EDU SOH is reduced, avoid encouraging unnecessarily high FCS output.
    # The actual EDU demand is still determined by driving dynamics and the
    # environment's EDU map/feasibility.
    max_fcs = np.where(
        soh_edu < EDU_SOH_SOFT_LIMIT,
        np.minimum(max_fcs, EDU_CAP_SOFT),
        max_fcs,
    )

    # If battery SOH is reduced, avoid unnecessarily low FCS power during
    # positive traction demand, which would increase battery discharge current.
    battery_mask = (
        (demand_pos > BATTERY_FLOOR_POWER_KW)
        & (soh_batt < BATTERY_SOH_SOFT_LIMIT)
    )
    battery_floor = np.where(
        battery_mask,
        np.minimum(MAX_FCS_KW, BATTERY_FLOOR_FRACTION * demand_pos),
        0.0,
    )
    p_target = np.maximum(p_target, battery_floor)

    # Idle cap depending on SOC.
    idle_max = np.where(
        soc < SOC_IDLE_VERY_LOW,
        IDLE_FCS_VERY_LOW,
        np.where(soc < SOC_IDLE_LOW, IDLE_FCS_LOW, IDLE_FCS_NORMAL),
    )
    p_target = np.where(low_demand_mask, np.minimum(p_target, idle_max), p_target)

    # Small SOC-target band at low demand: save hydrogen by keeping FCS near
    # zero and letting the battery maintain the target band.
    target_band = np.abs(soc - TARGET_SOC) < SOC_TARGET_BAND
    p_target = np.where(
        low_demand_mask & target_band,
        0.0,
        p_target,
    )

    p_target = np.clip(p_target, 0.0, max_fcs).astype(np.float32)

    u = _u_from_power(p_target)
    mean = _latent_mean_from_u(u)

    base_std = ENERGY_STD_STRONG if strength == "strong" else ENERGY_STD_NORMAL
    energy_std = np.full(batch, base_std, dtype=np.float32)

    extreme = (p_target < ENERGY_EXTREME_LOW_KW) | (p_target > ENERGY_EXTREME_HIGH_KW)
    energy_std = np.where(extreme, 0.5 * energy_std, energy_std)
    energy_std = np.clip(energy_std, MIN_PRIOR_STD, MAX_PRIOR_STD).astype(np.float32)

    return (
        mean.reshape(batch, 1),
        energy_std.reshape(batch, 1),
        np.ones((batch, 1), dtype=bool),
    )


# -----------------------------------------------------------------------------
# Rule output assembly
# -----------------------------------------------------------------------------

def _rule_output_from_obs(observation, driving_mode, energy_strength="normal"):
    x = _as_batch_observation(observation)
    batch = x.shape[0]

    if driving_mode is None:
        drive_mean = np.zeros((batch, 1), dtype=np.float32)
        drive_std = np.zeros((batch, 1), dtype=np.float32)
        drive_active = np.zeros((batch, 1), dtype=bool)
    else:
        a_target, valid_gap = _driving_accel_for_mode(x, driving_mode)

        drive_mean = _latent_mean_from_u(_u_from_accel(a_target))
        drive_std = _driving_std(x, driving_mode, valid_gap)

        # For driving modes that require a visible lead-following target,
        # deactivate the driving prior if the gap sentinel is observed.
        if driving_mode in ("follow", "follow_mild", "catch_up"):
            drive_active = valid_gap.reshape(batch, 1)
            drive_mean = np.where(drive_active, drive_mean, np.float32(0.0))
            drive_std = np.where(drive_active, drive_std, np.float32(0.0))
        else:
            drive_active = np.ones((batch, 1), dtype=bool)

        drive_mean = np.asarray(drive_mean, dtype=np.float32).reshape(batch, 1)
        drive_std = np.asarray(drive_std, dtype=np.float32).reshape(batch, 1)
        drive_active = np.asarray(drive_active, dtype=bool).reshape(batch, 1)

    energy_mean, energy_std, energy_active = _energy_prior(x, strength=energy_strength)

    return {
        "driving": {
            "mean": drive_mean,
            "std": drive_std,
            "active": drive_active,
        },
        "energy": {
            "mean": energy_mean,
            "std": energy_std,
            "active": energy_active,
        },
    }


# -----------------------------------------------------------------------------
# Individual rules
# -----------------------------------------------------------------------------

def rule_follow_gap(observation):
    """Primary following rule: keep gap between safe and maximum range."""
    return _rule_output_from_obs(observation, driving_mode="follow", energy_strength="normal")


def rule_catch_up_large_gap(observation):
    """Closing rule for excessive gaps while still obeying limits/signals."""
    return _rule_output_from_obs(observation, driving_mode="catch_up", energy_strength="normal")


def rule_signal_approach(observation):
    """Signal-approach rule with green-only crossing priority."""
    return _rule_output_from_obs(observation, driving_mode="signal", energy_strength="strong")


def rule_speed_limit_transition(observation):
    """Rule for approaching a lower posted-speed region."""
    return _rule_output_from_obs(observation, driving_mode="speed", energy_strength="normal")


def rule_energy_management(observation):
    """
    Energy-priority rule.  Driving prior is still mild car-following when a
    valid lead gap exists; the energy prior is stronger.
    """
    return _rule_output_from_obs(observation, driving_mode="follow_mild", energy_strength="strong")


RULES = {
    "follow_gap": rule_follow_gap,
    "catch_up_large_gap": rule_catch_up_large_gap,
    "signal_approach": rule_signal_approach,
    "speed_limit_transition": rule_speed_limit_transition,
    "energy_management": rule_energy_management,
}


# -----------------------------------------------------------------------------
# Selector
# -----------------------------------------------------------------------------

def select_rule(observation):
    """
    Deterministic selector evaluated once at each runtime selection interval.
    The selector itself does not maintain a counter or hidden mutable state.
    """
    x1 = _as_single_observation(observation)
    x = x1.reshape(1, OBS_DIM)

    v = float(x[0, 5])
    gap = float(x[0, 8])
    sig_state = int(round(float(x[0, 9])))
    sig_dist = float(x[0, 10])
    sig_remaining = float(x[0, 11])
    cur_limit = float(x[0, 12])
    next_limit = float(x[0, 13])
    next_limit_dist = float(x[0, 14])
    soc = float(x[0, 0])

    valid_gap = bool(gap > GAP_MIN_VALID and gap < GAP_NO_TARGET)

    # Approximate whether a proposed follow-with-speed-cap action would need
    # signal braking within the finite horizon.
    a_follow, _ = _initial_follow_accel(x)
    current_limit_cap = np.clip(x[:, 12] - x[:, 5], ACCEL_MIN, ACCEL_MAX)
    transition_cap = _speed_transition_cap(x)
    a_pre = np.minimum(np.minimum(a_follow, current_limit_cap), transition_cap)
    a_post = _apply_signal_constraint(x, a_pre)

    signal_action_risk = bool(np.any(np.abs(a_post - a_pre) > SIGNAL_RISK_TOL))

    signal_valid = bool(sig_dist < NO_SIGNAL_MIN)
    signal_watch = max(SIGNAL_WATCH_MIN, v * (HORIZON + UNCERTAIN_TIME))

    signal_non_green_watch = bool(
        signal_valid
        and (sig_state != SIGNAL_GREEN)
        and (sig_dist <= signal_watch)
    )

    # At-line non-green or zero remaining time is already handled conservatively
    # by the signal constraint; include it as a selector trigger as well.
    signal_at_line_risk = bool(
        signal_valid
        and sig_dist <= STOP_LINE_MARGIN
        and (sig_state != SIGNAL_GREEN or sig_remaining <= 0.0)
    )

    signal_stop_required = bool(signal_action_risk or signal_non_green_watch or signal_at_line_risk)

    speed_risk = bool(_speed_transition_mask(x)[0])

    max_gap_val = float(_max_gap(np.array([v], dtype=np.float32))[0])
    catch_risk = bool(
        valid_gap
        and gap > max_gap_val + GAP_CATCH_MARGIN
    )

    if signal_stop_required:
        return {"rule_name": "signal_approach", "lambda": float(LAMBDA_SIGNAL)}

    if speed_risk:
        return {"rule_name": "speed_limit_transition", "lambda": float(LAMBDA_SPEED)}

    if catch_risk:
        return {"rule_name": "catch_up_large_gap", "lambda": float(LAMBDA_CATCH)}

    if valid_gap:
        safe_eval_val = float(
            (_safe_gap(np.array([v], dtype=np.float32))
             + UNCERTAIN_TIME * max(v, 0.0)
             + SAFE_EXTRA)[0]
        )

        if gap < safe_eval_val + FOLLOW_CLOSE_MARGIN:
            follow_lambda = float(LAMBDA_FOLLOW_CLOSE)
        elif gap > max_gap_val + GAP_CATCH_MARGIN * 0.5:
            follow_lambda = float(LAMBDA_FOLLOW_WIDE)
        else:
            follow_lambda = float(LAMBDA_FOLLOW_STABLE)

        if soc < ENERGY_SOC_LOW or soc > ENERGY_SOC_HIGH:
            return {"rule_name": "energy_management", "lambda": float(LAMBDA_ENERGY_SOC)}

        return {"rule_name": "follow_gap", "lambda": follow_lambda}

    # No valid following target and no immediate constraint: still provide an
    # energy-management prior; driving prior will be inactive in the rule.
    return {"rule_name": "energy_management", "lambda": float(LAMBDA_ENERGY_IDLE)}
```

### Rule menu and task-stage table

| Rule name | Brief description | Task stage |
|---|---|---|
| `follow_gap` | Primary car-following prior. Uses gap, ego/lead speed and acceleration, and a conservative one-step earlier-arrival margin to keep the gap inside the safe–maximum range while respecting current speed limits. | Following |
| `catch_up_large_gap` | Gap-closing prior. Activates when the gap is excessively large relative to the speed-dependent maximum gap. It increases the closing tendency but still caps actions using speed-limit, lower-speed-region, and green-only signal constraints. | Following / catch-up |
| `signal_approach` | Traffic-signal approach prior. Predicts the next-light state at horizon-limited crossing times, treats yellow and red as non-green, applies conservative earlier-arrival margins, and suppresses acceleration that would cross during non-green phases. | Approaching a signalized intersection |
| `speed_limit_transition` | Speed-region transition prior. Brakes or limits acceleration when a lower posted speed region is approached within the finite horizon. It uses the one-step uncertainty margin for entry-speed prediction. | Speed-limit region transition |
| `energy_management` | Energy-priority prior with mild driving support. Estimates mechanical demand from measured ego acceleration/speed, biases power toward zero/regeneration when signal or lower-speed braking is observed, uses SOC error for charging/assist selection, and applies SOH-aware FCS/battery/EDU caps. Driving prior remains mild when a valid lead gap exists. | Energy management / SOC regulation / degradation management |