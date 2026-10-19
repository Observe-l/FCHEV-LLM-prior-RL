from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Demand:
    requested_acceleration_m_s2: float
    executed_acceleration_m_s2: float
    next_speed_m_s: float
    mean_speed_m_s: float
    traction_force_n: float
    wheel_mechanical_power_w: float
    motor_speed_rpm: float
    motor_torque_nm: float
    friction_brake_force_n: float = 0.0


def longitudinal_demand(speed_m_s: float, acceleration_m_s2: float, dt_s: float, cfg: dict) -> Demand:
    v = max(0.0, float(speed_m_s))
    dt = float(dt_s)
    a = float(acceleration_m_s2)
    # Keep the one-step kinematics forward-only; stopping exactly at v=0 is allowed.
    a = max(a, -v / dt) if dt > 0 else a
    # Do not replace a positive command with a zero-speed-cap command here.
    # FCHEVEnv terminates an episode when the requested action would exceed
    # the configured maximum speed, before this plant calculation is called.
    vn = max(0.0, v + a * dt)
    # The quasi-steady power is evaluated at the mean shaft speed over this
    # fixed-step transition; this also gives the correct kinetic-energy change.
    vm = 0.5 * (v + vn)
    sign = 1.0 if vm > 1e-9 else (1.0 if a > 0 else 0.0)
    grade = np.deg2rad(cfg["road_slope_deg"])
    rolling = cfg["rolling_resistance_coefficient"] * cfg["mass_kg"] * cfg["gravity_m_s2"] * np.cos(grade) * sign
    grade_force = cfg["mass_kg"] * cfg["gravity_m_s2"] * np.sin(grade)
    aero = 0.5 * cfg["air_density_kg_m3"] * cfg["drag_coefficient"] * cfg["frontal_area_m2"] * vm * abs(vm)
    inertia = cfg["rotational_mass_coefficient"] * cfg["mass_kg"] * a
    force = inertia + rolling + grade_force + aero
    power = force * vm
    omega_motor = vm / cfg["wheel_radius_m"] * cfg["gear_ratio"]
    speed_rpm = omega_motor * 60.0 / (2.0 * np.pi)
    torque = force * cfg["wheel_radius_m"] / cfg["gear_ratio"]
    return Demand(float(acceleration_m_s2), a, vn, vm, force, power, speed_rpm, torque)
