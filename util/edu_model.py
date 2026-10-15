from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import LinearNDInterpolator


@dataclass
class EDUPoint:
    feasible: bool
    speed_rpm: float
    torque_nm: float
    mechanical_power_w: float
    electrical_power_w: float
    loss_power_w: float
    efficiency: float | None
    reason: str | None = None


class EDUModel:
    """EPA Generic IPM lookup; its physical boundary is the complete EDU."""

    def __init__(self, table_data: dict):
        self.data = table_data
        self.speed_axis = table_data["speed_rpm"]
        self.torque_axis = table_data["torque_nm"]
        speed_grid, torque_grid = np.meshgrid(
            self.speed_axis, self.torque_axis, indexing="ij"
        )
        valid = (
            np.asarray(table_data["operating_mask"], dtype=bool)
            & np.isfinite(table_data["electric_power_w"])
            & np.isfinite(table_data["loss_w"])
        )
        points = np.column_stack((speed_grid[valid], torque_grid[valid]))
        self._electric = LinearNDInterpolator(
            points, np.asarray(table_data["electric_power_w"])[valid], fill_value=np.nan
        )
        self._loss = LinearNDInterpolator(
            points, np.asarray(table_data["loss_w"])[valid], fill_value=np.nan
        )
        self.max_motor_power_w = table_data["max_motor_power_w"]
        self.max_generator_power_w = table_data["max_generator_power_w"]
        self.max_speed_rpm = table_data["max_speed_rpm"]
        self.gear_ratio = table_data["gear_ratio"]

    def torque_limits(self, speed_rpm: float) -> tuple[float, float]:
        s = abs(float(speed_rpm))
        positive = float(np.interp(s, self.speed_axis, self.data["positive_torque_limit_nm"]))
        negative = float(np.interp(s, self.speed_axis, self.data["negative_torque_limit_nm"]))
        return (
            min(positive, float(self.torque_axis[-1])),
            max(negative, float(self.torque_axis[0])),
        )

    def lookup(self, speed_rpm: float, torque_nm: float) -> EDUPoint:
        speed = abs(float(speed_rpm))
        torque = float(torque_nm)
        omega = speed * (2.0 * np.pi / 60.0)
        p_mech = omega * torque
        empty = lambda reason: EDUPoint(
            False, speed, torque, p_mech, float("nan"), float("nan"), None, reason
        )
        if speed > self.max_speed_rpm + 1e-8:
            return empty("motor_speed_limit")
        if speed < self.speed_axis[0] or speed > self.speed_axis[-1]:
            return empty("speed_outside_lookup")
        if torque < self.torque_axis[0] or torque > self.torque_axis[-1]:
            return empty("torque_outside_lookup")
        positive_limit, negative_limit = self.torque_limits(speed)
        if torque > positive_limit + 1e-6 or torque < negative_limit - 1e-6:
            return empty("torque_speed_envelope")
        power_limit = self.max_motor_power_w if p_mech >= 0 else self.max_generator_power_w
        if abs(p_mech) > power_limit + 1e-6:
            return empty("mechanical_power_limit")
        p_elec = float(self._electric(speed, torque))
        p_loss = float(self._loss(speed, torque))
        if not np.isfinite(p_elec) or not np.isfinite(p_loss):
            return empty("outside_valid_lookup_hull")
        p_loss = max(0.0, p_loss)
        if p_mech > 1e-8 and p_elec > 0:
            eta = p_mech / p_elec
        elif p_mech < -1e-8 and p_elec < 0:
            eta = p_elec / p_mech
        else:
            eta = None
        return EDUPoint(True, speed, torque, p_mech, p_elec, p_loss, eta)

    def reference_point(self, power_w: float, torque_nm: float, speed_rpm: float) -> EDUPoint:
        point = self.lookup(speed_rpm, torque_nm)
        if not point.feasible:
            raise ValueError(f"EDU reference point is infeasible: {point.reason}")
        if not np.isclose(point.mechanical_power_w, power_w, rtol=0.01, atol=100.0):
            raise ValueError(
                "EDU reference point torque-speed product does not match configured "
                f"mechanical power ({point.mechanical_power_w:.1f} vs {power_w:.1f} W)"
            )
        return point
