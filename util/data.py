from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass
class ParameterQuery:
    value: float
    clipped: bool


class GridTable:
    """Bilinear interpolation over an author-provided SOC-temperature table."""

    def __init__(self, path: Path):
        rows = np.loadtxt(path, delimiter="\t", dtype=np.float64)
        soc = np.unique(rows[:, 0])
        temp = np.unique(rows[:, 1])
        values = rows[:, 2].reshape(soc.size, temp.size)
        if rows.shape[0] != soc.size * temp.size:
            raise ValueError(f"Irregular lookup table layout in {path}")
        self.soc_axis = soc
        self.temp_axis = temp
        self.values = values
        self._interp = RegularGridInterpolator(
            (soc, temp), values, method="linear", bounds_error=True
        )

    def query(self, soc: float, temp_c: float) -> ParameterQuery:
        s = float(np.clip(soc, self.soc_axis[0], self.soc_axis[-1]))
        t = float(np.clip(temp_c, self.temp_axis[0], self.temp_axis[-1]))
        value = float(self._interp([[s, t]])[0])
        return ParameterQuery(value=value, clipped=(s != soc or t != temp_c))


class BatteryTables:
    PARAM_FILES = {
        "rs": "rs_{direction}_ohm.txt",
        "r1": "r1_{direction}_ohm.txt",
        "r2": "r2_{direction}_ohm.txt",
        "c1": "c1_{direction}_f.txt",
        "c2": "c2_{direction}_f.txt",
    }

    def __init__(self, root: Path):
        self.root = Path(root)
        self.ocv = np.loadtxt(self.root / "ocv_c20_mean_v.txt", delimiter="\t")
        self.entropy = np.loadtxt(
            self.root / "entropic_coefficient_mV_per_K.txt", delimiter="\t"
        )
        self.tables = {
            (name, direction): GridTable(
                self.root / filename.format(direction=direction)
            )
            for name, filename in self.PARAM_FILES.items()
            for direction in ("charge", "discharge")
        }
        self.core_capacity_j_per_k = self._scalar("cc_j_per_k.txt")
        self.surface_capacity_j_per_k = self._scalar("cs_j_per_k.txt")
        self.core_surface_resistance_k_per_w = self._scalar("rc_k_per_w.txt")
        self.paper_surface_coolant_resistance_k_per_w = self._scalar(
            "ru_lin2014_k_per_w.txt"
        )

    def _scalar(self, name: str) -> float:
        return float(np.loadtxt(self.root / name, dtype=np.float64))

    def ocv_v(self, soc: float) -> float:
        return float(np.interp(soc, self.ocv[:, 0], self.ocv[:, 1]))

    def entropy_v_per_k(self, soc: float) -> float:
        return float(np.interp(soc, self.entropy[:, 0], self.entropy[:, 1]) / 1000.0)

    def parameters(self, soc: float, temp_c: float, direction: str):
        values = {}
        clipped = False
        for name in self.PARAM_FILES:
            query = self.tables[(name, direction)].query(soc, temp_c)
            values[name] = query.value
            clipped |= query.clipped
        return values, clipped


class ProjectData:
    def __init__(self, project_root: Path):
        root = Path(project_root)
        self.root = root
        self.battery = BatteryTables(root / "dataset" / "battery")
        self.fcs = np.loadtxt(root / "dataset" / "fcs" / "fcs_parameters.txt")
        self.motor = self._load_motor(root / "dataset" / "motor")

    @staticmethod
    def _load_motor(root: Path):
        def load(name):
            return np.loadtxt(root / f"{name}.txt", delimiter="\t", dtype=np.float64)

        data = {
            "speed_rpm": load("motor_speed_axis_rpm").reshape(-1),
            "speed_radps": load("motor_speed_axis_radps").reshape(-1),
            "torque_nm": load("motor_torque_axis_nm").reshape(-1),
            "electric_power_w": load("electric_power_lookup_w"),
            "mechanical_power_w": load("mechanical_power_lookup_w"),
            "loss_w": load("total_power_loss_lookup_w_valid_region"),
            "operating_mask": load("operating_mask").astype(bool),
            "efficiency_mask": load("efficiency_defined_mask").astype(bool),
            "positive_torque_limit_nm": load("positive_torque_limit_motor_nm").reshape(-1),
            "negative_torque_limit_nm": load("negative_torque_limit_motor_nm").reshape(-1),
            "max_motor_power_w": float(load("max_motor_power_w").reshape(-1)[0]),
            "max_generator_power_w": float(load("max_generator_power_w").reshape(-1)[0]),
            "max_speed_rpm": float(load("max_speed_motor_rpm").reshape(-1)[0]),
            "gear_ratio": float(load("gear_ratio").reshape(-1)[0]),
            "loss_component_breakdown_available": bool(
                load("loss_component_breakdown_available").reshape(-1)[0]
            ),
        }
        if data["electric_power_w"].shape != (data["speed_rpm"].size, data["torque_nm"].size):
            raise ValueError("EPA EDU electric-power matrix does not match its axes")
        return data
