from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BatteryState:
    soc: float
    vp1_v: float
    vp2_v: float
    core_temperature_c: float
    surface_temperature_c: float
    soh: float


@dataclass
class BatteryStep:
    requested_power_w: float
    terminal_power_w: float
    current_pack_a: float
    current_cell_a: float
    terminal_voltage_pack_v: float
    terminal_voltage_cell_v: float
    heat_generation_w_per_cell: float
    soh_decrement: float
    state: BatteryState
    parameter_table_clipped: bool
    aging_c_rate_clipped: bool


class BatteryModel:
    """Representative-cell 2RC/electrothermal pack model.

    The author-provided R/C lookup values are single-cell parameters. The model
    solves one representative cell and scales only pack voltage/current/power;
    thermal capacitances and resistances therefore remain single-cell values.
    """

    def __init__(self, tables, config: dict):
        self.tables = tables
        self.cfg = config
        self.ns = int(config["series_cells"])
        self.np = int(config["parallel_cells"])
        self.q_nom_ah = float(config["nominal_cell_capacity_ah"])
        self.q_nom_c = self.q_nom_ah * 3600.0
        self.state = BatteryState(
            soc=float(config["initial_soc"]), vp1_v=0.0, vp2_v=0.0,
            core_temperature_c=float(config["initial_core_temperature_c"]),
            surface_temperature_c=float(config["initial_surface_temperature_c"]),
            soh=1.0,
        )
        self.last_parameters = None
        self.last_limits = None

    def reset(self):
        self.state = BatteryState(
            soc=float(self.cfg["initial_soc"]), vp1_v=0.0, vp2_v=0.0,
            core_temperature_c=float(self.cfg["initial_core_temperature_c"]),
            surface_temperature_c=float(self.cfg["initial_surface_temperature_c"]),
            soh=1.0,
        )
        self.last_parameters = None
        self.last_limits = None
        return self.state

    @property
    def capacity_fraction(self):
        # Peng's aging indicator reaches zero at the modeled 20%-capacity EOL.
        return 1.0 - self.cfg["health_capacity_fade_at_eol"] * (1.0 - self.state.soh)

    @property
    def effective_cell_capacity_ah(self):
        return self.q_nom_ah * self.capacity_fraction

    def _electrical_parameters(self, direction: str):
        st = self.state
        average_temp_c = 0.5 * (st.core_temperature_c + st.surface_temperature_c)
        vals, clipped = self.tables.parameters(
            st.soc, average_temp_c, direction
        )
        vals["voc"] = self.tables.ocv_v(st.soc)
        return vals, clipped

    def _cell_current_limits(self, direction: str | None = None):
        st = self.state
        q_eff = self.effective_cell_capacity_ah
        current_dis_max = self.cfg["max_discharge_c_rate"] * q_eff
        current_chg_max = self.cfg["max_charge_c_rate"] * q_eff
        current_dis_max = min(
            current_dis_max,
            max(0.0, (st.soc - self.cfg["minimum_soc"]) * q_eff * 3600.0),
        )
        current_chg_max = min(
            current_chg_max,
            max(0.0, (self.cfg["maximum_soc"] - st.soc) * q_eff * 3600.0),
        )
        # The terminal voltage is V_eq - I*Rs, with discharge current positive.
        p_dis, _ = self._electrical_parameters("discharge")
        p_chg, _ = self._electrical_parameters("charge")
        veq = p_dis["voc"] - st.vp1_v - st.vp2_v
        rs_dis = max(p_dis["rs"], 1e-12)
        rs_chg = max(p_chg["rs"], 1e-12)
        vmin = self.cfg["minimum_cell_voltage_v"]
        vmax = self.cfg["maximum_cell_voltage_v"]
        current_dis_max = min(current_dis_max, max(0.0, (veq - vmin) / rs_dis))
        current_chg_max = min(current_chg_max, max(0.0, (vmax - veq) / rs_chg))
        p_dis_cell = veq * current_dis_max - rs_dis * current_dis_max**2
        p_chg_cell = -veq * current_chg_max - rs_chg * current_chg_max**2
        # p_chg_cell is negative terminal power for I=-current_chg_max.
        return {
            "minimum_power_w": p_chg_cell * self.ns * self.np,
            "maximum_power_w": p_dis_cell * self.ns * self.np,
            "minimum_cell_current_a": -current_chg_max,
            "maximum_cell_current_a": current_dis_max,
        }

    def power_limits(self):
        self.last_limits = self._cell_current_limits()
        return dict(self.last_limits)

    def _solve_current(self, power_w: float, params: dict):
        veq = params["voc"] - self.state.vp1_v - self.state.vp2_v
        rs = max(params["rs"], 1e-12)
        p_cell = float(power_w) / (self.ns * self.np)
        discriminant = veq * veq - 4.0 * rs * p_cell
        if discriminant < -1e-7:
            raise ValueError("Battery power request has no real current solution")
        discriminant = max(0.0, discriminant)
        denom = veq + np.sqrt(discriminant)
        if abs(denom) < 1e-12:
            raise ValueError("Battery terminal voltage is too small to solve current")
        return 2.0 * p_cell / denom

    def _aging_parameters(self, current_cell_a: float, temp_c: float):
        rates = np.asarray(self.cfg["aging_b_c_rate_points"], dtype=np.float64)
        b_values = np.asarray(self.cfg["aging_b_values"], dtype=np.float64)
        c_rate = abs(current_cell_a) / max(self.q_nom_ah, 1e-12)
        c_table = float(np.clip(c_rate, rates[0], rates[-1]))
        b_c = float(np.interp(c_table, rates, b_values))
        activation = (
            self.cfg["aging_activation_energy_base_j_per_mol"]
            - self.cfg["aging_activation_energy_c_rate_slope_j_per_mol"] * c_rate
        )
        temp_k = float(temp_c) + 273.15
        if temp_k <= 0:
            raise ValueError("Arrhenius aging requires positive Kelvin temperature")
        z = self.cfg["aging_power_law_exponent"]
        # Eq. (25): Ah_20 is the amp-hour throughput at the 20% capacity-loss
        # threshold. N=3600*Ah_20/Cn when Cn is expressed in coulombs; this is
        # dimensionally equivalent to Ah_20 / cell_capacity_Ah.
        exponent = np.clip(activation / (self.cfg["gas_constant_j_per_mol_k"] * temp_k), -100, 100)
        ah20 = (20.0 * np.exp(exponent) / b_c) ** (1.0 / z)
        n_cycles = 3600.0 * ah20 / self.q_nom_c
        return float(c_rate), float(ah20), float(n_cycles), bool(c_table != c_rate)

    def step(self, requested_power_w: float, dt_s: float) -> BatteryStep:
        p_req = float(requested_power_w)
        dt = float(dt_s)
        if dt <= 0:
            raise ValueError("Battery time step must be positive")
        direction = "charge" if p_req < 0.0 else "discharge"
        params, clipped = self._electrical_parameters(direction)
        limits = self._cell_current_limits()
        tol = max(1e-5, 1e-8 * max(abs(limits["minimum_power_w"]), abs(limits["maximum_power_w"]), 1.0))
        if p_req < limits["minimum_power_w"] - tol or p_req > limits["maximum_power_w"] + tol:
            raise ValueError(
                f"Battery power {p_req:.3f} W outside dynamic limits "
                f"[{limits['minimum_power_w']:.3f}, {limits['maximum_power_w']:.3f}] W"
            )
        i_cell = self._solve_current(p_req, params)
        if i_cell < limits["minimum_cell_current_a"] - 1e-7 or i_cell > limits["maximum_cell_current_a"] + 1e-7:
            raise ValueError("Battery current violates C-rate, SOC, or voltage limits")
        q_eff = self.effective_cell_capacity_ah
        soc_new = self.state.soc - i_cell * dt / (3600.0 * q_eff)
        if soc_new < self.cfg["minimum_soc"] - 1e-9 or soc_new > self.cfg["maximum_soc"] + 1e-9:
            raise ValueError("Battery SOC would leave the configured operating range")
        v_cell = params["voc"] - self.state.vp1_v - self.state.vp2_v - i_cell * params["rs"]
        if v_cell < self.cfg["minimum_cell_voltage_v"] - 1e-7 or v_cell > self.cfg["maximum_cell_voltage_v"] + 1e-7:
            raise ValueError("Battery terminal voltage violates cell limits")

        # Exact first-order RC-branch update for a constant current over dt.
        decay1 = np.exp(-dt / max(params["r1"] * params["c1"], 1e-12))
        decay2 = np.exp(-dt / max(params["r2"] * params["c2"], 1e-12))
        vp1_new = self.state.vp1_v * decay1 + params["r1"] * i_cell * (1.0 - decay1)
        vp2_new = self.state.vp2_v * decay2 + params["r2"] * i_cell * (1.0 - decay2)

        # Lin/Peng Eq. (21): I*(Vp1+Vp2+Rs*I) plus reversible entropy heat.
        # The ohmic term is always nonnegative; I*Vp can be negative briefly
        # when a previously charged polarization branch releases stored energy.
        ohmic_and_polarization_heat = i_cell * (
            self.state.vp1_v + self.state.vp2_v + params["rs"] * i_cell
        )
        average_temp_c = 0.5 * (self.state.core_temperature_c + self.state.surface_temperature_c)
        entropy_heat = i_cell * (average_temp_c + 273.15) * self.tables.entropy_v_per_k(self.state.soc)
        heat = ohmic_and_polarization_heat + entropy_heat

        cc = self.tables.core_capacity_j_per_k
        cs = self.tables.surface_capacity_j_per_k
        rc = self.tables.core_surface_resistance_k_per_w
        ru = self.cfg["ru_k_per_w"]
        tf = self.cfg["coolant_temperature_c"]
        tc0 = self.state.core_temperature_c
        ts0 = self.state.surface_temperature_c
        m11 = cc / dt + 1.0 / rc
        m12 = -1.0 / rc
        m21 = -1.0 / rc
        m22 = cs / dt + 1.0 / rc + 1.0 / ru
        rhs1 = cc / dt * tc0 + heat
        rhs2 = cs / dt * ts0 + tf / ru
        det = m11 * m22 - m12 * m21
        tc_new = (rhs1 * m22 - m12 * rhs2) / det
        ts_new = (m11 * rhs2 - m21 * rhs1) / det

        c_rate, ah20, n_cycles, aging_c_clipped = self._aging_parameters(
            i_cell, average_temp_c
        )
        del c_rate, ah20
        # Eq. (22), with current in A, dt in s, and Cn in coulombs.
        soh_decrement = abs(i_cell) * dt / max(2.0 * n_cycles * self.q_nom_c, 1e-12)
        soh_new = max(0.0, self.state.soh - soh_decrement)
        self.state = BatteryState(
            soc=float(np.clip(soc_new, self.cfg["minimum_soc"], self.cfg["maximum_soc"])),
            vp1_v=float(vp1_new), vp2_v=float(vp2_new),
            core_temperature_c=float(tc_new), surface_temperature_c=float(ts_new),
            soh=float(soh_new),
        )
        self.last_parameters = params
        self.last_limits = limits
        return BatteryStep(
            p_req, p_req, i_cell * self.np, i_cell,
            v_cell * self.ns, v_cell, float(heat), float(soh_decrement),
            self.state, bool(clipped), bool(aging_c_clipped),
        )
