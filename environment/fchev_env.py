from __future__ import annotations

from pathlib import Path
import json

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.scenario.build_scenario import build_scenario
from environment.sumo_interface import SumoInterface
from util.aging import EDUDegradation, FCSDegradation
from util.battery_model import BatteryModel
from util.data import ProjectData
from util.edu_model import EDUModel
from util.fcs_model import FCSModel
from util.reward import compute_reward, safety_cost
from util.vehicle_dynamics import longitudinal_demand


class FCHEVEnv(gym.Env):
    """SUMO-coupled 9-state FCHEV ACC/energy-management environment."""

    metadata = {"render_modes": ["human", "none"], "render_fps": 1}

    def __init__(self, config_path=None, project_root=None, render_mode=None, config=None):
        super().__init__()
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        if config is None:
            config_path = Path(config_path or self.project_root / "config" / "default.json")
            with config_path.open("r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = config
        self.sumo_cfg = dict(self.config["sumo"])
        self.vehicle_cfg = dict(self.config["vehicle"])
        if render_mode == "human":
            self.sumo_cfg["gui"] = True
        self.dt = float(self.sumo_cfg["step_length_s"])

        self.data = ProjectData(self.project_root)
        self.edu = EDUModel(self.data.motor)
        self.fcs = FCSModel(self.data.fcs, self.config["fcs"]["maximum_power_kw"])
        self.battery = BatteryModel(self.data.battery, self.config["battery"])
        reference = self.edu.reference_point(
            self.config["edu"]["reference_mechanical_power_w"],
            self.config["edu"]["reference_torque_nm"],
            self.config["edu"]["reference_speed_rpm"],
        )
        self.edu_reference = reference
        self.edu_aging = EDUDegradation(
            reference.loss_power_w, self.config["edu"]["lifetime_hours"]
        )
        self.fcs_aging = FCSDegradation(self.config["fcs"])

        self.action_space = spaces.Box(
            low=np.asarray([self.vehicle_cfg["min_action_acceleration_m_s2"], 0.0], dtype=np.float32),
            high=np.asarray([self.vehicle_cfg["max_action_acceleration_m_s2"], self.config["fcs"]["maximum_power_kw"]], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.asarray([0.15, 0, 0, 0, -6.68, 0, -6.68, 0, -8.0], dtype=np.float32),
            high=np.asarray([0.90, 1, 1, 1, 0.7, self.vehicle_cfg["max_speed_mps"], 0.7, self.vehicle_cfg["max_speed_mps"], 30000], dtype=np.float32),
            dtype=np.float32,
        )
        self.scenario_cfg = build_scenario(self.project_root, self.sumo_cfg, self.vehicle_cfg)
        self.sumo = SumoInterface(self.project_root, self.scenario_cfg, self.sumo_cfg)
        self._seed = int(self.config.get("seed", 0))
        self._step_count = 0
        self._last_accel = 0.0
        self._last_fcs_power = 0.0
        self._last_x = None
        self._stuck_steps = 0
        self._last_state = None
        self._last_reward_components = {}

    def _observation(self, states):
        last = self._last_state or {}
        ego = states.get("ego")
        lead = states.get("lead")
        if ego is None:
            ego = last.get("ego") or {
                "speed_m_s": 0.0, "acceleration_m_s2": 0.0, "x_m": 0.0,
            }
        if lead is None:
            lead = last.get("lead") or ego
        gap = states.get("gap_m", self.sumo_cfg["episode_limit_s"] * self.vehicle_cfg["max_speed_mps"])
        if not np.isfinite(gap):
            gap = float(self.observation_space.high[8])
        gap = float(np.clip(gap, self.observation_space.low[8], self.observation_space.high[8]))
        return np.asarray([
            self.battery.state.soc,
            self.battery.state.soh,
            1.0 - self.fcs_aging.cumulative,
            1.0 - self.edu_aging.cumulative,
            float(ego["acceleration_m_s2"]),
            float(ego["speed_m_s"]),
            float(lead["acceleration_m_s2"]),
            float(lead["speed_m_s"]),
            float(gap),
        ], dtype=np.float32)

    def _dispatch(self, p_edu_w, requested_fcs_kw):
        limits = self.battery.power_limits()
        eta_dc = self.config["fcs"]["dc_dc_efficiency"]
        pfc_requested_w = float(np.clip(requested_fcs_kw, 0.0, self.fcs.maximum_power_kw)) * 1000.0
        pfc_min_w = max(0.0, (p_edu_w - limits["maximum_power_w"]) / eta_dc)
        pfc_max_w = min(
            self.fcs.maximum_power_kw * 1000.0,
            (p_edu_w - limits["minimum_power_w"]) / eta_dc,
        )
        if pfc_min_w > pfc_max_w + 1e-6:
            return None
        pfc_w = float(np.clip(pfc_requested_w, pfc_min_w, pfc_max_w))
        p_batt_w = float(p_edu_w - eta_dc * pfc_w)
        if p_batt_w < limits["minimum_power_w"] - 1e-5 or p_batt_w > limits["maximum_power_w"] + 1e-5:
            return None
        return {
            "fcs_power_kw": pfc_w / 1000.0,
            "battery_power_w": p_batt_w,
            "fcs_clipped": abs(pfc_w / 1000.0 - requested_fcs_kw) > 1e-7,
        }

    def _candidate(self, speed_m_s, acceleration_m_s2, requested_fcs_kw):
        demand = longitudinal_demand(speed_m_s, acceleration_m_s2, self.dt, self.vehicle_cfg)
        torque = demand.motor_torque_nm
        regen_torque_clipped = False
        if torque < 0.0:
            _, min_torque = self.edu.torque_limits(demand.motor_speed_rpm)
            allowed_torque = max(torque, min_torque)
            regen_torque_clipped = allowed_torque > torque + 1e-8
            point = self.edu.lookup(demand.motor_speed_rpm, allowed_torque)
            dispatch = self._dispatch(point.electrical_power_w, requested_fcs_kw) if point.feasible else None
            if dispatch is None:
                # The EPA valid operating mask can be narrower than its scalar
                # torque envelope at some speeds. Battery charge acceptance can
                # also limit regen. In either case, reduce regen torque toward
                # zero and let friction brakes supply the remaining deceleration.
                lo, hi = allowed_torque, 0.0
                zero_point = self.edu.lookup(demand.motor_speed_rpm, 0.0)
                zero_dispatch = self._dispatch(zero_point.electrical_power_w, requested_fcs_kw) if zero_point.feasible else None
                if zero_dispatch is None:
                    return None
                best = (zero_point, zero_dispatch)
                for _ in range(36):
                    mid = 0.5 * (lo + hi)
                    p = self.edu.lookup(demand.motor_speed_rpm, mid)
                    d = self._dispatch(p.electrical_power_w, requested_fcs_kw) if p.feasible else None
                    if d is None:
                        lo = mid
                    else:
                        hi = mid
                        best = (p, d)
                point, dispatch = best
                regen_torque_clipped = True
            motor_force = point.torque_nm * self.vehicle_cfg["gear_ratio"] / self.vehicle_cfg["wheel_radius_m"]
            friction_force = min(0.0, demand.traction_force_n - motor_force)
        else:
            point = self.edu.lookup(demand.motor_speed_rpm, torque)
            if not point.feasible:
                return None
            dispatch = self._dispatch(point.electrical_power_w, requested_fcs_kw)
            if dispatch is None:
                return None
            friction_force = 0.0
        return {
            "demand": demand,
            "edu": point,
            "dispatch": dispatch,
            "motor_torque_nm": point.torque_nm,
            "friction_brake_force_n": friction_force,
            "regen_torque_clipped": regen_torque_clipped,
        }

    def _make_plan(self, speed_m_s, acceleration_m_s2, requested_fcs_kw):
        target = float(acceleration_m_s2)
        v = float(speed_m_s)
        target = max(target, -v / self.dt)
        target = min(target, (self.vehicle_cfg["max_speed_mps"] - v) / self.dt)
        candidate = self._candidate(v, target, requested_fcs_kw)
        if candidate is not None:
            return target, candidate, False
        lower = max(-self.vehicle_cfg["emergency_deceleration_m_s2"], -v / self.dt)
        low_candidate = self._candidate(v, lower, requested_fcs_kw)
        if low_candidate is None:
            raise RuntimeError("No feasible EDU/battery operating point exists even under braking")
        low, high = lower, target
        best = low_candidate
        for _ in range(40):
            mid = 0.5 * (low + high)
            candidate = self._candidate(v, mid, requested_fcs_kw)
            if candidate is None:
                high = mid
            else:
                low, best = mid, candidate
        return low, best, True

    def _safety_acceleration(self, states, requested_accel):
        ego = states["ego"]
        lead = states["lead"]
        v = ego["speed_m_s"]
        a = float(requested_accel)
        overridden = False
        reason = []
        vmax_accel = (self.vehicle_cfg["max_speed_mps"] - v) / self.dt
        if a > vmax_accel:
            a = vmax_accel
            overridden = True
            reason.append("speed_limit")
        if lead is not None:
            gap = float(states["gap_m"])
            lead_a = float(lead["acceleration_m_s2"])
            min_gap = 3.0
            lead_next_speed = max(0.0, lead["speed_m_s"] + lead_a * self.dt)
            ego_next_speed_limit = lead_next_speed + (gap - min_gap) / self.dt
            follow_bound = (ego_next_speed_limit - v) / self.dt
            if a > follow_bound:
                a = max(-self.vehicle_cfg["emergency_deceleration_m_s2"], follow_bound)
                overridden = True
                reason.append("following_safety")
        tls = self.sumo.next_tls()
        if tls and tls["state"].lower() in {"r", "y"}:
            distance = max(0.0, tls["distance_m"] - 2.0)
            stop_bound = distance / (self.dt * self.dt) - v / self.dt
            if a > stop_bound:
                a = max(-self.vehicle_cfg["emergency_deceleration_m_s2"], stop_bound)
                overridden = True
                reason.append("red_or_yellow_light")
        no_reverse_bound = -v / self.dt
        if a < no_reverse_bound:
            a = no_reverse_bound
            overridden = True
            reason.append("prevent_reverse_speed")
        return a, overridden, reason, tls

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = self._seed
        self._seed = int(seed)
        self.battery.reset()
        self.fcs_aging.reset()
        self.edu_aging.reset()
        self._step_count = 0
        self._last_accel = 0.0
        self._last_fcs_power = 0.0
        self._stuck_steps = 0
        states = self.sumo.start(seed=self._seed)
        self._last_state = states
        self._last_x = states["ego"]["x_m"] if states["ego"] else None
        self._last_reward_components = {}
        info = {
            "sumo_time_s": states["time_s"],
            "sumo_initialization_step_excluded": True,
            "traffic_light_count": self.sumo.signal_count,
            "edu_reference_efficiency": self.edu_reference.efficiency,
            "edu_reference_electrical_power_w": self.edu_reference.electrical_power_w,
            "edu_reference_loss_power_w": self.edu_reference.loss_power_w,
        }
        return self._observation(states), info

    def step(self, action):
        if not self.sumo.started:
            raise RuntimeError("Call reset() before step()")
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 2 or not np.all(np.isfinite(action)):
            raise ValueError("Action must be [acceleration_m_s2, FCS_power_kW]")
        requested_raw = action.copy()
        bounded = np.clip(action, self.action_space.low, self.action_space.high)
        requested_accel = float(bounded[0])
        requested_fcs = float(bounded[1])
        states_before = self.sumo.states()
        if states_before["ego"] is None:
            return self._observation(states_before), 0.0, True, False, {"termination_reason": "ego_not_active"}
        ego0 = states_before["ego"]
        safety_accel, safety_override, safety_reasons, tls = self._safety_acceleration(
            states_before, requested_accel
        )
        planned_accel, plan, physical_clip = self._make_plan(
            ego0["speed_m_s"], safety_accel, requested_fcs
        )
        planned_speed = max(0.0, min(
            self.vehicle_cfg["max_speed_mps"],
            ego0["speed_m_s"] + planned_accel * self.dt,
        ))
        self.sumo.set_ego_speed(planned_speed)
        time_before, time_after = self.sumo.simulation_step()
        if not np.isclose(time_after - time_before, self.dt, atol=1e-9):
            raise RuntimeError("SUMO advanced by a time interval different from one RL step")
        states_after = self.sumo.states()
        collisions = self.sumo.collision_ids()
        arrived = self.sumo.arrived_ids()
        ego1 = states_after["ego"]
        if ego1 is None:
            # Route completion removes the vehicle from TraCI; use the last commanded
            # interval for final energy accounting and mark a successful terminal.
            ego1 = dict(ego0)
            ego1["speed_m_s"] = plan["demand"].next_speed_m_s
            ego1["acceleration_m_s2"] = planned_accel
            states_after["ego"] = ego1
            states_after["gap_m"] = states_before["gap_m"]
        actual_accel = (ego1["speed_m_s"] - ego0["speed_m_s"]) / self.dt
        actual_plan = self._candidate(ego0["speed_m_s"], actual_accel, requested_fcs)
        sumo_intervention = abs(actual_accel - planned_accel) > 2e-3
        if actual_plan is None:
            raise RuntimeError(
                "SUMO executed acceleration produced an infeasible EDU/battery demand; "
                "inspect the recorded SUMO intervention and action constraint"
            )
        plan = actual_plan
        demand = plan["demand"]
        edu = plan["edu"]
        dispatch = plan["dispatch"]
        fcs_point = self.fcs.lookup(dispatch["fcs_power_kw"])
        battery_step = self.battery.step(dispatch["battery_power_w"], self.dt)
        fcs_deg = self.fcs_aging.step(dispatch["fcs_power_kw"], self.dt)
        edu_deg = self.edu_aging.step(edu.loss_power_w, self.dt)
        gap = float(states_after.get("gap_m", states_before["gap_m"]))
        if not np.isfinite(gap):
            gap = 30000.0
        collision = bool(collisions)
        reward_parts = {
            "hydrogen_g": fcs_point.hydrogen_rate_g_s * self.dt,
            "fcs_degradation_cumulative": fcs_deg.cumulative,
            "fcs_degradation_increment": fcs_deg.increment,
            "fcs_startup_degradation_percent": fcs_deg.startup_percent,
            "fcs_low_power_degradation_percent": fcs_deg.low_power_percent,
            "fcs_high_power_degradation_percent": fcs_deg.high_power_percent,
            "fcs_load_change_degradation_percent": fcs_deg.load_change_percent,
            "battery_soh_loss": battery_step.soh_decrement,
            "battery_soh_decrement": battery_step.soh_decrement,
            "edu_clr_cumulative": edu_deg.cumulative,
            "edu_clr_increment": edu_deg.increment,
            "comfort": abs(actual_accel - self._last_accel)
            / max(self.vehicle_cfg["max_action_acceleration_m_s2"] - self.vehicle_cfg["min_action_acceleration_m_s2"], 1e-9),
            "safety": safety_cost(ego1["speed_m_s"], gap, self.config["reward"], collision),
            "soc_penalty": abs(self.battery.state.soc - self.config["reward"]["target_soc"]),
        }
        reward = compute_reward(reward_parts, self.config["reward"])
        p_dc_loss = dispatch["fcs_power_kw"] * 1000.0 * (1.0 - self.config["fcs"]["dc_dc_efficiency"])
        balance_error = edu.electrical_power_w - (
            battery_step.terminal_power_w
            + self.config["fcs"]["dc_dc_efficiency"] * dispatch["fcs_power_kw"] * 1000.0
        )
        self._last_accel = actual_accel
        self._last_fcs_power = dispatch["fcs_power_kw"]
        self._step_count += 1
        if ego1["x_m"] <= self._last_x + 1e-6 and ego1["speed_m_s"] < 0.1:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        self._last_x = ego1["x_m"]
        self._last_state = states_after
        self._last_reward_components = reward_parts

        route_finished = "ego" in arrived or "ego" not in self.sumo.traci.vehicle.getIDList()
        terminated = collision or route_finished
        stuck = self._stuck_steps >= 300
        truncated = self._step_count >= int(self.sumo_cfg["episode_limit_s"]) or stuck
        info = {
            "sumo_time_s": states_after["time_s"],
            "requested_acceleration_raw_m_s2": float(requested_raw[0]),
            "requested_acceleration_m_s2": requested_accel,
            "executed_acceleration_m_s2": actual_accel,
            "safety_limited_acceleration_m_s2": safety_accel,
            "wheel_traction_force_n": demand.traction_force_n,
            "wheel_mechanical_power_w": demand.wheel_mechanical_power_w,
            "motor_speed_rpm": edu.speed_rpm,
            "motor_torque_nm": edu.torque_nm,
            "edu_electrical_power_w": edu.electrical_power_w,
            "edu_loss_power_w": edu.loss_power_w,
            "edu_efficiency": edu.efficiency,
            "friction_brake_force_n": plan["friction_brake_force_n"],
            "battery_power_w": battery_step.terminal_power_w,
            "battery_current_pack_a": battery_step.current_pack_a,
            "battery_current_cell_a": battery_step.current_cell_a,
            "battery_terminal_voltage_pack_v": battery_step.terminal_voltage_pack_v,
            "battery_terminal_voltage_cell_v": battery_step.terminal_voltage_cell_v,
            "battery_soc": self.battery.state.soc,
            "battery_soh": self.battery.state.soh,
            "battery_core_temperature_c": self.battery.state.core_temperature_c,
            "battery_surface_temperature_c": self.battery.state.surface_temperature_c,
            "battery_heat_generation_w_per_cell": battery_step.heat_generation_w_per_cell,
            "battery_power_min_w": self.battery.last_limits["minimum_power_w"],
            "battery_power_max_w": self.battery.last_limits["maximum_power_w"],
            "battery_parameter_table_clipped": battery_step.parameter_table_clipped,
            "battery_aging_c_rate_clipped": battery_step.aging_c_rate_clipped,
            "fcs_power_requested_kw": requested_fcs,
            "fcs_power_executed_kw": dispatch["fcs_power_kw"],
            "fcs_dc_dc_loss_w": p_dc_loss,
            "hydrogen_rate_g_s": fcs_point.hydrogen_rate_g_s,
            "hydrogen_consumed_g": reward_parts["hydrogen_g"],
            "fcs_efficiency_paper": fcs_point.efficiency_paper,
            "fcs_soh": fcs_deg.soh,
            "fcs_degradation_increment": fcs_deg.increment,
            "fcs_degradation_cumulative": fcs_deg.cumulative,
            "edu_soh": edu_deg.soh,
            "edu_clr_increment": edu_deg.increment,
            "edu_clr_cumulative": edu_deg.cumulative,
            "lead_x_m": states_after["lead"]["x_m"] if states_after["lead"] else None,
            "ego_x_m": ego1["x_m"],
            "lead_speed_m_s": states_after["lead"]["speed_m_s"] if states_after["lead"] else None,
            "ego_speed_m_s": ego1["speed_m_s"],
            "lead_acceleration_m_s2": states_after["lead"]["acceleration_m_s2"] if states_after["lead"] else None,
            "following_distance_m": gap,
            "collision_count": len(collisions),
            "collision_ids": collisions,
            "safety_violations": int(gap < 3.0 or collision),
            "constraint_clipping": bool(
                not np.array_equal(bounded, requested_raw)
                or safety_override or physical_clip or dispatch["fcs_clipped"]
                or plan["regen_torque_clipped"] or sumo_intervention
            ),
            "requested_action_clipped": not np.array_equal(bounded, requested_raw),
            "safety_override": safety_override,
            "safety_override_reasons": safety_reasons,
            "traffic_light": tls,
            "physical_powertrain_clip": physical_clip,
            "fcs_power_clipped_for_battery_limits": dispatch["fcs_clipped"],
            "regen_torque_clipped_for_limits": plan["regen_torque_clipped"],
            "sumo_acceleration_intervention": sumo_intervention,
            "power_balance_residual_w": balance_error,
            "reward_components": reward_parts,
            "terminated_reason": "collision" if collision else ("route_complete" if route_finished else None),
            "truncated_reason": "stuck" if stuck else ("time_limit" if truncated else None),
        }
        return self._observation(states_after), float(reward), terminated, truncated, info

    def close(self):
        if hasattr(self, "sumo"):
            self.sumo.close()


def load_config(path=None):
    p = Path(path or Path(__file__).resolve().parents[1] / "config" / "default.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
