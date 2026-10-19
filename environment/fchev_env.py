from __future__ import annotations

from pathlib import Path
import json
import tempfile

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.scenario.build_scenario import (
    build_scenario,
    generate_episode_randomization,
)
from environment.sumo_interface import SumoInterface
from util.aging import EDUDegradation, FCSDegradation
from util.battery_model import BatteryModel
from util.data import ProjectData
from util.edu_model import EDUModel
from util.fcs_model import FCSModel
from util.reward import compute_reward, following_reward, safety_distance
from util.vehicle_dynamics import longitudinal_demand


FEASIBILITY_BISECTION_ITERATIONS = 18


class FCHEVEnv(gym.Env):
    """SUMO-coupled 15-state FCHEV longitudinal-control environment."""

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
            low=np.asarray([
                0.15, 0, 0, 0, -6.68, 0, -6.68, 0, 0.0,
                0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ], dtype=np.float32),
            high=np.asarray([
                0.90, 1, 1, 1, 0.7, self.vehicle_cfg["max_speed_mps"],
                0.7, self.vehicle_cfg["max_speed_mps"], 30000, 2, 30000, 100,
                self.vehicle_cfg["max_speed_mps"], self.vehicle_cfg["max_speed_mps"],
                30000,
            ], dtype=np.float32),
            dtype=np.float32,
        )
        self._scenario_tmp = None
        scenario_output_dir = self.sumo_cfg.get("scenario_output_dir")
        if scenario_output_dir is None:
            self._scenario_tmp = tempfile.TemporaryDirectory(prefix="fchev_env_scenario_")
            scenario_output_dir = self._scenario_tmp.name
        self.scenario_output_dir = Path(scenario_output_dir).expanduser()
        if not self.scenario_output_dir.is_absolute():
            self.scenario_output_dir = self.project_root / self.scenario_output_dir
        self.scenario_output_dir = self.scenario_output_dir.resolve()
        self.scenario_cfg = build_scenario(
            self.project_root, self.sumo_cfg, self.vehicle_cfg,
            output_dir=self.scenario_output_dir,
        )
        self.sumo = SumoInterface(
            self.project_root, self.scenario_cfg, self.sumo_cfg, self.vehicle_cfg
        )
        self._seed = int(self.config.get("seed", 0))
        self._step_count = 0
        self._last_accel = 0.0
        self._last_fcs_power = 0.0
        self._last_x = None
        self._last_gap_m = None
        self._stuck_steps = 0
        self._last_state = None
        self._last_reward_components = {}
        self._ego_passed_tls_ids = set()
        self._lead_passed_tls_ids = set()

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
        tls = states.get("next_tls")
        if tls is None:
            tls_state, tls_distance, tls_remaining = 1.0, 30000.0, 0.0
        else:
            state = str(tls["state"]).lower()
            state = state[:1]
            tls_state = 0.0 if state == "r" else 2.0 if state == "y" else 1.0
            tls_distance = float(np.clip(tls["distance_m"], 0.0, 30000.0))
            tls_remaining = float(np.clip(tls.get("remaining_s", 0.0), 0.0, 100.0))
        current_speed_limit, next_speed_limit, distance_to_next_speed_region = (
            self._speed_region_observation(states, ego, tls)
        )
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
            tls_state,
            tls_distance,
            tls_remaining,
            current_speed_limit,
            next_speed_limit,
            distance_to_next_speed_region,
        ], dtype=np.float32)

    def _speed_region_observation(self, states, ego, tls):
        """Return current area cap and the next speed-control point.

        Ordinary road and signalized-intersection approach regions use the
        configured SUMO lane and intersection limits. A red light is a zero
        speed target at its stop line, not a zero legal speed throughout the
        approach area.
        """
        road_limit = float(self.vehicle_cfg["max_speed_mps"])
        intersection_limit = min(
            road_limit,
            float(self.sumo_cfg.get("intersection_speed_limit_kph", 30.0)) / 3.6,
        )
        zone_length = float(self.sumo_cfg.get("intersection_speed_zone_m", 100.0))
        in_zone = self._inside_intersection_speed_zone(ego, tls, zone_length)
        current_limit = intersection_limit if in_zone else road_limit

        upcoming = list(states.get("upcoming_tls") or [])
        if tls is None and upcoming:
            tls = upcoming[0]
        if tls is None:
            return current_limit, road_limit, 30000.0

        # The nearest red signal is a stop target, and its distance is to the
        # stop line. This takes precedence over the 30 km/h zone entrance.
        if str(tls.get("state", "")).lower()[:1] == "r":
            return current_limit, 0.0, float(np.clip(tls["distance_m"], 0.0, 30000.0))

        # Once the vehicle has entered the current intersection speed zone,
        # report the following region's entry instead of leaving this distance
        # at zero. On the approach lane, the first signal belongs to the zone
        # already entered; on an internal junction lane, SUMO's first returned
        # signal is already the next signal on the route.
        ego_lane = str(ego.get("lane_id", ""))
        tls_distance = float(tls["distance_m"])
        skip_current_signal = (
            in_zone
            and not ego_lane.startswith(":")
            and 0.0 <= tls_distance <= zone_length
        )
        candidate = tls
        if skip_current_signal:
            future = [
                item for item in upcoming
                if item.get("id") != tls.get("id")
                and float(item.get("distance_m", -1.0)) > tls_distance + 1e-6
            ]
            candidate = future[0] if future else None

        if candidate is None:
            return current_limit, road_limit, 30000.0
        candidate_distance = float(candidate["distance_m"])
        if str(candidate.get("state", "")).lower()[:1] == "r":
            next_target = 0.0
            distance = candidate_distance
        else:
            next_target = intersection_limit
            distance = max(0.0, candidate_distance - zone_length)
        return (
            current_limit,
            next_target,
            float(np.clip(distance, 0.0, 30000.0)),
        )

    def _dispatch(self, p_edu_w, requested_fcs_kw, battery_limits=None):
        limits = battery_limits if battery_limits is not None else self.battery.power_limits()
        eta_dc = self.config["fcs"]["dc_dc_efficiency"]
        pfc_requested_w = float(np.clip(requested_fcs_kw, 0.0, self.fcs.maximum_power_kw)) * 1000.0
        pfc_w = pfc_requested_w
        requested_battery_power_w = float(p_edu_w - eta_dc * pfc_w)

        # Keep the battery's physical C-rate, SOC, and voltage limits intact.
        # If a commanded FCS output plus regenerative power would overcharge it,
        # clamp battery charging at its BMS limit and absorb the surplus on an
        # explicit ideal DC-bus dump load. This removes battery charge acceptance
        # as a reason to silently clip the agent's FCS action.
        if requested_battery_power_w > limits["maximum_power_w"] + 1e-5:
            return None
        p_batt_w = max(requested_battery_power_w, limits["minimum_power_w"])
        dump_power_w = max(0.0, p_batt_w - requested_battery_power_w)
        return {
            "fcs_power_kw": pfc_w / 1000.0,
            "battery_power_w": p_batt_w,
            "dump_power_w": dump_power_w,
            "fcs_clipped": abs(pfc_w / 1000.0 - requested_fcs_kw) > 1e-7,
        }

    def _candidate(self, speed_m_s, acceleration_m_s2, requested_fcs_kw, battery_limits=None):
        demand = longitudinal_demand(speed_m_s, acceleration_m_s2, self.dt, self.vehicle_cfg)
        torque = demand.motor_torque_nm
        regen_torque_clipped = False
        if torque < 0.0:
            _, min_torque = self.edu.torque_limits(demand.motor_speed_rpm)
            allowed_torque = max(torque, min_torque)
            regen_torque_clipped = allowed_torque > torque + 1e-8
            point = self.edu.lookup(demand.motor_speed_rpm, allowed_torque)
            dispatch = self._dispatch(point.electrical_power_w, requested_fcs_kw, battery_limits) if point.feasible else None
            if dispatch is None:
                # The EPA valid operating mask can be narrower than its scalar
                # torque envelope at some speeds. Battery charge acceptance can
                # also limit regen. In either case, reduce regen torque toward
                # zero and let friction brakes supply the remaining deceleration.
                lo, hi = allowed_torque, 0.0
                zero_point = self.edu.lookup(demand.motor_speed_rpm, 0.0)
                zero_dispatch = self._dispatch(zero_point.electrical_power_w, requested_fcs_kw, battery_limits) if zero_point.feasible else None
                if zero_dispatch is None:
                    return None
                best = (zero_point, zero_dispatch)
                for _ in range(FEASIBILITY_BISECTION_ITERATIONS):
                    mid = 0.5 * (lo + hi)
                    p = self.edu.lookup(demand.motor_speed_rpm, mid)
                    d = self._dispatch(p.electrical_power_w, requested_fcs_kw, battery_limits) if p.feasible else None
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
            dispatch = self._dispatch(point.electrical_power_w, requested_fcs_kw, battery_limits)
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

    def _make_plan(self, speed_m_s, acceleration_m_s2, requested_fcs_kw, battery_limits=None):
        if battery_limits is None:
            battery_limits = self.battery.power_limits()
        requested_target = float(acceleration_m_s2)
        v = float(speed_m_s)
        nonreverse_target = max(requested_target, -v / self.dt)
        target = nonreverse_target
        no_reverse_clip = nonreverse_target > requested_target + 1e-12
        # Maximum-speed requests are handled as terminal events in step(); they
        # are no longer silently clipped to zero acceleration here.
        speed_limit_clip = False
        candidate = self._candidate(v, target, requested_fcs_kw, battery_limits)
        if candidate is not None:
            return target, candidate, False, no_reverse_clip, speed_limit_clip
        lower = max(-self.vehicle_cfg["emergency_deceleration_m_s2"], -v / self.dt)
        low_candidate = self._candidate(v, lower, requested_fcs_kw, battery_limits)
        if low_candidate is None:
            raise RuntimeError("No feasible EDU/battery operating point exists even under braking")
        low, high = lower, target
        best = low_candidate
        for _ in range(FEASIBILITY_BISECTION_ITERATIONS):
            mid = 0.5 * (low + high)
            candidate = self._candidate(v, mid, requested_fcs_kw, battery_limits)
            if candidate is None:
                high = mid
            else:
                low, best = mid, candidate

        return low, best, True, no_reverse_clip, speed_limit_clip

    @staticmethod
    def _crossed_traffic_light(tls_before, tls_after, vehicle_before, vehicle_after):
        """Detect a forward crossing of the next signal stop line."""
        if not tls_before:
            return False
        travel_m = max(
            0.0,
            float(vehicle_after["x_m"]) - float(vehicle_before["x_m"]),
        )
        if travel_m <= 1e-6:
            return False
        distance_m = float(tls_before["distance_m"])
        crossed = distance_m <= travel_m + 1e-6
        if tls_after is None or tls_after.get("id") != tls_before.get("id"):
            crossed = True
        if str(vehicle_after.get("lane_id", "")).startswith(":"):
            crossed = True
        return bool(crossed)

    @staticmethod
    def _crossed_red_light(tls_before, tls_after, ego_before, ego_after):
        """Detect a signal crossing commanded while its link was red."""
        return bool(
            tls_before
            and str(tls_before.get("state", "")).lower()[:1] == "r"
            and FCHEVEnv._crossed_traffic_light(
                tls_before, tls_after, ego_before, ego_after
            )
        )

    @staticmethod
    def _inside_intersection_speed_zone(vehicle_state, tls, zone_length_m):
        if vehicle_state is None:
            return False
        if str(vehicle_state.get("lane_id", "")).startswith(":"):
            return True
        return bool(
            tls is not None
            and -1e-6 <= float(tls.get("distance_m", float("inf")))
            <= float(zone_length_m)
        )

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
        self._ego_passed_tls_ids = set()
        self._lead_passed_tls_ids = set()
        randomization = generate_episode_randomization(self.sumo_cfg, self._seed)
        states = self.sumo.start(seed=self._seed, randomization=randomization)
        self._last_state = states
        self._last_x = states["ego"]["x_m"] if states["ego"] else None
        self._last_gap_m = float(states.get("gap_m", 0.0))
        self._last_reward_components = {}
        info = {
            "sumo_time_s": states["time_s"],
            "sumo_initialization_step_excluded": True,
            "traffic_light_count": self.sumo.signal_count,
            "scenario_seed": self._seed,
            "signal_initial_conditions": randomization["signal_initial_conditions"],
            "lead_stop_plan": randomization["lead_stop_plan"],
            "initial_gap_m": float(states["gap_m"]),
            "gap_diagnostic_threshold_m": float(self.sumo_cfg["gap_diagnostic_threshold_m"]),
            "gap_monitor_threshold_m": float(
                self.sumo_cfg.get("gap_monitor_threshold_m", 1000.0)
            ),
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
        max_speed = float(self.vehicle_cfg["max_speed_mps"])
        requested_next_speed = float(ego0["speed_m_s"]) + requested_accel * self.dt
        max_speed_action_violation = bool(
            requested_accel > 1e-9
            and requested_next_speed > max_speed + 1e-6
        )
        if max_speed_action_violation:
            # End on the action that would exceed the speed envelope. No SUMO
            # step is taken and no surrogate clipped action is executed.
            self._step_count += 1
            next_observation = self._observation(states_before)
            self._last_state = states_before
            terminal_penalty = float(
                self.config["reward"].get(
                    "abnormal_termination_penalty",
                    self.config["reward"].get(
                        "collision_red_light_terminal_penalty", -500.0
                    ),
                )
            )
            self._last_reward_components = {
                "terminal_penalty": terminal_penalty,
                "reward_contribution_terminal_penalty": terminal_penalty,
            }
            info = {
                "sumo_time_s": states_before["time_s"],
                "requested_acceleration_raw_m_s2": float(requested_raw[0]),
                "requested_fcs_power_raw_kw": float(requested_raw[1]),
                "requested_acceleration_m_s2": requested_accel,
                "fcs_power_requested_kw": requested_fcs,
                "powertrain_limited_acceleration_m_s2": None,
                "executed_acceleration_m_s2": None,
                "max_speed_m_s": max_speed,
                "requested_next_speed_m_s": requested_next_speed,
                "max_speed_action_violation": True,
                "speed_limit_clip": False,
                "physical_powertrain_clip": False,
                "no_reverse_clip": False,
                "constraint_clipping": not np.array_equal(requested_raw, bounded),
                "driver_supervisor_override": False,
                "collision_count": 0,
                "collision_event": False,
                "gap_overlap": False,
                "red_light_violation": False,
                "speeding_violation": False,
                "speeding_vehicles": [],
                "terminal_penalty": terminal_penalty,
                "collision_penalty": 0.0,
                "speed_limit_positive_accel_penalty": 0.0,
                "zero_speed_negative_accel_penalty": 0.0,
                "abnormal_termination": True,
                "terminated_reason": "max_speed_action",
                "truncated_reason": None,
                "ego_speed_m_s": float(ego0["speed_m_s"]),
                "ego_x_m": float(ego0["x_m"]),
                "following_distance_m": float(states_before.get("gap_m", 30000.0)),
                # This terminal path returns before advancing SUMO, but the
                # cumulative crossing counters still describe the episode.
                "traffic_lights_passed": len(self._ego_passed_tls_ids),
                "lead_traffic_lights_passed": len(self._lead_passed_tls_ids),
                "traffic_light_state_observation": float(next_observation[9]),
                "traffic_light_distance_m": float(next_observation[10]),
                "traffic_light_remaining_s": float(next_observation[11]),
                "current_speed_limit_m_s": float(next_observation[12]),
                "next_speed_limit_m_s": float(next_observation[13]),
                "distance_to_next_speed_region_m": float(next_observation[14]),
                "reward_components": dict(self._last_reward_components),
            }
            return next_observation, terminal_penalty, True, False, info

        tls = states_before.get("next_tls")
        lead_tls_before = states_before.get("lead_next_tls")
        lead0 = states_before.get("lead")
        # Battery limits depend on the current SOC, RC polarization, and
        # temperature, none of which changes during candidate feasibility
        # checks. Calculate once and reuse throughout the plan search.
        battery_limits = self.battery.power_limits()
        planned_accel, plan, physical_clip, no_reverse_clip, speed_limit_clip = self._make_plan(
            ego0["speed_m_s"], requested_accel, requested_fcs, battery_limits
        )
        planned_speed = max(0.0, ego0["speed_m_s"] + planned_accel * self.dt)
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
        red_light_violation = self._crossed_red_light(
            tls, states_after.get("next_tls"), ego0, ego1
        )
        ego_tls_crossed = self._crossed_traffic_light(
            tls, states_after.get("next_tls"), ego0, ego1
        )
        ego_crossed_tls_id = str(tls["id"]) if ego_tls_crossed and tls else None
        ego_new_tls_crossing = bool(
            ego_crossed_tls_id and ego_crossed_tls_id not in self._ego_passed_tls_ids
        )
        if ego_new_tls_crossing:
            self._ego_passed_tls_ids.add(ego_crossed_tls_id)
        lead1 = states_after.get("lead")
        lead_tls_after = states_after.get("lead_next_tls")
        lead_tls_crossed = self._crossed_traffic_light(
            lead_tls_before, lead_tls_after, lead0, lead1
        ) if lead0 is not None and lead1 is not None else False
        lead_crossed_tls_id = (
            str(lead_tls_before["id"])
            if lead_tls_crossed and lead_tls_before else None
        )
        lead_new_tls_crossing = bool(
            lead_crossed_tls_id and lead_crossed_tls_id not in self._lead_passed_tls_ids
        )
        if lead_new_tls_crossing:
            self._lead_passed_tls_ids.add(lead_crossed_tls_id)

        speed_limit_m_s = float(
            self.sumo_cfg.get("intersection_speed_limit_kph", 30.0)
        ) / 3.6
        speed_zone_m = float(self.sumo_cfg.get("intersection_speed_zone_m", 100.0))
        ego_speed_zone = (
            self._inside_intersection_speed_zone(ego0, tls, speed_zone_m)
            or self._inside_intersection_speed_zone(
                ego1, states_after.get("next_tls"), speed_zone_m
            )
        )
        lead_speed_zone = (
            self._inside_intersection_speed_zone(lead0, lead_tls_before, speed_zone_m)
            or self._inside_intersection_speed_zone(
                lead1, lead_tls_after, speed_zone_m
            )
        )
        speeding_vehicles = []
        if ego_speed_zone and max(
            float(ego0["speed_m_s"]), float(ego1["speed_m_s"])
        ) > speed_limit_m_s + 1e-3:
            speeding_vehicles.append("ego")
        if lead_speed_zone and lead0 is not None and lead1 is not None and max(
            float(lead0["speed_m_s"]), float(lead1["speed_m_s"])
        ) > speed_limit_m_s + 1e-3:
            speeding_vehicles.append("lead")
        speeding_violation = bool(speeding_vehicles)
        actual_accel = (ego1["speed_m_s"] - ego0["speed_m_s"]) / self.dt
        actual_plan = self._candidate(
            ego0["speed_m_s"], actual_accel, requested_fcs, battery_limits
        )
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
        battery_step = self.battery.step(
            dispatch["battery_power_w"], self.dt, power_limits=battery_limits
        )
        fcs_deg = self.fcs_aging.step(dispatch["fcs_power_kw"], self.dt)
        edu_deg = self.edu_aging.step(edu.loss_power_w, self.dt)
        gap = float(states_after.get("gap_m", states_before["gap_m"]))
        if not np.isfinite(gap):
            gap = 30000.0
        previous_gap = (
            float(self._last_gap_m)
            if self._last_gap_m is not None
            else float(states_before.get("gap_m", gap))
        )
        following_distance_reduction = previous_gap - gap
        collision = bool(collisions)
        gap_overlap = gap <= 0.0
        collision_event = collision or gap_overlap
        gap_over_diagnostic_threshold = gap > float(self.sumo_cfg["gap_diagnostic_threshold_m"])
        gap_over_monitor_threshold = gap > float(
            self.sumo_cfg.get("gap_monitor_threshold_m", 1000.0)
        )
        safety_min_distance, _ = safety_distance(ego1["speed_m_s"])
        safety_violation = gap < safety_min_distance or collision_event
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
            "following": following_reward(
                ego1["speed_m_s"], gap,
                self.config["reward"]["following_d_ref_m"],
                previous_gap,
                self.config["reward"]["following_c0"],
                self.config["reward"]["following_c1"],
                self.config["reward"]["following_c2"],
                self.config["reward"]["following_c3"],
            ),
            "following_distance_reduction_m": following_distance_reduction,
            "collision": collision_event,
            "safety_violation": safety_violation,
            "soc_penalty": abs(self.battery.state.soc - self.config["reward"]["target_soc"]),
        }
        reward = compute_reward(reward_parts, self.config["reward"])
        zero_speed_negative_accel_penalty = 0.0
        if ego0["speed_m_s"] <= 1e-6 and requested_accel < -1e-9:
            zero_speed_negative_accel_penalty = float(
                self.config["reward"].get("zero_speed_negative_accel_penalty", -10.0)
            )
        reward_parts["zero_speed_negative_accel_penalty"] = (
            zero_speed_negative_accel_penalty
        )
        reward_parts["reward_contribution_zero_speed_negative_accel_penalty"] = (
            zero_speed_negative_accel_penalty
        )
        reward += zero_speed_negative_accel_penalty
        collision_penalty = 0.0
        speed_limit_positive_accel_penalty = 0.0
        p_dc_loss = dispatch["fcs_power_kw"] * 1000.0 * (1.0 - self.config["fcs"]["dc_dc_efficiency"])
        balance_error = edu.electrical_power_w - (
            battery_step.terminal_power_w
            + self.config["fcs"]["dc_dc_efficiency"] * dispatch["fcs_power_kw"] * 1000.0
            - dispatch["dump_power_w"]
        )
        self._last_accel = actual_accel
        self._last_fcs_power = dispatch["fcs_power_kw"]
        self._step_count += 1
        self._last_gap_m = gap
        no_ego_progress = (
            self._last_x is not None
            and ego1["x_m"] <= self._last_x + 1e-6
            and ego1["speed_m_s"] < 0.1
        )
        if no_ego_progress:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        self._last_x = float(ego1["x_m"])
        self._last_state = states_after
        next_observation = self._observation(states_after)

        route_finished = "ego" in arrived or "ego" not in self.sumo.traci.vehicle.getIDList()
        lead = states_after.get("lead")
        terminal_lane = f"e{self.sumo.signal_count + 1}_0"
        terminal_position = float(self.sumo_cfg["lead_terminal_stop_offset_m"])
        lead_at_terminal_stop = bool(
            lead is not None
            and lead["lane_id"] == terminal_lane
            and lead["lane_position_m"] >= terminal_position - 1.0
            and lead["speed_m_s"] < 0.1
        )
        termination_reason = None
        abnormal_termination = False
        if collision_event:
            termination_reason = "collision"
            abnormal_termination = True
        elif red_light_violation:
            termination_reason = "red_light_violation"
            abnormal_termination = True
        elif speeding_violation:
            termination_reason = "speeding"
            abnormal_termination = True
        elif lead_at_terminal_stop:
            termination_reason = "lead_terminal_stop"
        elif route_finished:
            termination_reason = "route_complete"

        terminated = termination_reason is not None
        truncated = not terminated and self._step_count >= int(self.sumo_cfg["episode_limit_s"])
        terminal_penalty = 0.0
        if termination_reason in (
            "collision", "red_light_violation", "speeding", "max_speed_action"
        ):
            terminal_penalty = float(
                self.config["reward"].get(
                    "abnormal_termination_penalty",
                    self.config["reward"].get(
                        "collision_red_light_terminal_penalty", -500.0
                    ),
                )
            )
        reward_parts["terminal_penalty"] = terminal_penalty
        reward_parts["reward_contribution_terminal_penalty"] = terminal_penalty
        reward += terminal_penalty
        self._last_reward_components = reward_parts
        info = {
            "sumo_time_s": states_after["time_s"],
            "requested_acceleration_raw_m_s2": float(requested_raw[0]),
            "requested_fcs_power_raw_kw": float(requested_raw[1]),
            "requested_acceleration_m_s2": requested_accel,
            "executed_acceleration_m_s2": actual_accel,
            "powertrain_limited_acceleration_m_s2": planned_accel,
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
            "dc_bus_dump_power_w": dispatch["dump_power_w"],
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
            "following_distance_reduction_m": following_distance_reduction,
            "collision_count": len(collisions),
            "collision_ids": collisions,
            "collision_event": collision_event,
            "traffic_light_crossing": ego_new_tls_crossing,
            "traffic_light_crossed_id": ego_crossed_tls_id if ego_new_tls_crossing else None,
            "traffic_lights_passed": len(self._ego_passed_tls_ids),
            "lead_traffic_light_crossing": lead_new_tls_crossing,
            "lead_traffic_light_crossed_id": (
                lead_crossed_tls_id if lead_new_tls_crossing else None
            ),
            "lead_traffic_lights_passed": len(self._lead_passed_tls_ids),
            "red_light_violation": red_light_violation,
            "speeding_violation": speeding_violation,
            "speeding_vehicles": speeding_vehicles,
            "intersection_speed_limit_m_s": speed_limit_m_s,
            "ego_in_intersection_speed_zone": ego_speed_zone,
            "lead_in_intersection_speed_zone": lead_speed_zone,
            "gap_overlap": gap_overlap,
            "gap_over_diagnostic_threshold": gap_over_diagnostic_threshold,
            "gap_over_monitor_threshold": gap_over_monitor_threshold,
            "stuck_steps": self._stuck_steps,
            "stuck_duration_s": self._stuck_steps * self.dt,
            "leader_terminal_stop_reached": lead_at_terminal_stop,
            "abnormal_termination": abnormal_termination,
            "terminal_penalty": terminal_penalty,
            "collision_penalty": collision_penalty,
            "speed_limit_positive_accel_penalty": speed_limit_positive_accel_penalty,
            "zero_speed_negative_accel_penalty": zero_speed_negative_accel_penalty,
            "safety_violations": int(safety_violation),
            "constraint_clipping": bool(
                not np.array_equal(bounded, requested_raw)
                or no_reverse_clip or speed_limit_clip or physical_clip
                or dispatch["fcs_clipped"] or plan["regen_torque_clipped"]
                or sumo_intervention
            ),
            "requested_action_clipped": not np.array_equal(bounded, requested_raw),
            "driver_supervisor_override": False,
            "traffic_light": tls,
            "traffic_light_state_observation": float(next_observation[9]),
            "traffic_light_distance_m": float(next_observation[10]),
            "traffic_light_remaining_s": float(next_observation[11]),
            "current_speed_limit_m_s": float(next_observation[12]),
            "next_speed_limit_m_s": float(next_observation[13]),
            "distance_to_next_speed_region_m": float(next_observation[14]),
            "physical_powertrain_clip": physical_clip,
            "no_reverse_clip": no_reverse_clip,
            "speed_limit_clip": speed_limit_clip,
            "max_speed_action_violation": False,
            "fcs_power_clipped_for_battery_limits": dispatch["fcs_clipped"],
            "dc_bus_dump_load_active": dispatch["dump_power_w"] > 1e-6,
            "regen_torque_clipped_for_limits": plan["regen_torque_clipped"],
            "sumo_acceleration_intervention": sumo_intervention,
            "power_balance_residual_w": balance_error,
            "reward_components": reward_parts,
            "terminated_reason": termination_reason,
            "truncated_reason": "time_limit" if truncated else None,
        }
        return next_observation, float(reward), terminated, truncated, info

    def close(self):
        try:
            if hasattr(self, "sumo"):
                self.sumo.close()
        finally:
            if self._scenario_tmp is not None:
                self._scenario_tmp.cleanup()
                self._scenario_tmp = None


def load_config(path=None):
    p = Path(path or Path(__file__).resolve().parents[1] / "config" / "default.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
