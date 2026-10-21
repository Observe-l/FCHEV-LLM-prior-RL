from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import sys
import uuid


NO_FRONT_GAP_M = 30000.0


def _load_traci():
    try:
        import traci
        return traci
    except ImportError:
        candidates = []
        if os.environ.get("SUMO_HOME"):
            candidates.append(Path(os.environ["SUMO_HOME"]) / "tools")
        candidates.extend([Path("/usr/share/sumo/tools"), Path("/usr/local/share/sumo/tools")])
        for candidate in candidates:
            if (candidate / "traci").is_dir():
                sys.path.insert(0, str(candidate))
                import traci
                return traci
        raise RuntimeError("TraCI Python tools were not found; set SUMO_HOME or install SUMO tools")


class SumoInterface:
    def __init__(self, project_root: Path, sumo_cfg_file: Path, config: dict,
                 vehicle_config: dict | None = None):
        self.project_root = Path(project_root)
        self.sumo_cfg_file = Path(sumo_cfg_file)
        self.cfg = config
        self.vehicle_cfg = vehicle_config or {}
        self.traci = _load_traci()
        self.label = f"fchev_{uuid.uuid4().hex[:10]}"
        self.started = False
        self.step_calls = 0
        self.release_lead_start_speed_on_next_step = False
        self.fresh_start_state_pending = False
        self.signal_count = int(config["signal_count"])
        self.randomization = None
        self._lead_speed_limit_active = False
        self._lane_length_cache = {}

    @staticmethod
    def _resolve_binary(binary: str):
        if os.path.isabs(binary) and Path(binary).exists():
            return binary
        found = shutil.which(binary)
        if not found:
            raise FileNotFoundError(f"SUMO executable not found: {binary}")
        return found

    def start(self, seed: int = 0, randomization: dict | None = None):
        if self.started:
            self.close()
        self.randomization = randomization or {"seed": int(seed)}
        binary = self._resolve_binary(self.cfg["gui_binary"] if self.cfg.get("gui") else self.cfg["binary"])
        cmd = [
            binary, "-c", str(self.sumo_cfg_file), "--seed", str(int(seed)),
            "--no-warnings", "true", "--no-step-log", "true",
            "--duration-log.disable", "true", "--collision.action",
            self.cfg.get("collision_action", "warn"), "--time-to-teleport", "-1",
        ]
        if self.cfg.get("gui"):
            cmd.extend([
                "--start", "true", "--delay", str(int(self.cfg.get("gui_delay_ms", 400))),
                "--quit-on-end", "false",
            ])
        self.traci.start(cmd, label=self.label, verbose=False)
        self.traci.switch(self.label)
        self.started = True
        self.step_calls = 0
        actual_tls = self.traci.trafficlight.getIDList()
        if len(actual_tls) != self.signal_count:
            self.close()
            raise RuntimeError(
                f"SUMO loaded {len(actual_tls)} traffic signals; expected {self.signal_count}"
            )
        self.traci.vehicle.subscribe("lead", [
            self.traci.constants.VAR_POSITION,
            self.traci.constants.VAR_LANEPOSITION,
            self.traci.constants.VAR_SPEED,
            self.traci.constants.VAR_ACCELERATION,
            self.traci.constants.VAR_LENGTH,
            self.traci.constants.VAR_LANE_ID,
        ])
        self.traci.vehicle.subscribe("ego", [
            self.traci.constants.VAR_POSITION,
            self.traci.constants.VAR_LANEPOSITION,
            self.traci.constants.VAR_SPEED,
            self.traci.constants.VAR_ACCELERATION,
            self.traci.constants.VAR_LENGTH,
            self.traci.constants.VAR_LANE_ID,
        ])
        # Load both depart=0 vehicles. This initialization step is not an RL step.
        self.traci.simulationStep()
        self.traci.vehicle.setSpeedMode("ego", 0)
        # The bootstrap tick required to load depart=0 vehicles can advance
        # them. Restore the requested episode-start poses and zero speeds before
        # exposing the initial observation. The lead's IDM control is released
        # immediately before the first RL simulation tick.
        ego_pos = float(self.cfg["ego_depart_pos_m"])
        lead_pos = (
            ego_pos + float(self.cfg["initial_gap_m"])
            + float(self.vehicle_cfg.get("length_m", 8.0)) + 3.0
        )
        self.traci.vehicle.moveTo("ego", "e0_0", ego_pos)
        self.traci.vehicle.moveTo("lead", "e0_0", lead_pos)
        self.traci.vehicle.setSpeed("ego", 0.0)
        self.traci.vehicle.setSpeed("lead", 0.0)
        self._lead_speed_limit_active = False
        self._configure_traffic_lights()
        self._install_lead_stops()
        self.release_lead_start_speed_on_next_step = True
        self.fresh_start_state_pending = True
        return self.states(force_fresh=True)

    def _configure_traffic_lights(self):
        """Apply deterministic per-light phase positions for this episode."""
        self.traci.switch(self.label)
        for phase in self.randomization.get("signal_initial_conditions", []):
            tls_id = str(phase["id"])
            self.traci.trafficlight.setPhase(tls_id, int(phase["phase_index"]))
            self.traci.trafficlight.setPhaseDuration(tls_id, float(phase["remaining_s"]))

    def _install_lead_stops(self):
        """Insert one seeded, timed stop on each inter-signal road section."""
        self.traci.switch(self.label)
        for index, stop in enumerate(self.randomization.get("lead_stop_plan", [])):
            self.traci.vehicle.insertStop(
                "lead", index, str(stop["edge_id"]),
                pos=float(stop["position_m"]), laneIndex=0,
                duration=float(stop["duration_s"]), flags=0,
            )

    def states(self, force_fresh: bool = False):
        self.traci.switch(self.label)
        out = {}
        use_fresh_state = force_fresh or self.fresh_start_state_pending
        if not force_fresh:
            self.fresh_start_state_pending = False
        for vehicle_id in ("lead", "ego"):
            if vehicle_id not in self.traci.vehicle.getIDList():
                out[vehicle_id] = None
                continue
            res = {} if use_fresh_state else (
                self.traci.vehicle.getSubscriptionResults(vehicle_id) or {}
            )
            c = self.traci.constants
            pos = res.get(c.VAR_POSITION)
            if pos is None:
                pos = self.traci.vehicle.getPosition(vehicle_id)
            lane_position = res.get(c.VAR_LANEPOSITION)
            if lane_position is None:
                lane_position = self.traci.vehicle.getLanePosition(vehicle_id)
            speed = res.get(c.VAR_SPEED)
            if speed is None:
                speed = self.traci.vehicle.getSpeed(vehicle_id)
            acceleration = res.get(c.VAR_ACCELERATION)
            if acceleration is None:
                acceleration = self.traci.vehicle.getAcceleration(vehicle_id)
            length = res.get(c.VAR_LENGTH)
            if length is None:
                length = self.traci.vehicle.getLength(vehicle_id)
            lane_id = res.get(c.VAR_LANE_ID)
            if lane_id is None:
                lane_id = self.traci.vehicle.getLaneID(vehicle_id)
            lane_length = self.lane_length_m(str(lane_id))
            out[vehicle_id] = {
                "x_m": float(pos[0]), "y_m": float(pos[1]),
                "lane_position_m": float(lane_position),
                "lane_length_m": float(lane_length),
                "speed_m_s": float(speed),
                "acceleration_m_s2": float(acceleration),
                "length_m": float(length),
                "lane_id": str(lane_id),
            }
        if out["ego"] is not None and out["lead"] is not None:
            leader = self.traci.vehicle.getLeader("ego", 10000.0)
            if out["lead"]["x_m"] < out["ego"]["x_m"]:
                # No front gap exists once the named lead is behind the ego.
                # Do not misclassify a negative relative position as overlap;
                # actual contacts are reported separately by SUMO collisions.
                gap = NO_FRONT_GAP_M
                out["front_gap_valid"] = False
            elif leader and leader[0] == "lead":
                gap = float(leader[1])
                out["front_gap_valid"] = True
            else:
                gap = out["lead"]["x_m"] - out["ego"]["x_m"] - out["ego"]["length_m"]
                out["front_gap_valid"] = True
            out["gap_m"] = max(-out["ego"]["length_m"], gap)
        else:
            out["gap_m"] = float("inf")
            out["front_gap_valid"] = False
        out["time_s"] = float(self.traci.simulation.getTime())
        out["upcoming_tls"] = (
            self.traffic_lights_ahead("ego") if out["ego"] is not None else []
        )
        out["next_tls"] = out["upcoming_tls"][0] if out["upcoming_tls"] else None
        out["lead_next_tls"] = self.next_tls("lead") if out["lead"] is not None else None
        return out

    def lane_length_m(self, lane_id: str):
        """Return a lane's length in meters, caching the static network value."""
        lane_id = str(lane_id)
        if lane_id in self._lane_length_cache:
            return self._lane_length_cache[lane_id]
        try:
            self.traci.switch(self.label)
            length = float(self.traci.lane.getLength(lane_id))
        except Exception:
            # Internal lane IDs vary with network generation. Returning zero
            # lets callers keep the stop-line boundary as a conservative
            # fallback when a lane is unavailable.
            length = 0.0
        self._lane_length_cache[lane_id] = length
        return length

    def traffic_lights_ahead(self, vehicle_id: str = "ego"):
        self.traci.switch(self.label)
        if vehicle_id not in self.traci.vehicle.getIDList():
            return []
        values = self.traci.vehicle.getNextTLS(vehicle_id)
        if not values:
            return []
        now = float(self.traci.simulation.getTime())
        result = []
        seen = set()
        for tls_id, link_index, distance, state in values:
            tls_id = str(tls_id)
            # SUMO may return multiple links for the same signal. Keep the
            # first route-relevant link so future-zone distances are unique.
            if tls_id in seen:
                continue
            seen.add(tls_id)
            remaining_s = 0.0
            if not result:
                next_switch = float(self.traci.trafficlight.getNextSwitch(tls_id))
                remaining_s = max(0.0, next_switch - now)
            result.append({
                "id": tls_id, "link_index": int(link_index),
                "distance_m": float(distance), "state": str(state),
                "remaining_s": remaining_s,
            })
        return result

    def next_tls(self, vehicle_id: str = "ego"):
        ahead = self.traffic_lights_ahead(vehicle_id)
        return ahead[0] if ahead else None

    def _update_lead_intersection_speed_limit(self):
        """Apply the optional 30 km/h limit before the lead reaches a signal."""
        if "lead" not in self.traci.vehicle.getIDList():
            return
        lane_id = self.traci.vehicle.getLaneID("lead")
        tls = self.next_tls("lead")
        approach_m = float(self.cfg.get("intersection_speed_approach_m", 200.0))
        in_approach = bool(tls is not None and 0.0 <= tls["distance_m"] <= approach_m)
        zone_enabled = bool(self.cfg.get("intersection_speed_zone_enabled", True))
        should_limit = zone_enabled and (lane_id.startswith(":") or in_approach)
        if should_limit == self._lead_speed_limit_active:
            return
        self.traci.vehicle.setMaxSpeed(
            "lead",
            float(self.cfg.get("intersection_speed_limit_kph", 30.0)) / 3.6
            if should_limit else float(self.vehicle_cfg.get("max_speed_mps", 23.3333333333)),
        )
        self._lead_speed_limit_active = should_limit

    def set_ego_speed(self, speed_m_s: float):
        self.traci.switch(self.label)
        self.traci.vehicle.setSpeedMode("ego", 0)
        self.traci.vehicle.setSpeed("ego", max(0.0, float(speed_m_s)))

    def simulation_step(self):
        self.traci.switch(self.label)
        before = self.traci.simulation.getTime()
        if self.release_lead_start_speed_on_next_step:
            # -1 releases the one-time zero-speed departure command so that
            # SUMO's IDM resumes accelerating the lead from rest.
            self.traci.vehicle.setSpeed("lead", -1.0)
            self.release_lead_start_speed_on_next_step = False
        self._update_lead_intersection_speed_limit()
        self.traci.simulationStep()
        after = self.traci.simulation.getTime()
        self.step_calls += 1
        return float(before), float(after)

    def collision_ids(self):
        self.traci.switch(self.label)
        return list(self.traci.simulation.getCollidingVehiclesIDList())

    def arrived_ids(self):
        self.traci.switch(self.label)
        return list(self.traci.simulation.getArrivedIDList())

    def close(self):
        if not self.started:
            return
        try:
            self.traci.switch(self.label)
            # Reap the SUMO subprocess before another episode/environment is
            # launched; otherwise long or parallel runs accumulate zombies.
            self.traci.close(wait=True)
        finally:
            self.started = False

