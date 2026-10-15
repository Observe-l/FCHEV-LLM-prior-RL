from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import sys
import uuid


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
    def __init__(self, project_root: Path, sumo_cfg_file: Path, config: dict):
        self.project_root = Path(project_root)
        self.sumo_cfg_file = Path(sumo_cfg_file)
        self.cfg = config
        self.traci = _load_traci()
        self.label = f"fchev_{uuid.uuid4().hex[:10]}"
        self.started = False
        self.step_calls = 0
        self.signal_count = int(config["signal_count"])

    @staticmethod
    def _resolve_binary(binary: str):
        if os.path.isabs(binary) and Path(binary).exists():
            return binary
        found = shutil.which(binary)
        if not found:
            raise FileNotFoundError(f"SUMO executable not found: {binary}")
        return found

    def start(self, seed: int = 0):
        if self.started:
            self.close()
        binary = self._resolve_binary(self.cfg["gui_binary"] if self.cfg.get("gui") else self.cfg["binary"])
        cmd = [
            binary, "-c", str(self.sumo_cfg_file), "--seed", str(int(seed)),
            "--no-warnings", "true", "--no-step-log", "true",
            "--duration-log.disable", "true", "--collision.action",
            self.cfg.get("collision_action", "warn"), "--time-to-teleport", "-1",
        ]
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
        return self.states()

    def states(self):
        self.traci.switch(self.label)
        out = {}
        for vehicle_id in ("lead", "ego"):
            if vehicle_id not in self.traci.vehicle.getIDList():
                out[vehicle_id] = None
                continue
            res = self.traci.vehicle.getSubscriptionResults(vehicle_id) or {}
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
            out[vehicle_id] = {
                "x_m": float(pos[0]), "y_m": float(pos[1]),
                "lane_position_m": float(lane_position),
                "speed_m_s": float(speed),
                "acceleration_m_s2": float(acceleration),
                "length_m": float(length),
                "lane_id": str(lane_id),
            }
        if out["ego"] is not None and out["lead"] is not None:
            leader = self.traci.vehicle.getLeader("ego", 10000.0)
            if leader and leader[0] == "lead":
                gap = float(leader[1])
            else:
                gap = out["lead"]["x_m"] - out["ego"]["x_m"] - out["ego"]["length_m"]
            out["gap_m"] = max(-out["ego"]["length_m"], gap)
        else:
            out["gap_m"] = float("inf")
        out["time_s"] = float(self.traci.simulation.getTime())
        return out

    def next_tls(self):
        self.traci.switch(self.label)
        values = self.traci.vehicle.getNextTLS("ego")
        if not values:
            return None
        tls_id, link_index, distance, state = values[0]
        return {
            "id": str(tls_id), "link_index": int(link_index),
            "distance_m": float(distance), "state": str(state),
        }

    def set_ego_speed(self, speed_m_s: float):
        self.traci.switch(self.label)
        self.traci.vehicle.setSpeedMode("ego", 0)
        self.traci.vehicle.setSpeed("ego", max(0.0, float(speed_m_s)))

    def simulation_step(self):
        self.traci.switch(self.label)
        before = self.traci.simulation.getTime()
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
            self.traci.close(wait=False)
        finally:
            self.started = False

