from __future__ import annotations

import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def generate_episode_randomization(sumo_cfg: dict, seed: int) -> dict:
    """Create deterministic per-episode signal starts and lead stop points."""
    cycle_s = int(sumo_cfg.get("signal_cycle_s", 90))
    green_s = int(sumo_cfg.get("signal_green_s", 40))
    yellow_s = int(sumo_cfg.get("signal_yellow_s", 3))
    red_s = int(sumo_cfg.get("signal_red_s", 47))
    if green_s + yellow_s + red_s != cycle_s:
        raise ValueError("Traffic-light phase durations must sum to the cycle duration")

    signal_seed, stop_seed = np.random.SeedSequence(int(seed)).spawn(2)
    signal_rng = np.random.default_rng(signal_seed)
    stop_rng = np.random.default_rng(stop_seed)
    signal_count = int(sumo_cfg["signal_count"])
    if signal_count > red_s:
        raise ValueError("Cannot assign unique initial phase durations to every traffic light")
    # Pick distinct cycle positions and distinct time-to-transition values so
    # neither phase starts nor first-phase remaining durations are synchronized.
    candidate_positions = signal_rng.permutation(cycle_s)
    phases = []
    used_remaining = set()
    for cycle_position in candidate_positions:
        position = int(cycle_position)
        if position < green_s:
            state, phase_index, elapsed = "G", 0, position
            phase_duration = green_s
        elif position < green_s + yellow_s:
            state, phase_index, elapsed = "y", 1, position - green_s
            phase_duration = yellow_s
        else:
            state, phase_index, elapsed = "r", 2, position - green_s - yellow_s
            phase_duration = red_s
        remaining = phase_duration - elapsed
        if remaining in used_remaining:
            continue
        used_remaining.add(remaining)
        phases.append({
            "id": f"n{len(phases) + 1}",
            "cycle_position_s": position,
            "phase_index": phase_index,
            "state": state,
            "elapsed_s": elapsed,
            "remaining_s": remaining,
        })
        if len(phases) == signal_count:
            break

    spacing = float(sumo_cfg["intersection_spacing_m"])
    margin = float(sumo_cfg.get("lead_stop_margin_m", 100.0))
    if spacing <= 2.0 * margin:
        raise ValueError("Intersection spacing must exceed twice the stop-point margin")
    min_duration, max_duration = map(
        float, sumo_cfg.get("lead_stop_duration_range_s", [50.0, 100.0])
    )
    if min_duration < 0.0 or max_duration < min_duration:
        raise ValueError("Invalid lead stop duration range")
    stops = []
    # e1..e(N-1) are exactly the road sections between adjacent signals.
    for segment in range(1, signal_count):
        stops.append({
            "edge_id": f"e{segment}",
            "position_m": float(stop_rng.uniform(margin, spacing - margin)),
            "duration_s": float(stop_rng.uniform(min_duration, max_duration)),
        })
    return {"seed": int(seed), "signal_initial_conditions": phases, "lead_stop_plan": stops}


def build_scenario(project_root: Path, sumo_cfg: dict, vehicle_cfg: dict) -> Path:
    """Build a straight route with exactly the configured number of signal nodes."""
    root = Path(project_root)
    out = root / "environment" / "scenario" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    node_file = out / "road.nod.xml"
    edge_file = out / "road.edg.xml"
    tls_file = out / "signals.tll.xml"
    net_file = out / "road.net.xml"
    route_file = out / "vehicles.rou.xml"
    cfg_file = out / "scenario.sumocfg"

    n_tls = int(sumo_cfg["signal_count"])
    spacing = float(sumo_cfg["intersection_spacing_m"])
    signal_ids = []
    nodes = ET.Element("nodes")
    # Unsignalized tail edges provide a shared stopping segment beyond the
    # signalized route, so the ego remains in the simulation until the lead stops.
    for i in range(n_tls + 3):
        attrs = {"id": f"n{i}", "x": f"{(i - 1) * spacing:.3f}", "y": "0.0"}
        if 1 <= i <= n_tls:
            attrs["type"] = "traffic_light"
            signal_ids.append(attrs["id"])
        else:
            attrs["type"] = "priority"
        ET.SubElement(nodes, "node", attrs)
    ET.ElementTree(nodes).write(node_file, encoding="utf-8", xml_declaration=True)

    edges = ET.Element("edges")
    max_speed = float(vehicle_cfg["max_speed_mps"])
    for i in range(n_tls + 2):
        ET.SubElement(edges, "edge", {
            "id": f"e{i}", "from": f"n{i}", "to": f"n{i + 1}",
            "numLanes": "1", "speed": f"{max_speed:.6f}",
            "priority": "1", "spreadType": "center",
        })
    ET.ElementTree(edges).write(edge_file, encoding="utf-8", xml_declaration=True)

    green = int(sumo_cfg.get("signal_green_s", 40))
    yellow = int(sumo_cfg.get("signal_yellow_s", 3))
    red = int(sumo_cfg.get("signal_red_s", 47))
    cycle = int(sumo_cfg.get("signal_cycle_s", 90))
    if green + yellow + red != cycle:
        raise ValueError("Traffic-light phase durations must sum to the cycle duration")
    logics = ET.Element("tlLogics")
    for i, tls_id in enumerate(signal_ids):
        logic = ET.SubElement(logics, "tlLogic", {
            "id": tls_id, "type": "static", "programID": "0",
            "offset": "0",
        })
        ET.SubElement(logic, "phase", {"duration": str(green), "state": "G"})
        ET.SubElement(logic, "phase", {"duration": str(yellow), "state": "y"})
        ET.SubElement(logic, "phase", {"duration": str(red), "state": "r"})
    ET.ElementTree(logics).write(tls_file, encoding="utf-8", xml_declaration=True)

    netconvert = root / "environment" / "scenario" / "netconvert"
    executable = "netconvert"
    subprocess.run([
        executable, "--node-files", str(node_file), "--edge-files", str(edge_file),
        "--tllogic-files", str(tls_file), "--output-file", str(net_file),
        "--no-turnarounds", "true", "--junctions.corner-detail", "0",
        "--no-warnings", "true",
    ], check=True, capture_output=True, text=True)

    routes = ET.Element("routes")
    common = {
        "vClass": "bus", "length": f"{vehicle_cfg['length_m']:.6f}",
        "width": "2.5", "accel": f"{vehicle_cfg['max_action_acceleration_m_s2']:.6f}",
        "decel": f"{abs(vehicle_cfg['min_action_acceleration_m_s2']):.6f}",
        "emergencyDecel": f"{vehicle_cfg['emergency_deceleration_m_s2']:.6f}",
        "minGap": "3.0", "tau": "1.5", "sigma": "0.0", "speedDev": "0.0",
        "carFollowModel": "IDM", "speedFactor": "1.0",
    }
    lead_attrs = dict(
        common, id="leadType", maxSpeed=f"{max_speed:.6f}",
        speedFactor=f"{sumo_cfg['lead_target_speed_mps'] / max_speed:.8f}",
    )
    ego_attrs = dict(common, id="egoType", maxSpeed=f"{max_speed:.6f}")
    ET.SubElement(routes, "vType", lead_attrs)
    ET.SubElement(routes, "vType", ego_attrs)
    ego_route_id = "egoRoute"
    lead_route_id = "leadRoute"
    # SUMO's getLeader gap excludes the ego vehicle length and the 3 m minGap.
    ego_depart_pos = float(sumo_cfg["ego_depart_pos_m"])
    lead_depart_pos = (
        ego_depart_pos + float(sumo_cfg["initial_gap_m"])
        + float(vehicle_cfg["length_m"]) + 3.0
    )
    ET.SubElement(routes, "route", {
        "id": ego_route_id, "edges": " ".join(f"e{i}" for i in range(n_tls + 2)),
    })
    ET.SubElement(routes, "route", {
        "id": lead_route_id, "edges": " ".join(f"e{i}" for i in range(n_tls + 2)),
    })
    lead_vehicle = ET.SubElement(routes, "vehicle", {
        "id": "lead", "type": "leadType", "route": lead_route_id,
        "depart": "0", "departLane": "0", "departPos": f"{lead_depart_pos:.3f}",
        "departSpeed": f"{sumo_cfg['lead_depart_speed_mps']:.3f}",
        "arrivalSpeed": "0.0",
    })
    # Hold the lead at a defined terminal stop on the shared unsignalized tail.
    # Without a persistent stop, SUMO removes it from the network after arrival,
    # making the following gap infinite instead of leaving a stationary target.
    ET.SubElement(lead_vehicle, "stop", {
        "lane": f"e{n_tls + 1}_0",
        "endPos": f"{float(sumo_cfg['lead_terminal_stop_offset_m']):.3f}",
        # SUMO otherwise treats an omitted duration as an immediate release;
        # keep the leader available as a stopped target through the episode.
        "duration": f"{float(sumo_cfg['episode_limit_s']) + 1.0:.3f}",
        "triggered": "false",
        "parking": "false",
    })
    ET.SubElement(routes, "vehicle", {
        "id": "ego", "type": "egoType", "route": ego_route_id,
        "depart": "0", "departLane": "0", "departPos": f"{ego_depart_pos:.3f}",
        "departSpeed": f"{sumo_cfg['ego_depart_speed_mps']:.3f}",
    })
    ET.ElementTree(routes).write(route_file, encoding="utf-8", xml_declaration=True)

    cfg = ET.Element("configuration")
    inputs = ET.SubElement(cfg, "input")
    ET.SubElement(inputs, "net-file", {"value": net_file.name})
    ET.SubElement(inputs, "route-files", {"value": route_file.name})
    time = ET.SubElement(cfg, "time")
    ET.SubElement(time, "begin", {"value": "0"})
    ET.SubElement(time, "end", {"value": str(sumo_cfg["episode_limit_s"] + 1)})
    ET.SubElement(time, "step-length", {"value": str(sumo_cfg["step_length_s"])})
    ET.ElementTree(cfg).write(cfg_file, encoding="utf-8", xml_declaration=True)
    if len(signal_ids) != n_tls:
        raise RuntimeError(f"Built {len(signal_ids)} signals; expected {n_tls}")
    return cfg_file
