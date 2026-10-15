from __future__ import annotations

import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


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
    # One unsignalized tail edge keeps the lead bus present until the ego can
    # finish its route after following the same 25 signalized intersections.
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

    logics = ET.Element("tlLogics")
    for i, tls_id in enumerate(signal_ids):
        green = 25 + (i % 4)
        yellow = 3
        red = 32 + (i % 5)
        period = green + yellow + red
        logic = ET.SubElement(logics, "tlLogic", {
            "id": tls_id, "type": "static", "programID": "0",
            "offset": str((13 * i + 7) % period),
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
        speedFactor=f"{sumo_cfg['lead_speed_mps'] / max_speed:.8f}",
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
        "id": ego_route_id, "edges": " ".join(f"e{i}" for i in range(n_tls + 1)),
    })
    ET.SubElement(routes, "route", {
        "id": lead_route_id, "edges": " ".join(f"e{i}" for i in range(n_tls + 2)),
    })
    ET.SubElement(routes, "vehicle", {
        "id": "lead", "type": "leadType", "route": lead_route_id,
        "depart": "0", "departLane": "0", "departPos": f"{lead_depart_pos:.3f}",
        "departSpeed": f"{sumo_cfg['lead_speed_mps']:.3f}",
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
