from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from environment.scenario.build_scenario import (
    build_scenario,
    generate_episode_randomization,
)
from environment.sumo_interface import SumoInterface
from environment.fchev_env import load_config


def _state_at_cycle_position(position: int, green_s: int, yellow_s: int) -> str:
    if position < green_s:
        return "G"
    if position < green_s + yellow_s:
        return "y"
    return "r"


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(seed: int, output_root: Path, project_root: Path) -> Path:
    cfg = load_config(project_root / "config" / "default.json")
    sumo_cfg = cfg["sumo"]
    randomization = generate_episode_randomization(sumo_cfg, seed)
    sumo_cfg["gui"] = False
    scenario_cfg = build_scenario(project_root, sumo_cfg, cfg["vehicle"])
    sim = SumoInterface(project_root, scenario_cfg, sumo_cfg, cfg["vehicle"])
    output_dir = output_root / f"environment_preview_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lead_rows = []
    state_rows = []
    times = []
    signal_ids = [f"n{i}" for i in range(1, int(sumo_cfg["signal_count"]) + 1)]
    passed_signals: set[str] = set()
    previous = None
    stop_reached = False
    scenario_start_time_s = 0.0
    try:
        initial = sim.start(seed=seed, randomization=randomization)
        previous = initial
        scenario_start_time_s = float(initial["time_s"])
        for _ in range(int(sumo_cfg["episode_limit_s"])):
            sim.simulation_step()
            states = sim.states()
            lead = states.get("lead")
            if lead is None:
                break
            now = float(states["time_s"])
            times.append(now)
            lead_rows.append({
                "time_s": now,
                "position_m": lead["x_m"],
                "speed_m_s": lead["speed_m_s"],
                "speed_kph": 3.6 * lead["speed_m_s"],
                "acceleration_m_s2": lead["acceleration_m_s2"],
                "lane_id": lead["lane_id"],
                "lane_position_m": lead["lane_position_m"],
            })

            before_tls = previous.get("lead_next_tls") if previous else None
            after_tls = states.get("lead_next_tls")
            before_lead = previous.get("lead") if previous else None
            if before_tls and before_lead:
                travel = max(0.0, lead["x_m"] - before_lead["x_m"])
                crossed = (
                    float(before_tls["distance_m"]) <= travel + 1e-6
                    or after_tls is None
                    or after_tls.get("id") != before_tls.get("id")
                    or lead["lane_id"].startswith(":")
                )
                if crossed:
                    passed_signals.add(str(before_tls["id"]))

            terminal_lane = f"e{int(sumo_cfg['signal_count']) + 1}_0"
            if (
                lead["lane_id"] == terminal_lane
                and lead["lane_position_m"]
                >= float(sumo_cfg["lead_terminal_stop_offset_m"]) - 1.0
                and lead["speed_m_s"] < 0.1
            ):
                stop_reached = True

            previous = states
            if stop_reached:
                break

        cycle_s = int(sumo_cfg["signal_cycle_s"])
        green_s = int(sumo_cfg["signal_green_s"])
        yellow_s = int(sumo_cfg["signal_yellow_s"])
        phase_by_id = {
            phase["id"]: int(phase["cycle_position_s"])
            for phase in randomization["signal_initial_conditions"]
        }
        for time_s in times:
            # SUMO applies the seeded phase plan at simulation time 1 s, before
            # the first RL/simulation transition. These sampled states follow
            # the same fixed 40/3/47 s cycle as the live traffic lights.
            elapsed = int(round(time_s - scenario_start_time_s))
            row = {"time_s": time_s}
            for signal_id in signal_ids:
                pos = (phase_by_id[signal_id] + elapsed) % cycle_s
                row[signal_id] = _state_at_cycle_position(pos, green_s, yellow_s)
            state_rows.append(row)
    finally:
        sim.close()

    if not lead_rows:
        raise RuntimeError("SUMO preview did not produce any lead-vehicle samples")

    _write_csv(
        output_dir / "lead_speed_time.csv", lead_rows,
        ["time_s", "position_m", "speed_m_s", "speed_kph", "acceleration_m_s2", "lane_id", "lane_position_m"],
    )
    _write_csv(output_dir / "traffic_signal_timeline.csv", state_rows, ["time_s", *signal_ids])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    lead_time = np.asarray([row["time_s"] for row in lead_rows], dtype=float)
    lead_speed = np.asarray([row["speed_kph"] for row in lead_rows], dtype=float)
    speed_limit_kph = float(sumo_cfg["intersection_speed_limit_kph"])
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    spacing = float(sumo_cfg["intersection_spacing_m"])
    zone_length = float(sumo_cfg["intersection_speed_zone_m"])
    for index in range(1, int(sumo_cfg["signal_count"]) + 1):
        signal_position = index * spacing
        entry = next(
            (row for row in lead_rows if row["position_m"] >= signal_position - zone_length),
            None,
        )
        crossing = next(
            (row for row in lead_rows if row["position_m"] >= signal_position),
            None,
        )
        if entry is not None and crossing is not None:
            ax.axvspan(entry["time_s"], crossing["time_s"],
                       color="#f2cf5b", alpha=0.10, linewidth=0)
    ax.plot(lead_time, lead_speed, color="#3478a6", linewidth=1.2, label="Lead vehicle")
    ax.axhline(speed_limit_kph, color="#e45756", linestyle="--", linewidth=1.0,
               label=f"Intersection limit ({speed_limit_kph:g} km/h)")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(f"Lead vehicle speed profile (seed {seed})")
    ax.grid(True, alpha=0.25)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(color="#f2cf5b", alpha=0.35, label="100 m approach speed-limit zone"))
    ax.legend(handles=handles)
    fig.savefig(output_dir / "lead_speed_time.png", dpi=170)
    plt.close(fig)

    code = {"r": 0, "y": 1, "G": 2, "g": 2}
    matrix = np.asarray([
        [code[row[signal_id]] for row in state_rows]
        for signal_id in signal_ids
    ], dtype=np.int8)
    timeline = np.asarray([row["time_s"] for row in state_rows], dtype=float)
    if len(timeline) == 1:
        time_edges = np.asarray([timeline[0] - 0.5, timeline[0] + 0.5])
    else:
        time_edges = np.concatenate((
            [timeline[0] - 0.5], 0.5 * (timeline[:-1] + timeline[1:]),
            [timeline[-1] + 0.5],
        ))
    cmap = ListedColormap(["#e45756", "#f2cf5b", "#54a24b"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    ax.pcolormesh(time_edges, np.arange(len(signal_ids) + 1), matrix,
                  cmap=cmap, norm=norm, shading="flat")
    ax.set_yticks(np.arange(len(signal_ids)) + 0.5, signal_ids)
    ax.invert_yaxis()
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Traffic-light intersection")
    ax.set_title(f"Traffic-light states over time (seed {seed}; 40 s green / 3 s yellow / 47 s red)")
    ax.legend(handles=[
        Patch(color="#54a24b", label="Green"),
        Patch(color="#f2cf5b", label="Yellow"),
        Patch(color="#e45756", label="Red"),
    ], loc="upper right", ncol=3)
    ax.grid(axis="x", alpha=0.2)
    fig.savefig(output_dir / "traffic_signal_timeline.png", dpi=170)
    plt.close(fig)

    speeds = [row["speed_kph"] for row in lead_rows]
    signal_positions = [i * spacing for i in range(1, int(sumo_cfg["signal_count"]) + 1)]
    speed_zone_samples = [
        row["speed_kph"] for row in lead_rows
        if row["lane_id"].startswith(":")
        or any(signal_pos - zone_length <= row["position_m"] <= signal_pos
               for signal_pos in signal_positions)
    ]
    summary = {
        "scenario_seed": int(seed),
        "simulation_duration_s": float(lead_rows[-1]["time_s"]),
        "lead_terminal_stop_reached": stop_reached,
        "lead_traffic_lights_passed": len(passed_signals),
        "lead_max_speed_kph": float(max(speeds)),
        "lead_max_speed_in_intersection_zone_kph": (
            float(max(speed_zone_samples)) if speed_zone_samples else None
        ),
        "lead_intersection_zone_overspeed_samples": sum(
            speed > speed_limit_kph + 1e-3 for speed in speed_zone_samples
        ),
        "lead_stop_count": len(randomization["lead_stop_plan"]),
        "signal_cycle_s": cycle_s,
        "signal_green_s": green_s,
        "signal_yellow_s": yellow_s,
        "signal_red_s": int(sumo_cfg["signal_red_s"]),
        "intersection_speed_limit_kph": speed_limit_kph,
        "intersection_speed_zone_m": float(sumo_cfg["intersection_speed_zone_m"]),
        "signal_initial_conditions": randomization["signal_initial_conditions"],
        "lead_stop_plan": randomization["lead_stop_plan"],
    }
    with (output_dir / "environment_preview.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic SUMO scenario preview plots")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / "config" / "default.json")
    seed = int(cfg["seed"] if args.seed is None else args.seed)
    output = analyze(seed, root / args.output_root, root)
    print(f"Saved scenario preview to {output}")


if __name__ == "__main__":
    main()
