from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
import os
import json
from pathlib import Path
import random
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from algorithm.sac import SAC
from algorithm.td3 import TD3
from environment.fchev_env import FCHEVEnv, load_config
from util.observation import normalize_observation


TRACE_FIELDS = [
    "time_s", "reward", "soc", "battery_soh", "fcs_soh", "edu_soh",
    "core_temperature_c", "surface_temperature_c", "ego_speed_m_s",
    "ego_acceleration_m_s2", "lead_speed_m_s", "lead_acceleration_m_s2",
    "ego_x_m", "lead_x_m", "following_distance_m",
    "following_distance_reduction_m",
    "action_policy_acceleration_unit", "action_policy_fcs_power_unit",
    "action_requested_acceleration_raw_m_s2", "action_requested_fcs_power_raw_kw",
    "action_bounded_acceleration_m_s2", "action_bounded_fcs_power_kw",
    "action_powertrain_limited_acceleration_m_s2", "action_executed_acceleration_m_s2",
    "action_executed_fcs_power_kw", "action_resulting_battery_power_w",
    "requested_acceleration_m_s2", "requested_fcs_power_kw", "fcs_power_kw",
    "wheel_traction_force_n", "wheel_mechanical_power_w",
    "motor_speed_rpm", "motor_torque_nm", "edu_electrical_power_w",
    "edu_loss_power_w", "edu_efficiency", "friction_brake_force_n",
    "battery_power_w", "battery_current_pack_a", "battery_current_cell_a",
    "battery_terminal_voltage_pack_v", "hydrogen_consumed_g",
    "fcs_startup_degradation_percent", "fcs_low_power_degradation_percent",
    "fcs_high_power_degradation_percent", "fcs_load_change_degradation_percent",
    "dc_bus_dump_power_w", "collision_count", "collision_event", "gap_overlap",
    "red_light_violation", "speeding_violation", "speeding_vehicles",
    "ego_in_intersection_speed_zone", "lead_in_intersection_speed_zone",
    "intersection_speed_limit_m_s", "traffic_light_crossing",
    "traffic_light_crossed_id", "traffic_lights_passed",
    "lead_traffic_light_crossing", "lead_traffic_light_crossed_id",
    "lead_traffic_lights_passed", "traffic_light_state", "traffic_light_distance_m",
    "traffic_light_remaining_s",
    "current_speed_limit_m_s", "next_speed_limit_m_s",
    "distance_to_next_speed_region_m", "max_speed_action_violation",
    "gap_over_diagnostic_threshold", "gap_over_monitor_threshold",
    "stuck_steps", "stuck_duration_s", "leader_terminal_stop_reached", "abnormal_termination",
    "terminal_penalty", "terminated_reason", "safety_violations", "constraint_clipping",
    "physical_powertrain_clip", "no_reverse_clip", "speed_limit_clip",
    "regen_torque_clipped_for_limits", "fcs_power_clipped_for_battery_limits",
    "sumo_acceleration_intervention", "power_balance_residual_w",
    "battery_power_min_w", "battery_power_max_w",
    "reward_hydrogen_g", "reward_fcs_degradation_cumulative",
    "reward_fcs_degradation_increment",
    "reward_battery_soh_loss", "reward_edu_clr_cumulative",
    "reward_edu_clr_increment",
    "reward_comfort", "reward_following", "reward_soc_penalty",
    "reward_cost_following", "reward_cost_hydrogen_g",
    "reward_cost_fcs_degradation_percent", "reward_cost_battery_soh_loss_percent",
    "reward_cost_edu_degradation_percent", "reward_cost_comfort", "reward_cost_soc",
    "reward_beta_following", "reward_beta_hydrogen", "reward_beta_fcs",
    "reward_beta_battery", "reward_beta_edu", "reward_beta_comfort", "reward_beta_soc",
    "reward_contribution_following", "reward_contribution_hydrogen",
    "reward_contribution_fcs", "reward_contribution_battery",
    "reward_contribution_edu", "reward_contribution_comfort", "reward_contribution_soc",
    "reward_contribution_terminal_penalty",
]
REWARD_COMPONENTS = (
    "hydrogen", "fcs", "battery", "edu", "comfort", "following", "soc",
    "terminal_penalty",
)
REWARD_BETA_COMPONENTS = ("hydrogen", "fcs", "battery", "edu", "comfort", "following", "soc")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # The policy is a small MLP; allow fast CUDA kernels while retaining
        # seeded pseudorandom streams for reproducible runs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(1)


def map_action(action_unit, action_space):
    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    unit = np.asarray(action_unit, dtype=np.float32)
    physical = np.empty_like(unit)
    acceleration_unit = unit[..., 0]
    physical[..., 0] = np.where(
        acceleration_unit < 0.0,
        acceleration_unit * abs(float(low[0])),
        acceleration_unit * float(high[0]),
    )
    physical[..., 1] = low[1] + (unit[..., 1] + 1.0) * 0.5 * (high[1] - low[1])
    return physical


def make_agent(name, obs_dim, action_dim, config, seed):
    if name.lower() == "sac":
        return SAC(obs_dim, action_dim, config, seed)
    if name.lower() == "td3":
        return TD3(obs_dim, action_dim, config, seed)
    raise ValueError(f"Unknown algorithm {name!r}; choose sac or td3")


def _write_csv(path: Path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows, fields):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _plot_episode_rewards(episodes_csv: Path, output_png: Path, window: int = 10):
    """Save raw episode returns and a trailing moving average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with episodes_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=np.int32)
    rewards = np.asarray([float(row["return"]) for row in rows], dtype=np.float64)
    display_floor = -200.0
    window = max(1, min(int(window), len(rewards)))
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    # Values below the fixed visible range remain intact in episodes.csv but
    # are drawn at the lower boundary so they cannot distort the autoscaled top.
    ax.plot(episodes, np.maximum(rewards, display_floor), color="#4C78A8", alpha=0.45, linewidth=1.0,
            label="Episode reward")
    ax.plot(episodes[window - 1:], np.maximum(moving, display_floor), color="#E45756", linewidth=2.0,
            label=f"{window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (sum of per-step rewards)")
    ax.set_title("SAC training reward")
    ax.grid(True, alpha=0.25)
    # Keep the requested lower limit fixed; leave the upper limit data-driven.
    ax.set_ylim(bottom=display_floor)
    ax.text(0.99, 0.02, "Values below -200 are clipped; see episodes.csv for raw returns",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    ax.legend()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _read_episode_rows(episodes_csv: Path):
    with episodes_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _plot_episode_metrics(episodes_csv: Path, output_png: Path):
    """Plot episode hydrogen use and terminal SOC/SOH health indicators."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_episode_rows(episodes_csv)
    if not rows:
        return
    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=np.int32)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for ax, field, ylabel, title in (
        (axes[0, 0], "hydrogen_g", "Hydrogen consumption (g)", "Episode hydrogen consumption"),
        (axes[0, 1], "final_soc", "SOC", "Final battery SOC"),
    ):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        ax.plot(episodes, values, color="#4C78A8", linewidth=1.2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    soh_ax = axes[1, 0]
    for field, label, color in (
        ("final_battery_soh", "Battery SOH", "#4C78A8"),
        ("final_fcs_soh", "FCS SOH", "#F58518"),
        ("final_edu_soh", "EDU SOH", "#54A24B"),
    ):
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        soh_ax.plot(episodes, values, label=label, color=color, linewidth=1.2)
    soh_ax.set_xlabel("Episode")
    soh_ax.set_ylabel("SOH")
    soh_ax.set_title("Final component SOH")
    soh_ax.grid(True, alpha=0.25)
    soh_ax.legend()

    ax = axes[1, 1]
    lights = np.asarray(
        [int(row.get("traffic_lights_passed", 0) or 0) for row in rows],
        dtype=np.int32,
    )
    ax.plot(episodes, lights, color="#54A24B", linewidth=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Traffic lights passed")
    ax.set_title("Controlled vehicle intersections passed")
    ax.grid(True, alpha=0.25)

    ax = axes[2, 0]
    steps = np.asarray([int(row["steps"]) for row in rows], dtype=np.int32)
    ax.plot(episodes, steps, color="#B279A2", linewidth=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode duration")
    ax.grid(True, alpha=0.25)
    ax = axes[2, 1]
    lead_lights = np.asarray(
        [int(row.get("lead_traffic_lights_passed", 0) or 0) for row in rows],
        dtype=np.int32,
    )
    ax.plot(episodes, lead_lights, color="#E45756", linewidth=1.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Traffic lights passed")
    ax.set_title("Lead vehicle intersections passed")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _plot_reward_components(episodes_csv: Path, output_png: Path, window: int = 10):
    """Plot each component's signed cumulative contribution to episode return."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_episode_rows(episodes_csv)
    if not rows:
        return
    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=np.int32)
    window = max(1, min(int(window), len(rows)))
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = (
        "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2",
        "#FF9DA6", "#9D755D", "#BAB0AC", "#ECA82C",
    )
    for name, color in zip(REWARD_COMPONENTS, colors):
        values = np.asarray([float(row[f"reward_component_{name}"]) for row in rows], dtype=np.float64)
        moving = np.convolve(values, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], moving, label=name, color=color, linewidth=1.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Signed cumulative reward contribution")
    ax.set_title(f"Reward component contributions ({window}-episode moving average)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def evaluate(env, agent, cfg, episodes, base_seed, vehicle_cfg, max_steps=None):
    rows = []
    for i in range(episodes):
        obs, reset_info = env.reset(seed=base_seed + i)
        total = 0.0
        hydrogen = 0.0
        steps = 0
        collisions = 0
        collision_events = 0
        red_light_violations = 0
        speeding_violations = 0
        traffic_lights_passed = 0
        lead_traffic_lights_passed = 0
        terminal_penalty = 0.0
        min_gap = float("inf")
        termination_reason = None
        truncation_reason = None
        while True:
            norm_obs = normalize_observation(obs, vehicle_cfg)
            action_unit = agent.select_action(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(map_action(action_unit, env.action_space))
            total += reward
            hydrogen += info.get("hydrogen_consumed_g", 0.0)
            collisions += info.get("collision_count", 0)
            collision_events += int(info.get("collision_event", False))
            red_light_violations += int(info.get("red_light_violation", False))
            speeding_violations += int(info.get("speeding_violation", False))
            traffic_lights_passed = int(info.get("traffic_lights_passed", 0))
            lead_traffic_lights_passed = int(info.get("lead_traffic_lights_passed", 0))
            terminal_penalty += info.get("terminal_penalty", 0.0)
            min_gap = min(min_gap, info.get("following_distance_m", float("inf")))
            termination_reason = info.get("terminated_reason")
            truncation_reason = info.get("truncated_reason")
            steps += 1
            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break
        rows.append({
            "episode": i + 1, "seed": base_seed + i, "return": total,
            "penalty_magnitude": -total,
            "steps": steps, "hydrogen_g": hydrogen, "collisions": collisions,
            "collision_events": collision_events,
            "red_light_violations": red_light_violations,
            "speeding_violations": speeding_violations,
            "traffic_lights_passed": traffic_lights_passed,
            "lead_traffic_lights_passed": lead_traffic_lights_passed,
            "terminal_penalty": terminal_penalty,
            "minimum_gap_m": min_gap,
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
            "scenario_seed": reset_info.get("scenario_seed"),
            "final_soc": env.battery.state.soc,
            "final_battery_soh": env.battery.state.soh,
            "final_fcs_soh": 1.0 - env.fcs_aging.cumulative,
            "final_edu_soh": 1.0 - env.edu_aging.cumulative,
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train SAC or TD3 on the SUMO FCHEV environment")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--algorithm", choices=["sac", "td3"], default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode smoke-test cap")
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                        help="Compute device; CUDA also stores replay data on GPU")
    parser.add_argument("--evaluation-episodes", type=int, default=None,
                        help="Override periodic evaluation episode count (0 disables evaluations)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    algorithm_name = (args.algorithm or cfg["training"]["algorithm"]).lower()
    tr = dict(cfg["training"])
    episodes = int(args.episodes if args.episodes is not None else tr["episodes"])
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    learning_starts = int(args.learning_starts if args.learning_starts is not None else tr["learning_starts"])
    batch_size = int(args.batch_size if args.batch_size is not None else tr["batch_size"])
    if args.device is not None:
        tr["device"] = args.device
    requested_device = tr.get("device", "auto")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment")
    if args.evaluation_episodes is not None:
        tr["evaluation_episodes"] = max(0, int(args.evaluation_episodes))
    tr["episodes"] = episodes
    tr["learning_starts"] = learning_starts
    tr["batch_size"] = batch_size
    cfg["training"].update(tr)
    if args.gui:
        cfg["sumo"]["gui"] = True
    if args.max_steps is not None:
        cfg["sumo"]["episode_limit_s"] = int(args.max_steps)
    seed_everything(seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (root / args.output_root / f"{algorithm_name}_{stamp}_seed{seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    env = FCHEVEnv(project_root=root, config=cfg, render_mode="human" if args.gui else None)
    agent = make_agent(algorithm_name, env.observation_space.shape[0], env.action_space.shape[0], tr, seed)
    if agent.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agent.device)
        device_label = f"{torch.cuda.get_device_name(agent.device)} (CUDA)"
    else:
        device_label = str(agent.device).upper()
    replay_mb = agent.replay.data.numel() * agent.replay.data.element_size() / (1024 ** 2)
    print(
        f"Training on {device_label}; replay buffer on {agent.replay.device} "
        f"({replay_mb:.1f} MiB); SUMO/environment remains CPU-driven."
    )
    global_step = 0
    best_return = -float("inf")
    start_time = time.perf_counter()
    try:
        for episode in range(1, episodes + 1):
            # Keep the exogenous SUMO scenario identical across episodes.
            # Policy exploration and network updates remain stochastic, but
            # signals and the lead-vehicle stop plan use the same scene seed.
            obs, reset_info = env.reset(seed=seed)
            state = normalize_observation(obs, cfg["vehicle"])
            total_return = 0.0
            total_hydrogen = 0.0
            reward_component_totals = {name: 0.0 for name in REWARD_COMPONENTS}
            total_safety_violations = 0
            total_clips = 0
            collisions = 0
            collision_events = 0
            red_light_violations = 0
            speeding_violations = 0
            traffic_lights_passed = 0
            lead_traffic_lights_passed = 0
            terminal_penalty_total = 0.0
            min_gap = float("inf")
            trace = []
            updates = []
            local_step = 0
            while True:
                if global_step < learning_starts:
                    action_unit = np.random.uniform(-1.0, 1.0, size=env.action_space.shape[0]).astype(np.float32)
                else:
                    action_unit = agent.select_action(state, deterministic=False)
                    if algorithm_name == "td3":
                        noise = np.random.normal(0.0, tr["exploration_noise"], size=action_unit.shape)
                        action_unit = np.clip(action_unit + noise, -1.0, 1.0)
                physical_action = map_action(action_unit, env.action_space)
                next_obs, reward, terminated, truncated, info = env.step(physical_action)
                next_state = normalize_observation(next_obs, cfg["vehicle"])
                done = terminated or truncated
                agent.replay.add(state, action_unit, reward, next_state, done)
                state = next_state
                total_return += reward
                total_hydrogen += info.get("hydrogen_consumed_g", 0.0)
                total_safety_violations += info.get("safety_violations", 0)
                total_clips += int(info.get("constraint_clipping", False))
                collisions += info.get("collision_count", 0)
                collision_events += int(info.get("collision_event", False))
                red_light_violations += int(info.get("red_light_violation", False))
                speeding_violations += int(info.get("speeding_violation", False))
                traffic_lights_passed = int(info.get("traffic_lights_passed", 0))
                lead_traffic_lights_passed = int(info.get("lead_traffic_lights_passed", 0))
                terminal_penalty_total += info.get("terminal_penalty", 0.0)
                min_gap = min(min_gap, info.get("following_distance_m", float("inf")))
                local_step += 1
                global_step += 1
                reward_parts = info.get("reward_components", {})
                for name in REWARD_COMPONENTS:
                    reward_component_totals[name] += float(
                        reward_parts.get(f"reward_contribution_{name}", 0.0)
                    )
                trace.append({
                    "time_s": info.get("sumo_time_s"), "reward": reward,
                    "soc": info.get("battery_soc"), "battery_soh": info.get("battery_soh"),
                    "fcs_soh": info.get("fcs_soh"), "edu_soh": info.get("edu_soh"),
                    "core_temperature_c": info.get("battery_core_temperature_c"),
                    "surface_temperature_c": info.get("battery_surface_temperature_c"),
                    "ego_speed_m_s": info.get("ego_speed_m_s"),
                    "ego_acceleration_m_s2": info.get("executed_acceleration_m_s2"),
                    "lead_speed_m_s": info.get("lead_speed_m_s"),
                    "lead_acceleration_m_s2": info.get("lead_acceleration_m_s2"),
                    "ego_x_m": info.get("ego_x_m"), "lead_x_m": info.get("lead_x_m"),
                    "following_distance_m": info.get("following_distance_m"),
                    "following_distance_reduction_m": info.get("following_distance_reduction_m"),
                    "action_policy_acceleration_unit": float(action_unit[0]),
                    "action_policy_fcs_power_unit": float(action_unit[1]),
                    "action_requested_acceleration_raw_m_s2": info.get("requested_acceleration_raw_m_s2"),
                    "action_requested_fcs_power_raw_kw": info.get("requested_fcs_power_raw_kw"),
                    "action_bounded_acceleration_m_s2": info.get("requested_acceleration_m_s2"),
                    "action_bounded_fcs_power_kw": info.get("fcs_power_requested_kw"),
                    "action_powertrain_limited_acceleration_m_s2": info.get("powertrain_limited_acceleration_m_s2"),
                    "action_executed_acceleration_m_s2": info.get("executed_acceleration_m_s2"),
                    "action_executed_fcs_power_kw": info.get("fcs_power_executed_kw"),
                    "action_resulting_battery_power_w": info.get("battery_power_w"),
                    "dc_bus_dump_power_w": info.get("dc_bus_dump_power_w"),
                    "requested_acceleration_m_s2": info.get("requested_acceleration_m_s2"),
                    "requested_fcs_power_kw": info.get("fcs_power_requested_kw"),
                    "fcs_power_kw": info.get("fcs_power_executed_kw"),
                    "wheel_traction_force_n": info.get("wheel_traction_force_n"),
                    "wheel_mechanical_power_w": info.get("wheel_mechanical_power_w"),
                    "motor_speed_rpm": info.get("motor_speed_rpm"),
                    "motor_torque_nm": info.get("motor_torque_nm"),
                    "edu_efficiency": info.get("edu_efficiency"),
                    "friction_brake_force_n": info.get("friction_brake_force_n"),
                    "battery_power_w": info.get("battery_power_w"),
                    "battery_current_pack_a": info.get("battery_current_pack_a"),
                    "battery_current_cell_a": info.get("battery_current_cell_a"),
                    "battery_terminal_voltage_pack_v": info.get("battery_terminal_voltage_pack_v"),
                    "edu_electrical_power_w": info.get("edu_electrical_power_w"),
                    "edu_loss_power_w": info.get("edu_loss_power_w"),
                    "hydrogen_consumed_g": info.get("hydrogen_consumed_g"),
                    "fcs_startup_degradation_percent": info.get("fcs_startup_degradation_percent"),
                    "fcs_low_power_degradation_percent": info.get("fcs_low_power_degradation_percent"),
                    "fcs_high_power_degradation_percent": info.get("fcs_high_power_degradation_percent"),
                    "fcs_load_change_degradation_percent": info.get("fcs_load_change_degradation_percent"),
                    "collision_count": info.get("collision_count"),
                    "collision_event": info.get("collision_event"),
                    "red_light_violation": info.get("red_light_violation"),
                    "speeding_violation": info.get("speeding_violation"),
                    "speeding_vehicles": ";".join(info.get("speeding_vehicles", [])),
                    "ego_in_intersection_speed_zone": info.get("ego_in_intersection_speed_zone"),
                    "lead_in_intersection_speed_zone": info.get("lead_in_intersection_speed_zone"),
                    "intersection_speed_limit_m_s": info.get("intersection_speed_limit_m_s"),
                    "traffic_light_crossing": info.get("traffic_light_crossing"),
                    "traffic_light_crossed_id": info.get("traffic_light_crossed_id"),
                    "traffic_lights_passed": info.get("traffic_lights_passed"),
                    "lead_traffic_light_crossing": info.get("lead_traffic_light_crossing"),
                    "lead_traffic_light_crossed_id": info.get("lead_traffic_light_crossed_id"),
                    "lead_traffic_lights_passed": info.get("lead_traffic_lights_passed"),
                    "traffic_light_state": info.get("traffic_light_state_observation"),
                    "traffic_light_distance_m": info.get("traffic_light_distance_m"),
                    "traffic_light_remaining_s": info.get("traffic_light_remaining_s"),
                    "current_speed_limit_m_s": info.get("current_speed_limit_m_s"),
                    "next_speed_limit_m_s": info.get("next_speed_limit_m_s"),
                    "distance_to_next_speed_region_m": info.get("distance_to_next_speed_region_m"),
                    "max_speed_action_violation": info.get("max_speed_action_violation"),
                    "gap_overlap": info.get("gap_overlap"),
                    "gap_over_diagnostic_threshold": info.get("gap_over_diagnostic_threshold"),
                    "gap_over_monitor_threshold": info.get("gap_over_monitor_threshold"),
                    "stuck_steps": info.get("stuck_steps"),
                    "stuck_duration_s": info.get("stuck_duration_s"),
                    "leader_terminal_stop_reached": info.get("leader_terminal_stop_reached"),
                    "abnormal_termination": info.get("abnormal_termination"),
                    "terminal_penalty": info.get("terminal_penalty"),
                    "terminated_reason": info.get("terminated_reason"),
                    "safety_violations": info.get("safety_violations"),
                    "constraint_clipping": int(info.get("constraint_clipping", False)),
                    "physical_powertrain_clip": info.get("physical_powertrain_clip"),
                    "no_reverse_clip": info.get("no_reverse_clip"),
                    "speed_limit_clip": info.get("speed_limit_clip"),
                    "regen_torque_clipped_for_limits": info.get("regen_torque_clipped_for_limits"),
                    "fcs_power_clipped_for_battery_limits": info.get("fcs_power_clipped_for_battery_limits"),
                    "sumo_acceleration_intervention": info.get("sumo_acceleration_intervention"),
                    "power_balance_residual_w": info.get("power_balance_residual_w"),
                    "battery_power_min_w": info.get("battery_power_min_w"),
                    "battery_power_max_w": info.get("battery_power_max_w"),
                    "reward_hydrogen_g": reward_parts.get("hydrogen_g"),
                    "reward_fcs_degradation_cumulative": reward_parts.get("fcs_degradation_cumulative"),
                    "reward_fcs_degradation_increment": reward_parts.get("fcs_degradation_increment"),
                    "reward_battery_soh_loss": reward_parts.get("battery_soh_loss"),
                    "reward_edu_clr_cumulative": reward_parts.get("edu_clr_cumulative"),
                    "reward_edu_clr_increment": reward_parts.get("edu_clr_increment"),
                    "reward_comfort": reward_parts.get("comfort"),
                    "reward_following": reward_parts.get("following"),
                    "reward_soc_penalty": reward_parts.get("soc_penalty"),
                    "reward_cost_following": reward_parts.get("cost_following"),
                    "reward_cost_hydrogen_g": reward_parts.get("cost_hydrogen"),
                    "reward_cost_fcs_degradation_percent": reward_parts.get("cost_fcs"),
                    "reward_cost_battery_soh_loss_percent": reward_parts.get("cost_battery"),
                    "reward_cost_edu_degradation_percent": reward_parts.get("cost_edu"),
                    "reward_cost_comfort": reward_parts.get("cost_comfort"),
                    "reward_cost_soc": reward_parts.get("cost_soc"),
                    **{f"reward_beta_{name}": reward_parts.get(f"beta_{name}")
                       for name in REWARD_BETA_COMPONENTS},
                    **{f"reward_contribution_{name}": reward_parts.get(f"reward_contribution_{name}")
                       for name in REWARD_COMPONENTS},
                })
                if len(agent.replay) >= batch_size and global_step >= learning_starts:
                    for _ in range(int(tr.get("updates_per_step", 1))):
                        losses = agent.update(batch_size)
                        losses["global_step"] = global_step
                        updates.append(losses)
                if terminated or truncated or (args.max_steps is not None and local_step >= args.max_steps):
                    break
            _write_csv(run_dir / f"episode_{episode:04d}_trajectory.csv", trace, TRACE_FIELDS)
            row = {
                "episode": episode, "seed": seed,
                "return": total_return, "penalty_magnitude": -total_return,
                "steps": local_step,
                "hydrogen_g": total_hydrogen, "collisions": collisions,
                "collision_events": collision_events,
                "red_light_violations": red_light_violations,
                "speeding_violations": speeding_violations,
                "traffic_lights_passed": traffic_lights_passed,
                "lead_traffic_lights_passed": lead_traffic_lights_passed,
                "terminal_penalty": terminal_penalty_total,
                "safety_violations": total_safety_violations,
                "constraint_clipping_steps": total_clips,
                "minimum_gap_m": min_gap,
                "final_soc": env.battery.state.soc,
                "final_battery_soh": env.battery.state.soh,
                "final_fcs_soh": 1.0 - env.fcs_aging.cumulative,
                "final_edu_soh": 1.0 - env.edu_aging.cumulative,
                "final_core_temperature_c": env.battery.state.core_temperature_c,
                "final_surface_temperature_c": env.battery.state.surface_temperature_c,
                "termination_reason": info.get("terminated_reason"),
                "truncation_reason": info.get("truncated_reason"),
                "scenario_seed": reset_info.get("scenario_seed"),
                "signal_initial_conditions_json": json.dumps(
                    reset_info.get("signal_initial_conditions", []), separators=(",", ":")
                ),
                "lead_stop_plan_json": json.dumps(
                    reset_info.get("lead_stop_plan", []), separators=(",", ":")
                ),
            }
            row.update({f"reward_component_{name}": value for name, value in reward_component_totals.items()})
            row["reward_component_total"] = sum(reward_component_totals.values())
            _append_csv(run_dir / "episodes.csv", [row], list(row))
            loss_fields = ["global_step", "critic_loss", "actor_loss", "alpha_loss", "alpha"]
            if updates:
                _append_csv(run_dir / "losses.csv", updates, loss_fields)
            if total_return > best_return:
                best_return = total_return
                agent.save(run_dir / "best.pt")
            if episode % int(tr.get("save_interval_episodes", 10)) == 0 or episode == episodes:
                agent.save(run_dir / "final.pt")
            print(
                f"{algorithm_name.upper()} episode {episode}/{episodes}: "
                f"return={total_return:.3f}, H2={total_hydrogen:.3f} g, "
                f"steps={local_step}, clips={total_clips}"
            )
            if episode % int(tr.get("evaluation_interval_episodes", 10)) == 0 and tr.get("evaluation_episodes", 0) > 0:
                eval_rows = evaluate(
                    env, agent, cfg, int(tr["evaluation_episodes"]),
                    seed + 100000 + episode * 100, cfg["vehicle"],
                    max_steps=args.max_steps,
                )
                for r in eval_rows:
                    r["training_episode"] = episode
                eval_path = run_dir / "evaluations.csv"
                _append_csv(eval_path, eval_rows, list(eval_rows[0]) if eval_rows else [])
            plot_interval = max(1, int(tr.get("plot_interval_episodes", 10)))
            if episode % plot_interval == 0 or episode == episodes:
                _plot_episode_rewards(run_dir / "episodes.csv", run_dir / "episode_reward.png")
                _plot_episode_metrics(run_dir / "episodes.csv", run_dir / "episode_metrics.png")
                _plot_reward_components(run_dir / "episodes.csv", run_dir / "reward_components.png")
                print(f"Updated reward and metrics plots through episode {episode}.")
    finally:
        env.close()
    if agent.device.type == "cuda":
        torch.cuda.synchronize(agent.device)
    elapsed = time.perf_counter() - start_time
    reward_plot = run_dir / "episode_reward.png"
    _plot_episode_rewards(run_dir / "episodes.csv", reward_plot)
    metrics_plot = run_dir / "episode_metrics.png"
    _plot_episode_metrics(run_dir / "episodes.csv", metrics_plot)
    components_plot = run_dir / "reward_components.png"
    _plot_reward_components(run_dir / "episodes.csv", components_plot)
    episode_rows = _read_episode_rows(run_dir / "episodes.csv")
    termination_counts = Counter(row.get("termination_reason") or "none" for row in episode_rows)
    returns = np.asarray([float(row["return"]) for row in episode_rows], dtype=np.float64)
    steps = np.asarray([int(row["steps"]) for row in episode_rows], dtype=np.int64)
    passed = np.asarray(
        [int(row.get("traffic_lights_passed", 0) or 0) for row in episode_rows],
        dtype=np.int64,
    )
    summary = {
        "algorithm": algorithm_name, "seed": seed, "episodes": episodes,
        "environment_steps": global_step, "elapsed_seconds": elapsed,
        "env_steps_per_second_including_training": global_step / max(elapsed, 1e-9),
        "compute_device": str(agent.device),
        "replay_device": str(agent.replay.device),
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(agent.device) / (1024 ** 2)
            if agent.device.type == "cuda" else 0.0
        ),
        "run_directory": str(run_dir),
        "episode_reward_plot": str(reward_plot),
        "episode_metrics_plot": str(metrics_plot),
        "reward_components_plot": str(components_plot),
        "termination_reason_counts": dict(termination_counts),
        "mean_episode_return": float(np.mean(returns)) if returns.size else None,
        "median_episode_steps": float(np.median(steps)) if steps.size else None,
        "mean_traffic_lights_passed": float(np.mean(passed)) if passed.size else None,
        "episodes_with_collision": int(termination_counts.get("collision", 0)),
        "episodes_with_red_light_violation": int(
            termination_counts.get("red_light_violation", 0)
        ),
        "episodes_with_speeding": int(termination_counts.get("speeding", 0)),
        "episodes_reaching_lead_terminal_stop": int(
            termination_counts.get("lead_terminal_stop", 0)
        ),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved episode reward curve to {reward_plot}")
    print(f"Saved episode state metrics to {metrics_plot}")
    print(f"Saved reward component curves to {components_plot}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
