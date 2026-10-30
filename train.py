from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from importlib import import_module
import os
import json
from pathlib import Path
import re
import random
import time

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from algorithm.sac import SAC
from algorithm.td3 import TD3
from algorithm.hasac import HASAC
from algorithm.maddpg import MADDPG
from algorithm.mappo import MAPPO
from algorithm.two_agent_sac import TwoAgentSAC
from environment.fchev_env import FCHEVEnv, load_config
from util.observation import normalize_observation


MADDPGRuleAdapter = import_module(
    "llm-prior.maddpg_contract"
).MADDPGRuleAdapter
MADDPG_RULE_MODES = ("none", "physical_action", "normalized_action", "pre_tanh")


TRACE_FIELDS = [
    "time_s", "reward", "driving_agent_reward", "energy_agent_reward",
    # Raw pre-action observation and the exact Gaussian prior used at this
    # step. These columns make stale full_observation bugs visible in traces.
    "policy_obs_soc", "policy_obs_battery_soh", "policy_obs_fcs_soh",
    "policy_obs_edu_soh", "policy_obs_ego_acceleration_m_s2",
    "policy_obs_ego_speed_m_s", "policy_obs_lead_acceleration_m_s2",
    "policy_obs_lead_speed_m_s", "policy_obs_following_distance_m",
    "policy_obs_next_signal_state", "policy_obs_next_signal_distance_m",
    "policy_obs_signal_phase_remaining_s", "policy_obs_current_speed_limit_m_s",
    "policy_obs_next_speed_limit_m_s", "policy_obs_distance_to_next_speed_region_m",
    "policy_input_soc", "policy_input_battery_soh_loss_x1000",
    "policy_input_fcs_soh_loss_x1000", "policy_input_edu_soh_loss_x1000",
    "policy_input_ego_acceleration_div1p5", "policy_input_ego_speed_div20",
    "policy_input_lead_acceleration_div1p5", "policy_input_lead_speed_div20",
    "policy_input_following_distance_div1000", "policy_input_traffic_light_state",
    "policy_input_next_signal_distance_div1000",
    "policy_input_signal_phase_remaining_div100",
    "policy_input_current_speed_limit_div20", "policy_input_next_speed_limit_div20",
    "policy_input_distance_to_next_speed_region_div1000",
    "policy_driving_prior_mean", "policy_driving_prior_std", "policy_driving_prior_active",
    "policy_driving_prior_lambda_0", "policy_driving_prior_lambda_1",
    "policy_energy_prior_mean", "policy_energy_prior_std", "policy_energy_prior_active",
    "policy_energy_prior_lambda_0", "policy_energy_prior_lambda_1",
    "policy_maddpg_rule_mode", "policy_maddpg_rule_lambda",
    "policy_maddpg_driving_actor_unit", "policy_maddpg_energy_actor_unit",
    "policy_maddpg_driving_rule_value", "policy_maddpg_energy_rule_value",
    "policy_maddpg_driving_rule_active", "policy_maddpg_energy_rule_active",
    "policy_maddpg_driving_fused_unit", "policy_maddpg_energy_fused_unit",
    "energy_agent_observation_ego_acceleration_div1p5",
    "energy_agent_observation_ego_speed_div20",
    "energy_agent_observation_soc",
    "energy_agent_observation_battery_soh_loss_x1000",
    "energy_agent_observation_fcs_soh_loss_x1000",
    "energy_agent_observation_edu_soh_loss_x1000",
    "soc", "battery_soh", "fcs_soh", "edu_soh",
    "core_temperature_c", "surface_temperature_c", "ego_speed_m_s",
    "ego_acceleration_m_s2", "lead_speed_m_s", "lead_acceleration_m_s2",
    "ego_x_m", "lead_x_m", "following_distance_m", "following_distance_valid",
    "following_distance_reduction_m", "following_forward_distance_m",
    "following_region_max_speed_m_s",
    "action_policy_acceleration_unit", "action_policy_fcs_power_unit",
    "action_policy_gaussian_acceleration_raw",
    "action_policy_gaussian_fcs_power_raw",
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
    "speeding_termination_violation", "speeding_termination_threshold_m_s",
    "ego_in_intersection_speed_zone", "lead_in_intersection_speed_zone",
    "intersection_speed_limit_m_s", "active_speed_limit_m_s",
    "speeding_delta_v_m_s", "speeding_penalty_beta", "speeding_penalty",
    "traffic_light_crossing",
    "green_light_crossing", "green_light_crossing_bonus",
    "traffic_light_crossed_id", "traffic_lights_passed",
    "lead_traffic_light_crossing", "lead_traffic_light_crossed_id",
    "lead_traffic_lights_passed", "traffic_light_state", "traffic_light_distance_m",
    "traffic_light_remaining_s", "time_to_traffic_light_s",
    "traffic_light_state_before_crossing_step",
    "traffic_light_state_after_crossing_step",
    "traffic_light_state_at_crossing", "traffic_light_crossing_elapsed_s",
    "traffic_light_reward_active", "traffic_light_reward",
    "traffic_light_reward_contribution",
    "speed_limit_entry_active", "speed_limit_entry_error_before",
    "speed_limit_entry_error_after",
    "speed_limit_entry_predicted_speed_before_m_s",
    "speed_limit_entry_predicted_speed_after_m_s",
    "speed_limit_entry_target_m_s", "speed_limit_entry_distance_m",
    "speed_limit_entry_reward", "speed_limit_entry_beta",
    "current_speed_limit_m_s", "current_speed_limit_at_action_m_s",
    "next_speed_limit_m_s",
    "distance_to_next_speed_region_m", "max_speed_action_violation",
    "max_speed_action_clipped", "max_speed_limited_acceleration_m_s2",
    "max_speed_clip_delta_m_s2",
    "gap_over_diagnostic_threshold", "gap_over_monitor_threshold",
    "stuck_steps", "stuck_duration_s", "leader_terminal_stop_reached", "abnormal_termination",
    "terminal_penalty", "terminated_reason", "safety_violations", "constraint_clipping",
    "physical_powertrain_clip", "no_reverse_clip", "speed_limit_clip",
    "zero_speed_negative_accel_action", "speed_limit_positive_accel_action",
    "regen_torque_clipped_for_limits", "fcs_power_clipped_for_battery_limits",
    "sumo_acceleration_intervention", "power_balance_residual_w",
    "battery_power_min_w", "battery_power_max_w",
    "reward_hydrogen_g", "reward_fcs_degradation_cumulative",
    "reward_fcs_degradation_increment",
    "reward_battery_soh_loss", "reward_edu_clr_cumulative",
    "reward_edu_clr_increment",
    "reward_comfort", "reward_following", "reward_speeding", "reward_soc_penalty",
    "reward_cost_following", "reward_cost_hydrogen_g",
    "reward_cost_fcs_degradation_percent", "reward_cost_battery_soh_loss_percent",
    "reward_cost_edu_degradation_percent", "reward_cost_comfort", "reward_cost_soc",
    "reward_beta_following", "reward_beta_hydrogen", "reward_beta_fcs",
    "reward_beta_battery", "reward_beta_edu", "reward_beta_comfort", "reward_beta_soc",
    "reward_beta_traffic_light", "reward_beta_speeding",
    "reward_beta_speed_limit_entry",
    "reward_contribution_following", "reward_contribution_hydrogen",
    "reward_contribution_fcs", "reward_contribution_battery",
    "reward_contribution_edu", "reward_contribution_comfort", "reward_contribution_soc",
    "reward_contribution_traffic_light", "reward_contribution_speeding",
    "reward_contribution_green_light_crossing_bonus",
    "reward_contribution_speed_limit_entry",
    "reward_contribution_terminal_penalty",
    "zero_speed_negative_accel_penalty",
    "reward_contribution_zero_speed_negative_accel_penalty",
    "speed_limit_positive_accel_penalty",
    "reward_contribution_speed_limit_positive_accel_penalty",
]
REWARD_COMPONENTS = (
    "hydrogen", "fcs", "battery", "edu", "comfort", "following", "soc",
    "traffic_light", "speeding", "terminal_penalty",
    "green_light_crossing_bonus",
    "zero_speed_negative_accel_penalty", "speed_limit_positive_accel_penalty",
    "speed_limit_entry",
)
REWARD_BETA_COMPONENTS = (
    "hydrogen", "fcs", "battery", "edu", "comfort", "following", "soc",
    "traffic_light", "speeding", "speed_limit_entry",
)
DRIVING_OBSERVATION_INDICES = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
ENERGY_OBSERVATION_INDICES = (4, 5, 0, 1, 2, 3)


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


def map_split_actions(driving_action_unit, energy_action_unit, action_space):
    """Map the two scalar policy actions to their physical action channels."""
    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    acceleration_unit = float(np.asarray(driving_action_unit).reshape(-1)[0])
    power_unit = float(np.asarray(energy_action_unit).reshape(-1)[0])
    acceleration = (
        acceleration_unit * abs(float(low[0]))
        if acceleration_unit < 0.0
        else acceleration_unit * float(high[0])
    )
    power = float(low[1]) + (power_unit + 1.0) * 0.5 * float(high[1] - low[1])
    return np.asarray([acceleration, power], dtype=np.float32)


def split_agent_observations(observation, vehicle_cfg, normalization_enabled=True):
    """Build driving and energy views of the shared observation.

    Energy order: measured ego acceleration, ego speed, SOC, battery SOH,
    FCS SOH, and EDU SOH.
    """
    state = normalize_observation(
        observation, vehicle_cfg, enabled=normalization_enabled
    )
    driving_state = state[np.asarray(DRIVING_OBSERVATION_INDICES, dtype=np.int64)].copy()
    # The energy agent observes the measured acceleration present at this
    # decision point, not the driving policy's unexecuted request.
    energy_state = state[np.asarray(ENERGY_OBSERVATION_INDICES, dtype=np.int64)].copy()
    return driving_state, energy_state


def split_agent_rewards(info):
    """Partition per-step reward contributions while preserving their sum."""
    rewards = info.get("reward_components", {})

    def contribution(name):
        return float(rewards.get(f"reward_contribution_{name}", 0.0) or 0.0)

    driving_reward = sum(contribution(name) for name in (
        "following", "comfort", "traffic_light", "speeding",
        "green_light_crossing_bonus",
        "zero_speed_negative_accel_penalty",
        "speed_limit_positive_accel_penalty", "speed_limit_entry",
        "terminal_penalty"
    ))
    energy_reward = sum(contribution(name) for name in (
        "hydrogen", "fcs", "battery", "edu", "soc"
    ))
    return driving_reward, energy_reward


def is_dual_policy_agent(agent):
    return isinstance(agent, (TwoAgentSAC, HASAC, MAPPO, MADDPG))


def _apply_prior_lambda_schedule(agent, schedule, episode):
    """Apply an optional episode-indexed per-agent Gaussian-prior schedule."""
    if not isinstance(agent, (HASAC, MAPPO)) or not schedule:
        return None
    if not isinstance(schedule, list):
        raise TypeError("training.prior_lambda_schedule must be a list")
    for phase in schedule:
        start = int(phase.get("episode_start", 1))
        end = int(phase.get("episode_end", start))
        if start <= episode <= end:
            lambda_by_agent = phase.get("lambda_by_agent")
            if not isinstance(lambda_by_agent, dict):
                raise TypeError(
                    "Each prior lambda schedule phase must define lambda_by_agent"
                )
            for name in ("driving", "energy"):
                values = lambda_by_agent.get(name)
                if not isinstance(values, dict):
                    raise TypeError(
                        f"prior lambda schedule phase must define {name} lambdas"
                    )
                for key in ("lambda_0", "lambda_1"):
                    value = float(values[key])
                    if not 0.0 <= value <= 1.0:
                        raise ValueError(
                            f"{name}.{key} in prior lambda schedule must be in [0, 1]"
                        )
            agent.prior_cfg["lambda_by_agent"] = lambda_by_agent
            return phase
    return None


def make_agent(name, obs_dim, action_dim, config, seed):
    if name.lower() == "sac":
        return TwoAgentSAC(config, seed)
    if name.lower() == "hasac":
        return HASAC(config, seed)
    if name.lower() == "mappo":
        return MAPPO(config, seed)
    if name.lower() == "maddpg":
        return MADDPG(config, seed)
    if name.lower() == "td3":
        return TD3(obs_dim, action_dim, config, seed)
    raise ValueError(f"Unknown algorithm {name!r}; choose maddpg, mappo, hasac, sac, td3")


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


def _atomic_save_agent(agent, path: Path):
    """Write a checkpoint atomically so an interrupted save cannot clobber it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        agent.save(temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_top_checkpoint_manifest(path: Path, checkpoints):
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as f:
            json.dump(checkpoints, f, indent=2)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _consider_top_reward_checkpoint(agent, run_dir: Path, episode: int,
                                    episode_return: float, checkpoints, limit=5):
    """Keep model snapshots for the five highest-return completed episodes."""
    ranked = sorted(checkpoints, key=lambda row: (-row["return"], row["episode"]))
    if len(ranked) >= limit and episode_return <= ranked[limit - 1]["return"]:
        return ranked

    return_token = f"{episode_return:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    filename = f"top_reward_episode_{episode:04d}_return_{return_token}.pt"
    checkpoint_path = run_dir / filename
    _atomic_save_agent(agent, checkpoint_path)
    ranked.append({
        "episode": int(episode),
        "return": float(episode_return),
        "checkpoint": filename,
    })
    ranked.sort(key=lambda row: (-row["return"], row["episode"]))
    removed = ranked[limit:]
    ranked = ranked[:limit]
    for row in removed:
        stale_path = run_dir / row["checkpoint"]
        if stale_path.exists():
            stale_path.unlink()
    _write_top_checkpoint_manifest(run_dir / "top_reward_checkpoints.json", ranked)
    return ranked


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
    display_floor = 0.0
    window = max(1, min(int(window), len(rewards)))
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    # Negative returns remain intact in episodes.csv and are shown at zero so
    # the plot can use the requested zero baseline.
    ax.plot(episodes, np.maximum(rewards, display_floor), color="#4C78A8", alpha=0.45, linewidth=1.0,
            label="Episode reward")
    ax.plot(episodes[window - 1:], np.maximum(moving, display_floor), color="#E45756", linewidth=2.0,
            label=f"{window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (sum of per-step rewards)")
    ax.set_title("SAC training reward")
    ax.grid(True, alpha=0.25)
    # Keep the requested zero lower limit; leave the upper limit data-driven.
    ax.set_ylim(bottom=display_floor)
    ax.text(0.99, 0.02, "Negative returns are shown at 0; see episodes.csv for raw returns",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555")
    ax.legend()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def _plot_agent_episode_rewards(
    episodes_csv: Path,
    output_png: Path,
    reward_field: str,
    agent_label: str,
    window: int = 10,
):
    """Plot one independent agent's raw episode returns and moving average."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with episodes_csv.open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get(reward_field) not in (None, "")]
    if not rows:
        return
    episodes = np.asarray([int(row["episode"]) for row in rows], dtype=np.int32)
    rewards = np.asarray([float(row[reward_field]) for row in rows], dtype=np.float64)
    window = max(1, min(int(window), len(rewards)))
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(episodes, rewards, color="#4C78A8", alpha=0.45, linewidth=1.0,
            label="Episode reward")
    ax.plot(episodes[window - 1:], moving, color="#E45756", linewidth=2.0,
            label=f"{window}-episode moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward (sum of per-step rewards)")
    ax.set_title(f"{agent_label} SAC training reward")
    ax.grid(True, alpha=0.25)
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
        "#FF9DA6", "#9D755D", "#BAB0AC", "#ECA82C", "#5F9ED1", "#1F9E89",
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
    normalization_enabled = bool(
        cfg.get("observation", {}).get("normalization_enabled", True)
    )
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
            if isinstance(agent, TwoAgentSAC):
                driving_state, energy_state = split_agent_observations(
                    obs, vehicle_cfg, normalization_enabled
                )
                driving_action = agent.driving.select_action(driving_state, deterministic=True)
                energy_action = agent.energy.select_action(energy_state, deterministic=True)
                physical_action = map_split_actions(
                    driving_action, energy_action, env.action_space
                )
            elif isinstance(agent, MADDPG):
                driving_state, energy_state = split_agent_observations(
                    obs, vehicle_cfg, normalization_enabled
                )
                driving_action, energy_action = agent.select_actions(
                    driving_state, energy_state, deterministic=True
                )
                actor_action_unit = np.asarray(
                    [driving_action[0], energy_action[0]], dtype=np.float32
                )
                rule = (
                    agent.rule_adapter.predict(obs)
                    if getattr(agent, "rule_adapter", None) is not None else None
                )
                action_unit = agent.fuse_numpy_actions(
                    actor_action_unit,
                    rule["values"][0] if rule is not None else None,
                    rule["active"][0] if rule is not None else None,
                )
                physical_action = map_split_actions(
                    action_unit[:1], action_unit[1:], env.action_space
                )
            elif isinstance(agent, HASAC):
                driving_state, energy_state = split_agent_observations(
                    obs, vehicle_cfg, normalization_enabled
                )
                driving_action, energy_action = agent.select_actions(
                    driving_state, energy_state, deterministic=True,
                    full_observation=obs,
                )
                physical_action = map_split_actions(
                    driving_action, energy_action, env.action_space
                )
            elif isinstance(agent, MAPPO):
                driving_state, energy_state = split_agent_observations(
                    obs, vehicle_cfg, normalization_enabled
                )
                norm_obs = normalize_observation(
                    obs, vehicle_cfg, enabled=normalization_enabled
                )
                driving_action, energy_action, _, _ = agent.select_actions(
                    driving_state, energy_state, norm_obs, deterministic=True,
                    full_observation=obs,
                )
                physical_action = map_split_actions(
                    driving_action, energy_action, env.action_space
                )
            else:
                norm_obs = normalize_observation(
                    obs, vehicle_cfg, enabled=normalization_enabled
                )
                action_unit = agent.select_action(norm_obs, deterministic=True)
                physical_action = map_action(action_unit, env.action_space)
            obs, reward, terminated, truncated, info = env.step(physical_action)
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
    parser = argparse.ArgumentParser(description="Train multi-agent policies on the SUMO FCHEV environment")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--algorithm", choices=["maddpg", "mappo", "sac", "hasac", "td3"], default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override the selected algorithm's discount factor")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional per-episode smoke-test cap")
    parser.add_argument(
        "--normalize-observations", action=argparse.BooleanOptionalAction,
        default=None,
        help="Override observation normalization (default: enabled by config)",
    )
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                        help="Compute device; CUDA stores policy/value networks on GPU")
    parser.add_argument("--evaluation-episodes", type=int, default=None,
                        help="Override periodic evaluation episode count (0 disables evaluations)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prior-program-name", default=None,
        help="Override llm_prior.program_name for HASAC priors or MADDPG rule programs",
    )
    parser.add_argument(
        "--maddpg-rule-output-mode",
        choices=MADDPG_RULE_MODES,
        default=None,
        help="MADDPG rule output adapter: none, physical_action, normalized_action, or pre_tanh",
    )
    parser.add_argument(
        "--maddpg-rule-lambda", type=float, default=None,
        help="MADDPG actor/rule action blend weight in [0, 1]",
    )
    parser.add_argument(
        "--resume-checkpoint", default=None,
        help="Resume a checkpoint produced by the selected algorithm",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--experiment-name", default=None,
                        help="Optional label included in this run's unique output directory")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    observation_cfg = cfg.setdefault("observation", {})
    if args.normalize_observations is not None:
        observation_cfg["normalization_enabled"] = bool(args.normalize_observations)
    normalization_enabled = bool(observation_cfg.get("normalization_enabled", True))
    if args.prior_program_name is not None:
        cfg.setdefault("llm_prior", {})["program_name"] = args.prior_program_name
    algorithm_name = (args.algorithm or cfg["training"]["algorithm"]).lower()
    resume_checkpoint = None
    if args.resume_checkpoint is not None:
        resume_checkpoint = Path(args.resume_checkpoint)
        if not resume_checkpoint.is_absolute():
            resume_checkpoint = root / resume_checkpoint
        resume_checkpoint = resume_checkpoint.resolve()
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        if algorithm_name not in ("hasac", "mappo", "maddpg"):
            raise ValueError("--resume-checkpoint currently supports HASAC, MAPPO, and MADDPG")
        cfg["training"]["resume_checkpoint"] = str(resume_checkpoint)
        cfg["training"]["resume_replay_buffer"] = False
    resumed_from_checkpoint = resume_checkpoint is not None
    tr = dict(cfg["training"])
    maddpg_rule_mode = str(
        args.maddpg_rule_output_mode
        if args.maddpg_rule_output_mode is not None
        else tr.get("maddpg_rule_output_mode", "none")
    )
    maddpg_rule_lambda = float(
        args.maddpg_rule_lambda
        if args.maddpg_rule_lambda is not None
        else tr.get("maddpg_rule_lambda", cfg.get("llm_prior", {}).get("lambda_0", 0.2))
    )
    if maddpg_rule_mode not in MADDPG_RULE_MODES:
        raise ValueError(f"Unsupported MADDPG rule mode {maddpg_rule_mode!r}")
    if not 0.0 <= maddpg_rule_lambda <= 1.0:
        raise ValueError("--maddpg-rule-lambda must be in [0, 1]")
    if maddpg_rule_mode != "none" and algorithm_name != "maddpg":
        raise ValueError("MADDPG rule output modes can only be used with --algorithm maddpg")
    maddpg_rule_program = None
    if maddpg_rule_mode != "none":
        maddpg_rule_program = cfg.get("llm_prior", {}).get("program_name")
        if not maddpg_rule_program:
            raise ValueError(
                "A MADDPG rule mode requires llm_prior.program_name or --prior-program-name"
            )
    tr["maddpg_rule_output_mode"] = maddpg_rule_mode
    tr["maddpg_rule_lambda"] = maddpg_rule_lambda
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
    tr["actor_hidden_activation"] = "LeakyReLU(negative_slope=0.01)"
    tr["critic_hidden_activation"] = "ReLU"
    tr["energy_observation_features"] = [
        "ego_acceleration_m_s2", "ego_speed_m_s", "soc", "battery_soh",
        "fcs_soh", "edu_soh",
    ]
    if args.gamma is not None:
        tr["gamma"] = float(args.gamma)
    cfg["training"].update(tr)
    if args.gui:
        cfg["sumo"]["gui"] = True
    if args.max_steps is not None:
        cfg["sumo"]["episode_limit_s"] = int(args.max_steps)
    seed_everything(seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{algorithm_name}_{stamp}_seed{seed}"
    if args.experiment_name:
        experiment_label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.experiment_name).strip("._-")
        if not experiment_label:
            raise ValueError("--experiment-name must contain at least one letter or number")
        run_name = f"{algorithm_name}_{experiment_label}_{stamp}_seed{seed}"
    run_dir = (root / args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    # Keep the requested singular filename available while preserving the
    # established losses.csv name without duplicating a potentially large log.
    try:
        (run_dir / "loss.csv").symlink_to("losses.csv")
    except OSError:
        pass
    # Every process owns its SUMO XML and network files. This prevents concurrent
    # ablation/evaluation runs from overwriting one another's generated scenario.
    cfg["sumo"]["scenario_output_dir"] = str(run_dir / "scenario")
    cfg["training"]["agent_topology"] = (
        "hasac_shared_reward_centralized_critic_decentralized_actors"
        if algorithm_name == "hasac" else
        "mappo_shared_reward_centralized_value_decentralized_actors"
        if algorithm_name == "mappo" else
        "maddpg_separate_rewards_centralized_critics_decentralized_actors"
        if algorithm_name == "maddpg" else
        "independent_driving_and_energy_sac" if algorithm_name == "sac" else
        "single_agent"
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    with (run_dir / "checkpoint_policy.json").open("w", encoding="utf-8") as f:
        json.dump({
            "top_reward_checkpoints": 5,
            "ranking_metric": "episode return",
            "final_checkpoint": "final_episode_<episode_count>.pt",
            "rolling_recovery_checkpoint": "final.pt",
            "checkpoint_contents": "full agent and optimizer state",
            "agents": (
                ["driving", "energy_management"]
                if algorithm_name in ("sac", "hasac", "mappo", "maddpg") else ["joint"]
            ),
            "hasac_critic": "centralized twin soft Q over joint state and actions"
            if algorithm_name == "hasac" else None,
            "hasac_reward": "same shared reward for both agents: driving_reward + energy_reward"
            if algorithm_name == "hasac" else None,
            "mappo_critic": "centralized value function over the full 15-dimensional normalized state"
            if algorithm_name == "mappo" else None,
            "mappo_reward": "same shared team reward for both decentralized actors"
            if algorithm_name == "mappo" else None,
            "maddpg_critics": "two centralized critics over the full 15-dimensional state and joint action"
            if algorithm_name == "maddpg" else None,
            "maddpg_reward": "driving critic uses driving reward; energy critic uses energy reward"
            if algorithm_name == "maddpg" else None,
            "maddpg_rule_output_mode": maddpg_rule_mode if algorithm_name == "maddpg" else None,
            "maddpg_rule_program": maddpg_rule_program if algorithm_name == "maddpg" else None,
            "maddpg_rule_lambda": maddpg_rule_lambda if algorithm_name == "maddpg" else None,
        }, f, indent=2)
    top_reward_checkpoints = []
    _write_top_checkpoint_manifest(run_dir / "top_reward_checkpoints.json", top_reward_checkpoints)

    env = FCHEVEnv(project_root=root, config=cfg, render_mode="human" if args.gui else None)
    agent_config = dict(tr)
    agent_config["llm_prior"] = dict(cfg.get("llm_prior", {}))
    agent_config["action_low"] = env.action_space.low.tolist()
    agent_config["action_high"] = env.action_space.high.tolist()
    agent = make_agent(
        algorithm_name, env.observation_space.shape[0],
        env.action_space.shape[0], agent_config, seed,
    )
    if isinstance(agent, MADDPG) and maddpg_rule_mode != "none":
        agent.rule_adapter = MADDPGRuleAdapter(
            maddpg_rule_program,
            maddpg_rule_mode,
            env.action_space.low,
            env.action_space.high,
            config=cfg.get("llm_prior", {}),
        )
    elif isinstance(agent, MADDPG):
        agent.rule_adapter = None
    if resumed_from_checkpoint:
        if not isinstance(agent, (HASAC, MAPPO, MADDPG)):
            raise TypeError("Checkpoint loading requires a HASAC, MAPPO, or MADDPG agent")
        requested_prior_cfg = dict(cfg.get("llm_prior", {}))
        agent.load(resume_checkpoint, load_optimizers=True)
        # The source checkpoint may have been trained without a prior. Restore
        # this run's requested prior after loading the network/optimizer state.
        agent.prior_cfg = requested_prior_cfg
        agent.lambda_mean = float(requested_prior_cfg.get("lambda_0", 0.5))
        agent.lambda_std = float(requested_prior_cfg.get("lambda_1", 0.5))
        if isinstance(agent, MAPPO):
            agent._validate_prior_lambdas()
        elif isinstance(agent, HASAC):
            if not 0.0 <= agent.lambda_mean <= 1.0:
                raise ValueError("llm_prior.lambda_0 must be in [0, 1]")
            if not 0.0 <= agent.lambda_std <= 1.0:
                raise ValueError("llm_prior.lambda_1 must be in [0, 1]")
        if isinstance(agent, (HASAC, MAPPO)):
            agent.prior_provider = agent._build_prior_provider()
            if requested_prior_cfg.get("enabled", False) and agent.prior_provider is None:
                raise ValueError("Resume config enables prior but did not load its program")
        # Preserve the new run's configured discount factor; on-policy data is
        # always recollected after resume.
        agent.gamma = float(tr["gamma"])
    if agent.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(agent.device)
        device_label = f"{torch.cuda.get_device_name(agent.device)} (CUDA)"
    else:
        device_label = str(agent.device).upper()
    if isinstance(agent, MAPPO):
        replay_mb = 0.0
        replay_device_label = "on-policy episode rollouts (CPU), networks on " + str(agent.device)
    elif isinstance(agent, TwoAgentSAC):
        replay_agents = agent.agents
        replay_mb = sum(
            subagent.replay.data.numel() * subagent.replay.data.element_size()
            for subagent in replay_agents.values()
        ) / (1024 ** 2)
        replay_device_label = ", ".join(
            f"{name}={subagent.replay.device}"
            for name, subagent in replay_agents.items()
        )
    elif isinstance(agent, HASAC):
        replay_mb = agent.replay.data.numel() * agent.replay.data.element_size() / (1024 ** 2)
        replay_device_label = str(agent.replay.device)
    else:
        replay_mb = agent.replay.data.numel() * agent.replay.data.element_size() / (1024 ** 2)
        replay_device_label = str(agent.replay.device)
    if isinstance(agent, MAPPO):
        print(
            f"Training on {device_label}; on-policy rollout collection is on CPU, "
            f"MAPPO networks are on {agent.device}; SUMO remains CPU-driven."
        )
    else:
        print(
            f"Training on {device_label}; replay buffers on {replay_device_label} "
            f"({replay_mb:.1f} MiB); SUMO/environment remains CPU-driven."
        )
    print(
        f"Run directory: {run_dir}; topology={cfg['training']['agent_topology']}; "
        f"gamma={tr['gamma']}; seed={seed}",
        flush=True,
    )
    global_step = 0
    best_return = -float("inf")
    start_time = time.perf_counter()
    try:
        for episode in range(1, episodes + 1):
            prior_schedule_phase = _apply_prior_lambda_schedule(
                agent, tr.get("prior_lambda_schedule"), episode
            )
            if isinstance(agent, (HASAC, MAPPO)):
                driving_lambda_0, driving_lambda_1 = agent._agent_lambda_weights("driving")
                energy_lambda_0, energy_lambda_1 = agent._agent_lambda_weights("energy")
            else:
                driving_lambda_0 = driving_lambda_1 = None
                energy_lambda_0 = energy_lambda_1 = None
            # Keep the exogenous SUMO scenario identical across episodes.
            # Policy exploration and network updates remain stochastic, but
            # signals and the lead-vehicle stop plan use the same scene seed.
            obs, reset_info = env.reset(seed=seed)
            state = normalize_observation(
                obs, cfg["vehicle"], enabled=normalization_enabled
            )
            if is_dual_policy_agent(agent):
                driving_state, energy_state = split_agent_observations(
                    obs, cfg["vehicle"], normalization_enabled
                )
            total_return = 0.0
            total_driving_return = 0.0
            total_energy_return = 0.0
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
                # Keep the exact raw state passed to the prior for the trace.
                # The actor states are normalized/split views, while a rule
                # prior receives the complete raw environment observation.
                policy_observation = np.asarray(obs, dtype=np.float32).copy()
                # Capture the exact normalized full observation used by the
                # actor/critic before env.step advances state to next_state.
                policy_input_observation = np.asarray(state, dtype=np.float32).copy()
                prior_output = None
                policy_record = None
                maddpg_rule_output = None
                maddpg_actor_action_unit = None
                maddpg_rule_fused_action_unit = None
                energy_observed_acceleration_normalized = (
                    float(energy_state[0]) if is_dual_policy_agent(agent) else None
                )
                if isinstance(agent, TwoAgentSAC):
                    if global_step < learning_starts and not resumed_from_checkpoint:
                        driving_action_unit = np.random.uniform(-1.0, 1.0, size=1).astype(np.float32)
                        energy_action_unit = np.random.uniform(-1.0, 1.0, size=1).astype(np.float32)
                    else:
                        driving_action_unit = agent.driving.select_action(
                            driving_state, deterministic=False
                        )
                        energy_action_unit = agent.energy.select_action(
                            energy_state, deterministic=False
                        )
                    action_unit = np.asarray(
                        [driving_action_unit[0], energy_action_unit[0]], dtype=np.float32
                    )
                    physical_action = map_split_actions(
                        driving_action_unit, energy_action_unit, env.action_space
                    )
                elif isinstance(agent, MADDPG):
                    maddpg_rule_output = (
                        agent.rule_adapter.predict(obs)
                        if getattr(agent, "rule_adapter", None) is not None else None
                    )
                    rule_values_now = (
                        maddpg_rule_output["values"][0]
                        if maddpg_rule_output is not None
                        else np.zeros(2, dtype=np.float32)
                    )
                    rule_active_now = (
                        maddpg_rule_output["active"][0]
                        if maddpg_rule_output is not None
                        else np.zeros(2, dtype=bool)
                    )
                    if global_step < learning_starts and not resumed_from_checkpoint:
                        actor_behavior_action_unit = np.random.uniform(
                            -1.0, 1.0, size=2
                        ).astype(np.float32)
                        # Keep the initial replay warm-up random and prior-free.
                        action_unit = actor_behavior_action_unit.copy()
                    else:
                        driving_action_unit, energy_action_unit = agent.select_actions(
                            driving_state, energy_state, deterministic=True
                        )
                        maddpg_actor_action_unit = np.asarray(
                            [driving_action_unit[0], energy_action_unit[0]],
                            dtype=np.float32,
                        )
                        exploration_noise = float(tr.get("exploration_noise", 0.1))
                        actor_behavior_action_unit = np.clip(
                            maddpg_actor_action_unit
                            + np.random.normal(0.0, exploration_noise, size=2),
                            -1.0, 1.0,
                        ).astype(np.float32)
                        action_unit = agent.fuse_numpy_actions(
                            actor_behavior_action_unit,
                            rule_values_now,
                            rule_active_now,
                        )
                    maddpg_rule_fused_action_unit = action_unit.copy()
                    physical_action = map_split_actions(
                        action_unit[:1], action_unit[1:], env.action_space
                    )
                elif isinstance(agent, HASAC):
                    if global_step < learning_starts and not resumed_from_checkpoint:
                        driving_action_unit = np.random.uniform(-1.0, 1.0, size=1).astype(np.float32)
                        energy_action_unit = np.random.uniform(-1.0, 1.0, size=1).astype(np.float32)
                    else:
                        if agent.prior_provider is not None:
                            prior_output = agent._predict_prior(obs, batch_size=1)
                        driving_action_unit, energy_action_unit = agent.select_actions(
                            driving_state, energy_state, deterministic=False,
                            full_observation=obs, prior_output=prior_output,
                        )
                    action_unit = np.asarray(
                        [driving_action_unit[0], energy_action_unit[0]], dtype=np.float32
                    )
                    physical_action = map_split_actions(
                        driving_action_unit, energy_action_unit, env.action_space
                    )
                elif isinstance(agent, MAPPO):
                    driving_action_unit, energy_action_unit, policy_record, prior_output = (
                        agent.select_actions(
                            driving_state, energy_state, state,
                            deterministic=False, full_observation=obs,
                        )
                    )
                    action_unit = np.asarray(
                        [driving_action_unit[0], energy_action_unit[0]], dtype=np.float32
                    )
                    physical_action = map_split_actions(
                        driving_action_unit, energy_action_unit, env.action_space
                    )
                else:
                    if global_step < learning_starts and not resumed_from_checkpoint:
                        action_unit = np.random.uniform(-1.0, 1.0, size=env.action_space.shape[0]).astype(np.float32)
                    else:
                        action_unit = agent.select_action(state, deterministic=False)
                        if algorithm_name == "td3":
                            noise = np.random.normal(0.0, tr["exploration_noise"], size=action_unit.shape)
                            action_unit = np.clip(action_unit + noise, -1.0, 1.0)
                    physical_action = map_action(action_unit, env.action_space)
                next_obs, reward, terminated, truncated, info = env.step(physical_action)
                next_state = normalize_observation(
                    next_obs, cfg["vehicle"], enabled=normalization_enabled
                )
                done = terminated or truncated
                if is_dual_policy_agent(agent):
                    next_driving_state, next_energy_state = split_agent_observations(
                        next_obs, cfg["vehicle"], normalization_enabled
                    )
                    executed_acceleration = info.get("executed_acceleration_m_s2")
                    if executed_acceleration is not None:
                        # Energy actor observations use the normalized ego
                        # acceleration feature just like the current state.
                        next_energy_state[0] = (
                            float(executed_acceleration) / 1.5
                            if normalization_enabled
                            else float(executed_acceleration)
                        )
                    driving_reward, energy_reward = split_agent_rewards(info)
                    if not np.isclose(driving_reward + energy_reward, reward, atol=1e-6):
                        raise RuntimeError(
                            "Driving and energy rewards do not sum to the environment reward"
                        )
                    if isinstance(agent, HASAC):
                        agent.store_transition(
                            driving_state, energy_state,
                            driving_action_unit, energy_action_unit,
                            driving_reward, energy_reward,
                            next_driving_state, next_energy_state, done,
                            expected_team_reward=reward,
                        )
                    elif isinstance(agent, MAPPO):
                        if not np.isclose(driving_reward + energy_reward, reward, atol=1e-6):
                            raise RuntimeError(
                                "MAPPO shared reward must equal driving and energy reward components"
                            )
                        artificial_truncation = (
                            args.max_steps is not None
                            and local_step + 1 >= args.max_steps
                            and not terminated and not truncated
                        )
                        agent.store_transition(
                            policy_record, reward, next_state,
                            terminated=terminated,
                            truncated=bool(truncated or artificial_truncation),
                        )
                    elif isinstance(agent, MADDPG):
                        next_maddpg_rule_output = (
                            agent.rule_adapter.predict(next_obs)
                            if getattr(agent, "rule_adapter", None) is not None else None
                        )
                        next_rule_values = (
                            next_maddpg_rule_output["values"][0]
                            if next_maddpg_rule_output is not None
                            else np.zeros(2, dtype=np.float32)
                        )
                        next_rule_active = (
                            next_maddpg_rule_output["active"][0]
                            if next_maddpg_rule_output is not None
                            else np.zeros(2, dtype=bool)
                        )
                        agent.store_transition(
                            state, action_unit, driving_reward, energy_reward,
                            next_state, bool(terminated),
                            rule_values_now, rule_active_now,
                            next_rule_values, next_rule_active,
                        )
                    else:
                        agent.driving.replay.add(
                            driving_state, driving_action_unit,
                            driving_reward, next_driving_state, done,
                        )
                        agent.energy.replay.add(
                            energy_state, energy_action_unit,
                            energy_reward, next_energy_state, done,
                        )
                    driving_state, energy_state = next_driving_state, next_energy_state
                    total_driving_return += driving_reward
                    total_energy_return += energy_reward
                else:
                    agent.replay.add(state, action_unit, reward, next_state, done)
                state = next_state
                # Keep the raw full observation synchronized with the split
                # actor states. Without this assignment, prior-enabled runs
                # repeatedly condition on the episode's reset observation.
                obs = next_obs
                total_return += reward
                total_hydrogen += info.get("hydrogen_consumed_g", 0.0)
                total_safety_violations += info.get("safety_violations", 0)
                total_clips += int(info.get("constraint_clipping", False))
                collisions += info.get("collision_count", 0)
                collision_events += int(info.get("collision_event", False))
                red_light_violations += int(info.get("red_light_violation", False))
                speeding_violations += int(info.get("speeding_violation", False))
                # Keep the last cumulative count if a terminal info dict omits
                # it; a missing field must not reset an episode total to zero.
                traffic_lights_passed = int(
                    info.get("traffic_lights_passed", traffic_lights_passed)
                )
                lead_traffic_lights_passed = int(
                    info.get("lead_traffic_lights_passed", lead_traffic_lights_passed)
                )
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
                    "driving_agent_reward": driving_reward if is_dual_policy_agent(agent) else None,
                    "energy_agent_reward": energy_reward if is_dual_policy_agent(agent) else None,
                    **{
                        f"policy_obs_{name}": float(value)
                        for name, value in zip(
                            (
                                "soc", "battery_soh", "fcs_soh", "edu_soh",
                                "ego_acceleration_m_s2", "ego_speed_m_s",
                                "lead_acceleration_m_s2", "lead_speed_m_s",
                                "following_distance_m", "next_signal_state",
                                "next_signal_distance_m", "signal_phase_remaining_s",
                                "current_speed_limit_m_s", "next_speed_limit_m_s",
                                "distance_to_next_speed_region_m",
                            ),
                            policy_observation,
                        )
                    },
                    **{
                        f"policy_input_{name}": float(value)
                        for name, value in zip(
                            (
                                "soc", "battery_soh_loss_x1000",
                                "fcs_soh_loss_x1000", "edu_soh_loss_x1000",
                                "ego_acceleration_div1p5", "ego_speed_div20",
                                "lead_acceleration_div1p5", "lead_speed_div20",
                                "following_distance_div1000", "traffic_light_state",
                                "next_signal_distance_div1000",
                                "signal_phase_remaining_div100",
                                "current_speed_limit_div20",
                                "next_speed_limit_div20",
                                "distance_to_next_speed_region_div1000",
                            ),
                            policy_input_observation,
                        )
                    },
                    "policy_driving_prior_mean": (
                        float(prior_output.driving.mean[0, 0])
                        if prior_output is not None else None
                    ),
                    "policy_driving_prior_std": (
                        float(prior_output.driving.std[0, 0])
                        if prior_output is not None else None
                    ),
                    "policy_driving_prior_active": (
                        int(prior_output.driving.active[0, 0])
                        if prior_output is not None else None
                    ),
                    "policy_driving_prior_lambda_0": (
                        agent._agent_lambda_weights("driving")[0]
                        if isinstance(agent, (HASAC, MAPPO)) and prior_output is not None else None
                    ),
                    "policy_driving_prior_lambda_1": (
                        agent._agent_lambda_weights("driving")[1]
                        if isinstance(agent, (HASAC, MAPPO)) and prior_output is not None else None
                    ),
                    "policy_energy_prior_mean": (
                        float(prior_output.energy.mean[0, 0])
                        if prior_output is not None else None
                    ),
                    "policy_energy_prior_std": (
                        float(prior_output.energy.std[0, 0])
                        if prior_output is not None else None
                    ),
                    "policy_energy_prior_active": (
                        int(prior_output.energy.active[0, 0])
                        if prior_output is not None else None
                    ),
                    "policy_energy_prior_lambda_0": (
                        agent._agent_lambda_weights("energy")[0]
                        if isinstance(agent, (HASAC, MAPPO)) and prior_output is not None else None
                    ),
                    "policy_energy_prior_lambda_1": (
                        agent._agent_lambda_weights("energy")[1]
                        if isinstance(agent, (HASAC, MAPPO)) and prior_output is not None else None
                    ),
                    "policy_maddpg_rule_mode": (
                        agent.rule_output_mode if isinstance(agent, MADDPG) else None
                    ),
                    "policy_maddpg_rule_lambda": (
                        agent.rule_lambda if isinstance(agent, MADDPG) else None
                    ),
                    "policy_maddpg_driving_actor_unit": (
                        float(maddpg_actor_action_unit[0])
                        if maddpg_actor_action_unit is not None else None
                    ),
                    "policy_maddpg_energy_actor_unit": (
                        float(maddpg_actor_action_unit[1])
                        if maddpg_actor_action_unit is not None else None
                    ),
                    "policy_maddpg_driving_rule_value": (
                        float(maddpg_rule_output["values"][0, 0])
                        if maddpg_rule_output is not None else None
                    ),
                    "policy_maddpg_energy_rule_value": (
                        float(maddpg_rule_output["values"][0, 1])
                        if maddpg_rule_output is not None else None
                    ),
                    "policy_maddpg_driving_rule_active": (
                        int(maddpg_rule_output["active"][0, 0])
                        if maddpg_rule_output is not None else None
                    ),
                    "policy_maddpg_energy_rule_active": (
                        int(maddpg_rule_output["active"][0, 1])
                        if maddpg_rule_output is not None else None
                    ),
                    "policy_maddpg_driving_fused_unit": (
                        float(maddpg_rule_fused_action_unit[0])
                        if maddpg_rule_fused_action_unit is not None else None
                    ),
                    "policy_maddpg_energy_fused_unit": (
                        float(maddpg_rule_fused_action_unit[1])
                        if maddpg_rule_fused_action_unit is not None else None
                    ),
                    "energy_agent_observation_ego_acceleration_div1p5": energy_observed_acceleration_normalized,
                    "energy_agent_observation_ego_speed_div20": (
                        float(energy_state[1]) if is_dual_policy_agent(agent) else None
                    ),
                    "energy_agent_observation_soc": (
                        float(energy_state[2]) if is_dual_policy_agent(agent) else None
                    ),
                    "energy_agent_observation_battery_soh_loss_x1000": (
                        float(energy_state[3]) if is_dual_policy_agent(agent) else None
                    ),
                    "energy_agent_observation_fcs_soh_loss_x1000": (
                        float(energy_state[4]) if is_dual_policy_agent(agent) else None
                    ),
                    "energy_agent_observation_edu_soh_loss_x1000": (
                        float(energy_state[5]) if is_dual_policy_agent(agent) else None
                    ),
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
                    "following_distance_valid": info.get("following_distance_valid"),
                    "following_distance_reduction_m": info.get("following_distance_reduction_m"),
                    "following_forward_distance_m": info.get("following_forward_distance_m"),
                    "following_region_max_speed_m_s": info.get(
                        "following_region_max_speed_m_s"
                    ),
                    "action_policy_acceleration_unit": float(action_unit[0]),
                    "action_policy_fcs_power_unit": float(action_unit[1]),
                    "action_policy_gaussian_acceleration_raw": (
                        float(policy_record["agents"]["driving"]["latent"][0])
                        if isinstance(agent, MAPPO) and policy_record is not None else None
                    ),
                    "action_policy_gaussian_fcs_power_raw": (
                        float(policy_record["agents"]["energy"]["latent"][0])
                        if isinstance(agent, MAPPO) and policy_record is not None else None
                    ),
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
                    "speeding_termination_violation": info.get(
                        "speeding_termination_violation"
                    ),
                    "speeding_termination_threshold_m_s": info.get(
                        "speeding_termination_threshold_m_s"
                    ),
                    "ego_in_intersection_speed_zone": info.get("ego_in_intersection_speed_zone"),
                    "lead_in_intersection_speed_zone": info.get("lead_in_intersection_speed_zone"),
                    "intersection_speed_limit_m_s": info.get("intersection_speed_limit_m_s"),
                    "active_speed_limit_m_s": info.get("active_speed_limit_m_s"),
                    "speeding_delta_v_m_s": info.get("speeding_delta_v_m_s"),
                    "speeding_penalty_beta": info.get("speeding_penalty_beta"),
                    "speeding_penalty": info.get("speeding_penalty"),
                    "traffic_light_crossing": info.get("traffic_light_crossing"),
                    "green_light_crossing": info.get("green_light_crossing"),
                    "green_light_crossing_bonus": info.get(
                        "green_light_crossing_bonus"
                    ),
                    "traffic_light_crossed_id": info.get("traffic_light_crossed_id"),
                    "traffic_lights_passed": info.get("traffic_lights_passed"),
                    "lead_traffic_light_crossing": info.get("lead_traffic_light_crossing"),
                    "lead_traffic_light_crossed_id": info.get("lead_traffic_light_crossed_id"),
                    "lead_traffic_lights_passed": info.get("lead_traffic_lights_passed"),
                    "traffic_light_state": info.get("traffic_light_state_observation"),
                    "traffic_light_distance_m": info.get("traffic_light_distance_m"),
                    "traffic_light_remaining_s": info.get("traffic_light_remaining_s"),
                    "traffic_light_state_before_crossing_step": info.get(
                        "traffic_light_state_before_crossing_step"
                    ),
                    "traffic_light_state_after_crossing_step": info.get(
                        "traffic_light_state_after_crossing_step"
                    ),
                    "traffic_light_state_at_crossing": info.get(
                        "traffic_light_state_at_crossing"
                    ),
                    "traffic_light_crossing_elapsed_s": info.get(
                        "traffic_light_crossing_elapsed_s"
                    ),
                    "time_to_traffic_light_s": info.get("time_to_traffic_light_s"),
                    "traffic_light_reward_active": info.get("traffic_light_reward_active"),
                    "traffic_light_reward": info.get("traffic_light_reward"),
                    "traffic_light_reward_contribution": info.get(
                        "traffic_light_reward_contribution"
                    ),
                    "speed_limit_entry_active": info.get("speed_limit_entry_active"),
                    "speed_limit_entry_error_before": info.get(
                        "speed_limit_entry_error_before"
                    ),
                    "speed_limit_entry_error_after": info.get(
                        "speed_limit_entry_error_after"
                    ),
                    "speed_limit_entry_predicted_speed_before_m_s": info.get(
                        "speed_limit_entry_predicted_speed_before_m_s"
                    ),
                    "speed_limit_entry_predicted_speed_after_m_s": info.get(
                        "speed_limit_entry_predicted_speed_after_m_s"
                    ),
                    "speed_limit_entry_target_m_s": info.get(
                        "speed_limit_entry_target_m_s"
                    ),
                    "speed_limit_entry_distance_m": info.get(
                        "speed_limit_entry_distance_m"
                    ),
                    "speed_limit_entry_reward": info.get("speed_limit_entry_reward"),
                    "speed_limit_entry_beta": info.get("speed_limit_entry_beta"),
                    "current_speed_limit_m_s": info.get("current_speed_limit_m_s"),
                    "current_speed_limit_at_action_m_s": info.get(
                        "current_speed_limit_at_action_m_s"
                    ),
                    "next_speed_limit_m_s": info.get("next_speed_limit_m_s"),
                    "distance_to_next_speed_region_m": info.get("distance_to_next_speed_region_m"),
                    "max_speed_action_violation": info.get("max_speed_action_violation"),
                    "max_speed_action_clipped": info.get("max_speed_action_clipped"),
                    "max_speed_limited_acceleration_m_s2": info.get(
                        "max_speed_limited_acceleration_m_s2"
                    ),
                    "max_speed_clip_delta_m_s2": info.get("max_speed_clip_delta_m_s2"),
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
                    "zero_speed_negative_accel_action": info.get(
                        "zero_speed_negative_accel_action"
                    ),
                    "speed_limit_positive_accel_action": info.get(
                        "speed_limit_positive_accel_action"
                    ),
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
                    "reward_speeding": reward_parts.get("speeding"),
                    "reward_soc_penalty": reward_parts.get("soc_penalty"),
                    "zero_speed_negative_accel_penalty": reward_parts.get(
                        "zero_speed_negative_accel_penalty"
                    ),
                    "speed_limit_positive_accel_penalty": reward_parts.get(
                        "speed_limit_positive_accel_penalty"
                    ),
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
                if isinstance(agent, TwoAgentSAC):
                    subagents = agent.agents
                elif isinstance(agent, HASAC):
                    subagents = {"joint": agent}
                else:
                    subagents = {}
                for subagent_name, subagent in subagents.items():
                    if len(subagent.replay) >= batch_size and (
                        global_step >= learning_starts or resumed_from_checkpoint
                    ):
                        for _ in range(int(tr.get("updates_per_step", 1))):
                            losses = subagent.update(batch_size)
                            losses["agent"] = subagent_name
                            losses["global_step"] = global_step
                            if isinstance(agent, HASAC):
                                for name in ("driving", "energy"):
                                    lambda_0, lambda_1 = agent._agent_lambda_weights(name)
                                    losses[f"{name}_prior_lambda_0"] = lambda_0
                                    losses[f"{name}_prior_lambda_1"] = lambda_1
                            updates.append(losses)
                if isinstance(agent, MADDPG) and len(agent.replay) >= batch_size and (
                    global_step >= learning_starts or resumed_from_checkpoint
                ):
                    for _ in range(int(tr.get("updates_per_step", 1))):
                        losses = agent.update(batch_size)
                        losses["agent"] = "maddpg"
                        losses["global_step"] = global_step
                        updates.append(losses)
                if terminated or truncated or (args.max_steps is not None and local_step >= args.max_steps):
                    break
            if isinstance(agent, MAPPO):
                mappo_losses = agent.update()
                if mappo_losses:
                    mappo_losses.update({
                        "agent": "mappo_shared_team",
                        "global_step": global_step,
                        "driving_prior_lambda_0": driving_lambda_0,
                        "driving_prior_lambda_1": driving_lambda_1,
                        "energy_prior_lambda_0": energy_lambda_0,
                        "energy_prior_lambda_1": energy_lambda_1,
                    })
                    updates.append(mappo_losses)
            _write_csv(run_dir / f"episode_{episode:04d}_trajectory.csv", trace, TRACE_FIELDS)
            row = {
                "episode": episode, "seed": seed,
                "prior_schedule_phase": (
                    prior_schedule_phase.get("name")
                    if prior_schedule_phase else None
                ),
                "driving_prior_lambda_0": driving_lambda_0,
                "driving_prior_lambda_1": driving_lambda_1,
                "energy_prior_lambda_0": energy_lambda_0,
                "energy_prior_lambda_1": energy_lambda_1,
                "return": total_return, "penalty_magnitude": -total_return,
                "driving_agent_return": total_driving_return if is_dual_policy_agent(agent) else None,
                "energy_agent_return": total_energy_return if is_dual_policy_agent(agent) else None,
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
            loss_fields = [
                "agent", "global_step", "critic_loss", "actor_loss",
                "driving_prior_lambda_0", "driving_prior_lambda_1",
                "energy_prior_lambda_0", "energy_prior_lambda_1",
                "driving_actor_loss", "energy_actor_loss", "alpha_loss", "alpha",
                "driving_critic_loss", "energy_critic_loss",
                "driving_q_mean", "energy_q_mean",
                "critic_q1_replay_mean", "critic_q2_replay_mean",
                "critic_target_mean", "critic_target_state_std",
                "driving_q_policy", "driving_q1_policy", "driving_q2_policy",
                "driving_log_prob", "driving_entropy", "driving_entropy_term",
                "driving_negative_q_term", "driving_raw_mean",
                "driving_raw_mean_state_std", "driving_raw_std",
                "driving_raw_std_state_std", "driving_fused_mean",
                "driving_fused_mean_state_std", "driving_fused_std",
                "driving_fused_std_state_std", "driving_sampled_action",
                "driving_sampled_action_state_std", "driving_actor_grad_norm",
                "driving_actor_hidden_grad_norm", "driving_actor_output_grad_norm",
                "energy_q_policy", "energy_q1_policy", "energy_q2_policy",
                "energy_log_prob", "energy_entropy", "energy_entropy_term",
                "energy_negative_q_term", "energy_raw_mean",
                "energy_raw_mean_state_std", "energy_raw_std",
                "energy_raw_std_state_std", "energy_fused_mean",
                "energy_fused_mean_state_std", "energy_fused_std",
                "energy_fused_std_state_std", "energy_sampled_action",
                "energy_sampled_action_state_std", "energy_actor_grad_norm",
                "energy_actor_hidden_grad_norm", "energy_actor_output_grad_norm",
                "driving_policy_loss", "energy_policy_loss",
                "driving_clip_fraction", "energy_clip_fraction",
                "driving_ratio", "energy_ratio", "value_loss",
                "critic_grad_norm", "rollout_steps", "team_reward_mean",
                "team_return", "value_normalizer_mean", "value_normalizer_std",
                "update_count",
            ]
            if updates:
                _append_csv(run_dir / "losses.csv", updates, loss_fields)
            if total_return > best_return:
                best_return = total_return
                _atomic_save_agent(agent, run_dir / "best.pt")
            top_reward_checkpoints = _consider_top_reward_checkpoint(
                agent, run_dir, episode, total_return, top_reward_checkpoints, limit=5
            )
            if episode % int(tr.get("save_interval_episodes", 10)) == 0 or episode == episodes:
                _atomic_save_agent(agent, run_dir / "final.pt")
                if episode == episodes:
                    _atomic_save_agent(
                        agent, run_dir / f"final_episode_{episode:04d}.pt"
                    )
            print(
                f"{algorithm_name.upper()} episode {episode}/{episodes}: "
                f"return={total_return:.3f}, H2={total_hydrogen:.3f} g, "
                f"steps={local_step}, clips={total_clips}, "
                f"end={info.get('terminated_reason') or info.get('truncated_reason') or 'unknown'}, "
                f"traffic_lights={traffic_lights_passed}/{lead_traffic_lights_passed}"
                + (
                    f", driving_return={total_driving_return:.3f}, "
                    f"energy_return={total_energy_return:.3f}"
                    if is_dual_policy_agent(agent) else ""
                )
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
                if is_dual_policy_agent(agent):
                    _plot_agent_episode_rewards(
                        run_dir / "episodes.csv",
                        run_dir / "driving_agent_episode_reward.png",
                        "driving_agent_return",
                        "Driving agent",
                    )
                    _plot_agent_episode_rewards(
                        run_dir / "episodes.csv",
                        run_dir / "energy_agent_episode_reward.png",
                        "energy_agent_return",
                        "Energy agent",
                    )
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
    driving_reward_plot = None
    energy_reward_plot = None
    if is_dual_policy_agent(agent):
        driving_reward_plot = run_dir / "driving_agent_episode_reward.png"
        energy_reward_plot = run_dir / "energy_agent_episode_reward.png"
        _plot_agent_episode_rewards(
            run_dir / "episodes.csv", driving_reward_plot,
            "driving_agent_return", "Driving agent",
        )
        _plot_agent_episode_rewards(
            run_dir / "episodes.csv", energy_reward_plot,
            "energy_agent_return", "Energy agent",
        )
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
        "replay_device": (
            "not_applicable_on_policy"
            if isinstance(agent, MAPPO) else
            {name: str(subagent.replay.device) for name, subagent in agent.agents.items()}
            if isinstance(agent, TwoAgentSAC) else str(agent.replay.device)
        ),
        "agent_topology": (
            "hasac_shared_reward_centralized_critic_decentralized_actors"
            if isinstance(agent, HASAC) else
            "mappo_shared_reward_centralized_value_decentralized_actors"
            if isinstance(agent, MAPPO) else
            "maddpg_separate_rewards_centralized_critics_decentralized_actors"
            if isinstance(agent, MADDPG) else
            "independent_driving_and_energy_sac"
            if isinstance(agent, TwoAgentSAC) else "single_agent"
        ),
        "agent_observation_dimensions": (
            {"driving": agent.driving_state_dim, "energy": agent.energy_state_dim}
            if isinstance(agent, (HASAC, MAPPO, MADDPG)) else
            {"driving": 11, "energy": 5}
            if isinstance(agent, TwoAgentSAC) else
            {"joint": env.observation_space.shape[0]}
        ),
        "centralized_critic_state_dimension": (
            agent.global_state_dim if isinstance(agent, MAPPO) else
            agent.driving_state_dim + agent.energy_state_dim
            if isinstance(agent, HASAC) else
            agent.global_state_dim if isinstance(agent, MADDPG) else None
        ),
        "hasac_entropy_temperature": agent.alpha if isinstance(agent, HASAC) else None,
        "hasac_shared_reward_definition": (
            "driving_reward + energy_reward; identical cooperative reward for both actors"
            if isinstance(agent, HASAC) else None
        ),
        "mappo_gamma": agent.gamma if isinstance(agent, MAPPO) else None,
        "mappo_gae_lambda": agent.gae_lambda if isinstance(agent, MAPPO) else None,
        "mappo_shared_reward_definition": (
            "driving_reward + energy_reward; identical team reward for both actors"
            if isinstance(agent, MAPPO) else None
        ),
        "maddpg_gamma": agent.gamma if isinstance(agent, MADDPG) else None,
        "maddpg_reward_definition": (
            "driving actor/critic optimize driving_reward; energy actor/critic optimize energy_reward"
            if isinstance(agent, MADDPG) else None
        ),
        "maddpg_rule_output_mode": (
            agent.rule_output_mode if isinstance(agent, MADDPG) else None
        ),
        "maddpg_rule_program": (
            agent.rule_adapter.program_name
            if isinstance(agent, MADDPG) and getattr(agent, "rule_adapter", None) is not None
            else None
        ),
        "maddpg_rule_lambda": (
            agent.rule_lambda if isinstance(agent, MADDPG) else None
        ),
        "gpu_peak_memory_mb": (
            torch.cuda.max_memory_allocated(agent.device) / (1024 ** 2)
            if agent.device.type == "cuda" else 0.0
        ),
        "run_directory": str(run_dir),
        "episode_reward_plot": str(reward_plot),
        "driving_agent_episode_reward_plot": (
            str(driving_reward_plot) if driving_reward_plot is not None else None
        ),
        "energy_agent_episode_reward_plot": (
            str(energy_reward_plot) if energy_reward_plot is not None else None
        ),
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
    if driving_reward_plot is not None:
        print(f"Saved driving-agent reward curve to {driving_reward_plot}")
        print(f"Saved energy-agent reward curve to {energy_reward_plot}")
    print(f"Saved episode state metrics to {metrics_plot}")
    print(f"Saved reward component curves to {components_plot}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
