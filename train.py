from __future__ import annotations

import argparse
import csv
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
    "requested_acceleration_m_s2", "requested_fcs_power_kw", "fcs_power_kw",
    "wheel_traction_force_n", "wheel_mechanical_power_w",
    "motor_speed_rpm", "motor_torque_nm", "edu_electrical_power_w",
    "edu_loss_power_w", "edu_efficiency", "friction_brake_force_n",
    "battery_power_w", "battery_current_pack_a", "battery_current_cell_a",
    "battery_terminal_voltage_pack_v", "hydrogen_consumed_g",
    "fcs_startup_degradation_percent", "fcs_low_power_degradation_percent",
    "fcs_high_power_degradation_percent", "fcs_load_change_degradation_percent",
    "collision_count", "safety_violations", "constraint_clipping",
    "reward_hydrogen_g", "reward_fcs_degradation_cumulative",
    "reward_battery_soh_loss", "reward_edu_clr_cumulative",
    "reward_comfort", "reward_safety", "reward_soc_penalty",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def map_action(action_unit, action_space):
    low = action_space.low.astype(np.float32)
    high = action_space.high.astype(np.float32)
    return low + (np.asarray(action_unit, dtype=np.float32) + 1.0) * 0.5 * (high - low)


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


def evaluate(env, agent, cfg, episodes, base_seed, vehicle_cfg, max_steps=None):
    rows = []
    for i in range(episodes):
        obs, _ = env.reset(seed=base_seed + i)
        total = 0.0
        hydrogen = 0.0
        steps = 0
        collisions = 0
        min_gap = float("inf")
        while True:
            norm_obs = normalize_observation(obs, vehicle_cfg)
            action_unit = agent.select_action(norm_obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(map_action(action_unit, env.action_space))
            total += reward
            hydrogen += info.get("hydrogen_consumed_g", 0.0)
            collisions += info.get("collision_count", 0)
            min_gap = min(min_gap, info.get("following_distance_m", float("inf")))
            steps += 1
            if terminated or truncated or (max_steps is not None and steps >= max_steps):
                break
        rows.append({
            "episode": i + 1, "seed": base_seed + i, "return": total,
            "steps": steps, "hydrogen_g": hydrogen, "collisions": collisions,
            "minimum_gap_m": min_gap,
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
    global_step = 0
    best_return = -float("inf")
    start_time = time.perf_counter()
    try:
        for episode in range(1, episodes + 1):
            obs, _ = env.reset(seed=seed + episode - 1)
            state = normalize_observation(obs, cfg["vehicle"])
            total_return = 0.0
            total_hydrogen = 0.0
            total_safety_violations = 0
            total_clips = 0
            collisions = 0
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
                min_gap = min(min_gap, info.get("following_distance_m", float("inf")))
                local_step += 1
                global_step += 1
                reward_parts = info.get("reward_components", {})
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
                    "safety_violations": info.get("safety_violations"),
                    "constraint_clipping": int(info.get("constraint_clipping", False)),
                    "reward_hydrogen_g": reward_parts.get("hydrogen_g"),
                    "reward_fcs_degradation_cumulative": reward_parts.get("fcs_degradation_cumulative"),
                    "reward_battery_soh_loss": reward_parts.get("battery_soh_loss"),
                    "reward_edu_clr_cumulative": reward_parts.get("edu_clr_cumulative"),
                    "reward_comfort": reward_parts.get("comfort"),
                    "reward_safety": reward_parts.get("safety"),
                    "reward_soc_penalty": reward_parts.get("soc_penalty"),
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
                "episode": episode, "seed": seed + episode - 1,
                "return": total_return, "steps": local_step,
                "hydrogen_g": total_hydrogen, "collisions": collisions,
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
            }
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
    finally:
        env.close()
    elapsed = time.perf_counter() - start_time
    summary = {
        "algorithm": algorithm_name, "seed": seed, "episodes": episodes,
        "environment_steps": global_step, "elapsed_seconds": elapsed,
        "env_steps_per_second_including_training": global_step / max(elapsed, 1e-9),
        "run_directory": str(run_dir),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
