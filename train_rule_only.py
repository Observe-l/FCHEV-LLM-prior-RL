from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from importlib import import_module
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np

from environment.fchev_env import FCHEVEnv, load_config


REWARD_COMPONENTS = (
    "hydrogen", "fcs", "battery", "edu", "comfort", "following", "soc",
    "traffic_light", "speeding", "terminal_penalty", "green_light_crossing_bonus",
    "zero_speed_negative_accel_penalty", "speed_limit_positive_accel_penalty",
    "speed_limit_entry",
)

OBSERVATION_NAMES = (
    "soc", "battery_soh", "fcs_soh", "edu_soh", "ego_acceleration_m_s2",
    "ego_speed_m_s", "lead_acceleration_m_s2", "lead_speed_m_s",
    "following_distance_m", "next_signal_state", "next_signal_distance_m",
    "signal_phase_remaining_s", "current_speed_limit_m_s", "next_speed_limit_m_s",
    "distance_to_next_speed_region_m",
)

UPDATE_TRACE_FIELDS = (
    "time_s", "driving_agent_reward", "energy_agent_reward",
    *(f"policy_obs_{name}" for name in OBSERVATION_NAMES),
    "requested_acceleration_m_s2", "action_executed_acceleration_m_s2",
    "action_requested_fcs_power_raw_kw", "action_executed_fcs_power_kw",
    "traffic_light_state_at_crossing", "traffic_light_crossing",
    "red_light_violation", "speeding_violation", "collision_event",
    "gap_overlap", "terminated_reason",
)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _split_rewards(info):
    parts = info.get("reward_components", {})

    def value(name):
        return float(parts.get(f"reward_contribution_{name}", 0.0) or 0.0)

    driving = sum(value(name) for name in (
        "following", "comfort", "traffic_light", "speeding",
        "green_light_crossing_bonus", "zero_speed_negative_accel_penalty",
        "speed_limit_positive_accel_penalty", "speed_limit_entry", "terminal_penalty",
    ))
    energy = sum(value(name) for name in (
        "hydrogen", "fcs", "battery", "edu", "soc",
    ))
    return driving, energy


def _map_actions(u_driving, u_energy, action_space):
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    u_driving = float(np.clip(u_driving, -1.0, 1.0))
    u_energy = float(np.clip(u_energy, -1.0, 1.0))
    acceleration = (
        u_driving * abs(float(low[0]))
        if u_driving < 0.0
        else u_driving * float(high[0])
    )
    power_kw = float(low[1]) + (u_energy + 1.0) * 0.5 * float(high[1] - low[1])
    return np.asarray([acceleration, power_kw], dtype=np.float32)


def _write_csv(path: Path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, row):
    fields = list(row)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _plot_results(run_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes_path = run_dir / "episodes.csv"
    with episodes_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    episodes = np.asarray([int(row["episode"]) for row in rows])
    returns = np.asarray([float(row["return"]) for row in rows])
    window = min(10, len(returns))
    moving = np.convolve(returns, np.ones(window) / window, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(episodes, returns, color="#4C78A8", alpha=0.45, label="Episode reward")
    ax.plot(episodes[window - 1:], moving, color="#E45756", linewidth=2,
            label=f"{window}-episode moving average")
    ax.set(xlabel="Episode", ylabel="Episode reward", title="Rule-only online LLM reward")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(run_dir / "episode_reward.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    for axis, field, ylabel, title in (
        (axes[0, 0], "hydrogen_g", "Hydrogen (g)", "Hydrogen consumption"),
        (axes[0, 1], "final_soc", "SOC", "Final battery SOC"),
        (axes[1, 0], "final_battery_soh", "SOH", "Battery SOH"),
        (axes[1, 1], "final_fcs_soh", "SOH", "FCS SOH"),
        (axes[2, 0], "final_edu_soh", "SOH", "EDU SOH"),
        (axes[2, 1], "steps", "Steps", "Episode length"),
    ):
        axis.plot(episodes, [float(row[field]) for row in rows], linewidth=1.2)
        axis.set(xlabel="Episode", ylabel=ylabel, title=title)
        axis.grid(True, alpha=0.25)
    fig.savefig(run_dir / "episode_metrics.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(REWARD_COMPONENTS)))
    for name, color in zip(REWARD_COMPONENTS, colors):
        values = np.asarray([float(row[f"reward_component_{name}"]) for row in rows])
        ax.plot(episodes, values, label=name, color=color, linewidth=1.2)
    ax.set(xlabel="Episode", ylabel="Reward component total",
           title="Rule-only episode reward components")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(run_dir / "reward_components.png", dpi=160)
    plt.close(fig)


def _finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rule_only_source(source: str) -> str:
    """Relabel legacy fusion comments without changing the initial rule logic."""
    old_header = '''# fchev_rule_prior.py
#
# Rule prior for FCHEV + HASAC fusion.
#
# This program returns latent Gaussian parameters before tanh:
#   driving: mean/std/active for the driving actor latent z_d
#   energy:  mean/std/active for the energy actor latent z_e
#
# It does NOT return physical acceleration or kW.
# It does NOT bypass SUMO, vehicle dynamics, battery/FCS/EDU feasibility,
# no-reverse handling, speed limits, or episode termination logic.
#
# The rule is a soft prior. HASAC fuses:
#   mu_f = (1 - lambda0) * mu_actor + lambda0 * mu_prior
#   sigma_f = (1 - lambda1) * sigma_actor + lambda1 * sigma_prior
# with current lambda0 = lambda1 = 0.5 in the project configuration.
'''
    new_header = '''# Initial rule-only FCHEV controller, adapted from fchev_rule_prior.py.
#
# The program returns Gaussian mean/std parameters before tanh for the driving
# and energy action channels. In the rule-only experiment these parameters
# directly define each action distribution; no actor or fusion is applied.
#
# The program does NOT return physical acceleration or kW and does NOT bypass
# SUMO, vehicle dynamics, battery/FCS/EDU feasibility, no-reverse handling,
# speed limits, or episode termination logic.
'''
    source = source.replace(old_header, new_header, 1)
    source = source.replace(
        "Larger mean_gain makes the prior stronger after lambda fusion, but can\n"
        "    overdrive if the actor mean already matches the target.",
        "Larger mean_gain moves the rule distribution farther toward its target\n"
        "    normalized action and may produce more extreme latent means.",
    )
    source = source.replace("# Latent Gaussian prior shaping.", "# Gaussian rule-action distribution shaping.")
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run episode-updated LLM rule control without any RL policy"
    )
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--experiment-name", default="rule_only_online_update")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--llm-model", default="auto")
    parser.add_argument("--llm-api-key-env", default="STRATA_API_KEY")
    parser.add_argument("--llm-timeout-s", type=float, default=900.0)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.llm_timeout_s <= 0:
        raise ValueError("--llm-timeout-s must be positive")
    if (
        args.llm_max_tokens is not None
        and args.llm_max_tokens not in (0, -1)
        and args.llm_max_tokens <= 0
    ):
        raise ValueError("--llm-max-tokens must be positive, 0, or -1")

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 7))
    _seed_everything(seed)
    cfg.setdefault("training", {})["algorithm"] = "rule_only"
    cfg["training"]["episodes"] = int(args.episodes)
    cfg["training"]["controller_mode"] = "episode_updated_llm_rule_only_no_rl"
    cfg["seed"] = seed
    if args.gui:
        cfg.setdefault("sumo", {})["gui"] = True
    if args.max_steps is not None:
        cfg.setdefault("sumo", {})["episode_limit_s"] = int(args.max_steps)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.experiment_name).strip("._-")
    if not label:
        raise ValueError("--experiment-name must contain a letter or number")
    run_dir = (root / args.output_root / f"rule_only_{label}_{stamp}_seed{seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg.setdefault("sumo", {})["scenario_output_dir"] = str(run_dir / "scenario")
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(cfg, stream, ensure_ascii=False, indent=2)

    online_module = import_module("llm-prior.online_rule_update")
    contract = import_module("llm-prior.contract")
    updater = online_module.StrataEpisodeRuleUpdater(
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key_env=args.llm_api_key_env,
        timeout_s=args.llm_timeout_s,
        max_tokens=args.llm_max_tokens,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        reasoning_effort="low",
        enable_thinking=True,
        mode="rule_only",
    )
    models = updater.check_server()
    current_source = _rule_only_source((
        root / "llm-prior" / "programs" / "fchev_rule_prior.py"
    ).read_text(encoding="utf-8"))
    current_provider = online_module.compile_rule_provider(current_source)
    artifacts = updater.save_initial_artifacts(run_dir, current_source)
    artifacts.update({
        "controller": "rule-only; no RL actors, critics, replay buffer, or policy updates",
        "action_generation": "sample rule Gaussian mean/std, tanh, then map to physical actions",
        "active_mask": "ignored; rule mean/std directly control both channels on every step",
        "update_frequency": "once after every episode; accepted rule applies starting next episode",
        "base_url": updater.base_url,
        "model": updater.model,
        "model_list_response": models,
        "sampling": updater.parameters,
        "conversation_mode": "stateless single-turn; one fresh user message per episode",
        "api_key_env": args.llm_api_key_env,
        "initial_program": "llm-prior/programs/fchev_rule_prior.py",
    })
    (run_dir / "online_rule_updates" / "run.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    env = FCHEVEnv(project_root=root, config=cfg, render_mode="human" if args.gui else None)
    rng = np.random.default_rng(seed)
    source_version = 0
    episode_rows = []
    start_time = time.perf_counter()
    global_step = 0
    try:
        for episode in range(1, args.episodes + 1):
            obs, reset_info = env.reset(seed=seed)
            obs = np.asarray(obs, dtype=np.float32)
            totals = {name: 0.0 for name in REWARD_COMPONENTS}
            total_return = total_driving = total_energy = total_hydrogen = 0.0
            collisions = collision_events = red_violations = speed_violations = 0
            passed = int(reset_info.get("traffic_lights_passed", 0) or 0)
            lead_passed = int(reset_info.get("lead_traffic_lights_passed", 0) or 0)
            terminal_penalty = safety_violations = clip_steps = 0
            minimum_gap = float("inf")
            trace = []
            steps = 0
            termination_reason = truncation_reason = None
            while True:
                raw_prior = current_provider.predict(obs)
                prior = contract.validate_prior_output(raw_prior, batch_size=1)
                drive = prior.driving
                energy = prior.energy
                # The pure-rule experiment uses the rule distribution directly.
                # The legacy active mask has no RL policy to fall back to.
                z_drive = float(drive.mean[0, 0]) + float(drive.std[0, 0]) * float(rng.standard_normal())
                z_energy = float(energy.mean[0, 0]) + float(energy.std[0, 0]) * float(rng.standard_normal())
                u_drive = float(np.tanh(z_drive))
                u_energy = float(np.tanh(z_energy))
                action = _map_actions(u_drive, u_energy, env.action_space)

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                driving_reward, energy_reward = _split_rewards(info)
                if not math.isclose(driving_reward + energy_reward, float(reward), abs_tol=1e-5):
                    raise RuntimeError("Driving and energy reward components do not sum to env reward")
                total_return += float(reward)
                total_driving += driving_reward
                total_energy += energy_reward
                total_hydrogen += float(info.get("hydrogen_consumed_g", 0.0) or 0.0)
                collisions += int(info.get("collision_count", 0) or 0)
                collision_events += int(bool(info.get("collision_event", False)))
                red_violations += int(bool(info.get("red_light_violation", False)))
                speed_violations += int(bool(info.get("speeding_violation", False)))
                passed = int(info.get("traffic_lights_passed", passed) or 0)
                lead_passed = int(info.get("lead_traffic_lights_passed", lead_passed) or 0)
                terminal_penalty += float(info.get("terminal_penalty", 0.0) or 0.0)
                safety_violations += int(info.get("safety_violations", 0) or 0)
                clip_steps += int(bool(info.get("constraint_clipping", False)))
                gap = _finite_or_none(info.get("following_distance_m"))
                if gap is not None:
                    minimum_gap = min(minimum_gap, gap)
                parts = info.get("reward_components", {})
                for name in REWARD_COMPONENTS:
                    totals[name] += float(parts.get(f"reward_contribution_{name}", 0.0) or 0.0)

                trace_row = {
                    "time_s": info.get("sumo_time_s"),
                    "driving_agent_reward": driving_reward,
                    "energy_agent_reward": energy_reward,
                    **{f"policy_obs_{name}": float(value)
                       for name, value in zip(OBSERVATION_NAMES, obs)},
                    "requested_acceleration_m_s2": info.get("requested_acceleration_m_s2"),
                    "action_executed_acceleration_m_s2": info.get("executed_acceleration_m_s2"),
                    "action_requested_fcs_power_raw_kw": info.get("fcs_power_requested_kw"),
                    "action_executed_fcs_power_kw": info.get("fcs_power_executed_kw"),
                    "traffic_light_state_at_crossing": info.get("traffic_light_state_at_crossing"),
                    "traffic_light_crossing": info.get("traffic_light_crossing"),
                    "red_light_violation": info.get("red_light_violation"),
                    "speeding_violation": info.get("speeding_violation"),
                    "collision_event": info.get("collision_event"),
                    "gap_overlap": info.get("gap_overlap"),
                    "terminated_reason": info.get("terminated_reason"),
                }
                trace.append(trace_row)
                steps += 1
                global_step += 1
                obs = next_obs
                termination_reason = info.get("terminated_reason")
                truncation_reason = info.get("truncated_reason")
                if terminated or truncated or (args.max_steps is not None and steps >= args.max_steps):
                    if args.max_steps is not None and steps >= args.max_steps and not (terminated or truncated):
                        truncation_reason = "max_steps_cli"
                    break

            trace_path = run_dir / f"episode_{episode:04d}_trajectory.csv"
            _write_csv(trace_path, trace, list(UPDATE_TRACE_FIELDS))
            if not math.isfinite(minimum_gap):
                minimum_gap = float("nan")
            battery = getattr(env, "battery", None)
            fcs_aging = getattr(env, "fcs_aging", None)
            edu_aging = getattr(env, "edu_aging", None)
            row = {
                "episode": episode,
                "seed": seed,
                "return": total_return,
                "driving_agent_return": total_driving,
                "energy_agent_return": total_energy,
                "steps": steps,
                "termination_reason": termination_reason,
                "truncation_reason": truncation_reason,
                "hydrogen_g": total_hydrogen,
                "collisions": collisions,
                "collision_events": collision_events,
                "red_light_violations": red_violations,
                "speeding_violations": speed_violations,
                "traffic_lights_passed": passed,
                "lead_traffic_lights_passed": lead_passed,
                "terminal_penalty": terminal_penalty,
                "safety_violations": safety_violations,
                "constraint_clipping_steps": clip_steps,
                "minimum_gap_m": minimum_gap,
                "final_soc": float(battery.state.soc),
                "final_battery_soh": float(battery.state.soh),
                "final_fcs_soh": 1.0 - float(fcs_aging.cumulative),
                "final_edu_soh": 1.0 - float(edu_aging.cumulative),
                "final_core_temperature_c": float(battery.state.core_temperature_c),
                "final_surface_temperature_c": float(battery.state.surface_temperature_c),
                "active_rule_version": source_version,
                "active_rule_sha256": updater.source_sha256(current_source),
                "scenario_seed": reset_info.get("scenario_seed"),
                "signal_initial_conditions_json": json.dumps(
                    reset_info.get("signal_initial_conditions", []), separators=(",", ":")
                ),
                "lead_stop_plan_json": json.dumps(
                    reset_info.get("lead_stop_plan", []), separators=(",", ":")
                ),
            }
            row.update({f"reward_component_{name}": value for name, value in totals.items()})
            row["reward_component_total"] = sum(totals.values())
            _append_csv(run_dir / "episodes.csv", row)
            episode_rows.append(row)

            context = online_module.build_rule_only_episode_context(
                episode_rows[:-1], row, trace
            )
            previous_version = source_version
            proposal = updater.request_update(current_rule_source=current_source, context=context)
            update_error = proposal.get("response_error")
            accepted_rule = False
            if not update_error:
                decision = proposal["decision"]
                proposed_source = proposal["candidate_source"]
                proposed_provider = proposal["candidate_provider"]
                if proposal.get("candidate_error"):
                    update_error = proposal["candidate_error"]
                elif decision["change_rule"] and proposed_provider is not None and (
                    updater.source_sha256(proposed_source) != updater.source_sha256(current_source)
                ):
                    current_source = proposed_source
                    current_provider = proposed_provider
                    source_version += 1
                    accepted_rule = True
            record_path = updater.write_episode_record(
                run_dir=run_dir,
                episode=episode,
                context=context,
                result=proposal,
                previous_version=previous_version,
                current_version=source_version,
                current_source=current_source,
                accepted_rule=accepted_rule,
                error=update_error,
            )
            print(
                f"Rule-only episode {episode}/{args.episodes}: return={total_return:.3f}, "
                f"driving={total_driving:.3f}, energy={total_energy:.3f}, steps={steps}, "
                f"end={termination_reason or truncation_reason or 'unknown'}, "
                f"rule v{previous_version}->v{source_version}; record={record_path.name}",
                flush=True,
            )
            _plot_results(run_dir)
    finally:
        env.close()

    elapsed = time.perf_counter() - start_time
    reasons = Counter(row.get("termination_reason") or row.get("truncation_reason") or "none"
                      for row in episode_rows)
    summary = {
        "controller": "LLM-generated rule only; no RL algorithm or policy optimization",
        "episodes": args.episodes,
        "seed": seed,
        "environment_steps": global_step,
        "elapsed_seconds": elapsed,
        "mean_episode_return": float(np.mean([row["return"] for row in episode_rows])),
        "mean_episode_steps": float(np.mean([row["steps"] for row in episode_rows])),
        "mean_hydrogen_g": float(np.mean([row["hydrogen_g"] for row in episode_rows])),
        "mean_traffic_lights_passed": float(np.mean([row["traffic_lights_passed"] for row in episode_rows])),
        "termination_reason_counts": dict(reasons),
        "final_rule_version": source_version,
        "final_rule_sha256": updater.source_sha256(current_source),
        "rule_update_records": str(run_dir / "online_rule_updates"),
        "plots": {
            "episode_reward": str(run_dir / "episode_reward.png"),
            "episode_metrics": str(run_dir / "episode_metrics.png"),
            "reward_components": str(run_dir / "reward_components.png"),
        },
        "run_directory": str(run_dir),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
