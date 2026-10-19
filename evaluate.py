from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import torch

from algorithm.sac import SAC
from algorithm.td3 import TD3
from environment.fchev_env import FCHEVEnv, load_config
from train import evaluate, seed_everything


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a saved SAC/TD3 FCHEV policy")
    parser.add_argument("checkpoint")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold-gui", action="store_true", help="Keep the SUMO-GUI open after evaluation")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    if args.gui:
        cfg["sumo"]["gui"] = True
    if args.max_steps is not None:
        cfg["sumo"]["episode_limit_s"] = args.max_steps
    seed_everything(args.seed)
    env = FCHEVEnv(project_root=root, config=cfg, render_mode="human" if args.gui else None)
    algo_cfg = cfg["training"]
    name = Path(args.checkpoint).name.lower()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    algorithm = ckpt.get("algorithm", "sac")
    cls = SAC if algorithm == "sac" else TD3
    agent = cls(env.observation_space.shape[0], env.action_space.shape[0], algo_cfg, args.seed)
    agent.load(args.checkpoint)
    try:
        rows = evaluate(
            env, agent, cfg, args.episodes, args.seed, cfg["vehicle"], args.max_steps
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out = Path(args.checkpoint).resolve().parent / f"evaluation_seed{args.seed}_{stamp}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for row in rows:
            print(row)
        print(f"Saved evaluation to {out}")
        if args.gui and args.hold_gui:
            input("SUMO-GUI is paused at the final state. Press Enter here to close it. ")
    finally:
        env.close()


if __name__ == "__main__":
    main()
