from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithm.networks import Critic, DeterministicActor
from algorithm.replay_buffer import ReplayBuffer


class TD3:
    def __init__(self, state_dim, action_dim, config, seed=0):
        self.cfg = config
        self.device = torch.device(
            "cuda" if config.get("device", "auto") == "auto" and torch.cuda.is_available()
            else ("cpu" if config.get("device", "auto") == "auto" else config["device"])
        )
        hidden = tuple(config.get("hidden_layers", [256, 128, 64]))
        self.actor = DeterministicActor(state_dim, action_dim, hidden).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.q1 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q2 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_learning_rate", 3e-4))
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.get("critic_learning_rate", 3e-4),
        )
        self.gamma = float(config.get("gamma", 0.95))
        self.tau = float(config.get("tau", 0.005))
        self.policy_noise = float(config.get("td3_policy_noise", 0.2))
        self.noise_clip = float(config.get("td3_noise_clip", 0.5))
        self.policy_delay = int(config.get("td3_policy_delay", 2))
        self.replay = ReplayBuffer(
            state_dim, action_dim, config.get("replay_capacity", 100000), seed,
            device=self.device,
        )
        self.update_count = 0

    @torch.no_grad()
    def select_action(self, state, deterministic=True):
        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
        return self.actor(s).cpu().numpy()[0]

    def update(self, batch_size):
        s, a, r, ns, done = self.replay.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(ns) + noise).clamp(-1.0, 1.0)
            next_q = torch.minimum(self.target_q1(ns, next_action), self.target_q2(ns, next_action))
            target = r + self.gamma * (1.0 - done) * next_q
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.critic_opt.step()
        self.update_count += 1
        result = {"critic_loss": float(critic_loss.detach().cpu())}
        if self.update_count % self.policy_delay == 0:
            actor_loss = -self.q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)
            result["actor_loss"] = float(actor_loss.detach().cpu())
        return result

    def _soft_update(self, source, target):
        with torch.no_grad():
            for p, tp in zip(source.parameters(), target.parameters()):
                tp.mul_(1.0 - self.tau).add_(p, alpha=self.tau)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "algorithm": "td3", "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
            "update_count": self.update_count, "config": self.cfg,
        }, path)

    def load(self, path, load_optimizers=False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        for name in ("actor", "actor_target", "q1", "q2", "target_q1", "target_q2"):
            getattr(self, name).load_state_dict(ckpt[name])
        self.update_count = int(ckpt.get("update_count", 0))
        if load_optimizers:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
