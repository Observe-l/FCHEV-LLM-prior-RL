from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithm.networks import Critic, GaussianActor
from algorithm.replay_buffer import ReplayBuffer


class SAC:
    def __init__(self, state_dim, action_dim, config, seed=0):
        self.cfg = config
        self.device = torch.device(
            "cuda" if config.get("device", "auto") == "auto" and torch.cuda.is_available()
            else ("cpu" if config.get("device", "auto") == "auto" else config["device"])
        )
        hidden = tuple(config.get("hidden_layers", [256, 128, 64]))
        self.actor = GaussianActor(state_dim, action_dim, hidden).to(self.device)
        self.q1 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.q2 = Critic(state_dim, action_dim, hidden).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_learning_rate", 3e-4))
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=config.get("critic_learning_rate", 3e-4),
        )
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.get("temperature_learning_rate", 3e-4))
        self.target_entropy = -float(action_dim)
        self.gamma = float(config.get("gamma", 0.9999))
        self.tau = float(config.get("tau", 0.005))
        self.replay = ReplayBuffer(
            state_dim, action_dim, config.get("replay_capacity", 100000), seed,
            device=self.device,
        )
        self.update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
        a, _ = self.actor.sample(s, deterministic=deterministic)
        return a.cpu().numpy()[0]

    def update(self, batch_size):
        s, a, r, ns, done = self.replay.sample(batch_size)
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(ns)
            q_target = torch.minimum(
                self.target_q1(ns, next_action), self.target_q2(ns, next_action)
            ) - self.alpha * next_logp
            target = r + self.gamma * (1.0 - done) * q_target
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.critic_opt.step()

        policy_action, logp = self.actor.sample(s)
        q_pi = torch.minimum(self.q1(s, policy_action), self.q2(s, policy_action))
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.cpu()),
        }

    def _soft_update(self, source, target):
        with torch.no_grad():
            for p, tp in zip(source.parameters(), target.parameters()):
                tp.mul_(1.0 - self.tau).add_(p, alpha=self.tau)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def checkpoint_state(self):
        return {
            "algorithm": "sac", "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(), "alpha_opt": self.alpha_opt.state_dict(),
            "update_count": self.update_count, "config": self.cfg,
        }

    def load(self, path, load_optimizers=False):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.load_checkpoint_state(ckpt, load_optimizers=load_optimizers)

    def load_checkpoint_state(self, ckpt, load_optimizers=False):
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.target_q1.load_state_dict(ckpt["target_q1"])
        self.target_q2.load_state_dict(ckpt["target_q2"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
        self.update_count = int(ckpt.get("update_count", 0))
        if load_optimizers:
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.critic_opt.load_state_dict(ckpt["critic_opt"])
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
