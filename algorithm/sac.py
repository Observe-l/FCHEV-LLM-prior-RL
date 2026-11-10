from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithm.networks import Critic, GaussianActor
from algorithm.replay_buffer import ReplayBuffer


class SAC:
    def __init__(self, state_dim, action_dim, config, seed=0, *,
                 prior_provider=None, prior_config=None, agent_name=None):
        self.cfg = config
        self.prior_cfg = dict(prior_config or config.get("llm_prior", {}))
        self.prior_provider = prior_provider
        self.agent_name = agent_name
        if self.prior_provider is not None and self.agent_name not in ("driving", "energy"):
            raise ValueError("A rule-prior SAC agent must be named 'driving' or 'energy'")
        self.lambda_mean, self.lambda_std = self._agent_lambda_weights()
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
            auxiliary_dim=15 if self.prior_provider is not None else 0,
        )
        self.update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def _agent_lambda_weights(self):
        if self.agent_name is None:
            return 0.0, 0.0
        overrides = self.prior_cfg.get("lambda_by_agent", {})
        override = overrides.get(self.agent_name, {}) if isinstance(overrides, dict) else {}
        if isinstance(override, (int, float)):
            lambda_mean = lambda_std = float(override)
        elif isinstance(override, dict):
            lambda_mean = float(override.get("lambda_0", self.prior_cfg.get("lambda_0", 0.5)))
            lambda_std = float(override.get("lambda_1", self.prior_cfg.get("lambda_1", 0.5)))
        else:
            raise TypeError(f"llm_prior.lambda_by_agent.{self.agent_name} must be a mapping or number")
        if not 0.0 <= lambda_mean <= 1.0 or not 0.0 <= lambda_std <= 1.0:
            raise ValueError("SAC prior mean/std lambda values must be in [0, 1]")
        return lambda_mean, lambda_std

    def _predict_prior(self, full_observation):
        if self.prior_provider is None:
            return None
        if full_observation is None:
            raise ValueError("full_observation is required when the SAC rule prior is enabled")
        if torch.is_tensor(full_observation):
            full_numpy = full_observation.detach().cpu().numpy()
        else:
            full_numpy = np.asarray(full_observation, dtype=np.float32)
        if full_numpy.ndim == 1:
            full_numpy = full_numpy.reshape(1, -1)
        if full_numpy.ndim != 2 or full_numpy.shape[1] != 15:
            raise ValueError("Rule prior input must be the raw full observation with shape (batch, 15)")
        contract = import_module("llm-prior.contract")
        return contract.validate_prior_output(
            self.prior_provider.predict(full_numpy), batch_size=full_numpy.shape[0]
        )

    def _policy_parameter_details(self, state, full_observation=None, prior_output=None):
        raw_mean, log_std = self.actor(state)
        raw_std = log_std.exp()
        mean, std = raw_mean, raw_std
        prior_mean, prior_std = raw_mean, raw_std
        active = torch.zeros_like(raw_mean)
        if self.prior_provider is None:
            return {
                "mean": mean, "std": std, "raw_mean": raw_mean,
                "raw_std": raw_std, "prior_mean": prior_mean,
                "prior_std": prior_std, "prior_active": active,
            }
        if prior_output is None:
            prior_output = self._predict_prior(full_observation)
        prior = prior_output.for_agent(self.agent_name)
        prior_mean = torch.as_tensor(prior.mean, dtype=raw_mean.dtype, device=raw_mean.device)
        prior_std = torch.as_tensor(prior.std, dtype=raw_std.dtype, device=raw_std.device)
        active = torch.as_tensor(prior.active, dtype=raw_mean.dtype, device=raw_mean.device)
        mean_weight = active * self.lambda_mean
        std_weight = active * self.lambda_std
        mean = (1.0 - mean_weight) * raw_mean + mean_weight * prior_mean
        std = (1.0 - std_weight) * raw_std + std_weight * prior_std
        return {
            "mean": mean, "std": std, "raw_mean": raw_mean,
            "raw_std": raw_std, "prior_mean": prior_mean,
            "prior_std": prior_std, "prior_active": active,
        }

    @torch.no_grad()
    def select_action(self, state, deterministic=False, full_observation=None,
                      prior_output=None):
        s = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device).reshape(1, -1)
        parameters = self._policy_parameter_details(
            s, full_observation=full_observation, prior_output=prior_output
        )
        a, _ = GaussianActor.sample_distribution(
            parameters["mean"], parameters["std"],
            deterministic=deterministic,
            min_std=1e-6 if self.prior_provider is not None else None,
        )
        return a.cpu().numpy()[0]

    def update(self, batch_size):
        if self.prior_provider is not None:
            s, a, r, ns, done, full_obs, next_full_obs = (
                self.replay.sample_with_auxiliary(batch_size)
            )
            current_prior = self._predict_prior(full_obs)
            next_prior = self._predict_prior(next_full_obs)
        else:
            s, a, r, ns, done = self.replay.sample(batch_size)
            full_obs = next_full_obs = current_prior = next_prior = None
        with torch.no_grad():
            next_parameters = self._policy_parameter_details(
                ns, full_observation=next_full_obs, prior_output=next_prior
            )
            next_action, next_logp = GaussianActor.sample_distribution(
                next_parameters["mean"], next_parameters["std"],
                min_std=1e-6 if self.prior_provider is not None else None,
            )
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

        policy_parameters = self._policy_parameter_details(
            s, full_observation=full_obs, prior_output=current_prior
        )
        policy_action, logp = GaussianActor.sample_distribution(
            policy_parameters["mean"], policy_parameters["std"],
            min_std=1e-6 if self.prior_provider is not None else None,
        )
        q1_pi = self.q1(s, policy_action)
        q2_pi = self.q2(s, policy_action)
        q_pi = torch.minimum(q1_pi, q2_pi)
        actor_loss = (self.alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        self.update_count += 1
        result = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_q1_replay_mean": float(q1.detach().mean().cpu()),
            "critic_q2_replay_mean": float(q2.detach().mean().cpu()),
            "critic_target_mean": float(target.detach().mean().cpu()),
            "critic_target_state_std": float(target.detach().std(unbiased=False).cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.cpu()),
        }
        if self.agent_name is not None:
            result[f"{self.agent_name}_critic_loss"] = result["critic_loss"]
            result[f"{self.agent_name}_actor_loss"] = result["actor_loss"]
            result.update({
                f"{self.agent_name}_q_policy": float(q_pi.detach().mean().cpu()),
                f"{self.agent_name}_q1_policy": float(q1_pi.detach().mean().cpu()),
                f"{self.agent_name}_q2_policy": float(q2_pi.detach().mean().cpu()),
                f"{self.agent_name}_log_prob": float(logp.detach().mean().cpu()),
                f"{self.agent_name}_entropy": float((-logp.detach()).mean().cpu()),
                f"{self.agent_name}_entropy_term": float((self.alpha * logp.detach()).mean().cpu()),
                f"{self.agent_name}_negative_q_term": float((-q_pi.detach()).mean().cpu()),
                f"{self.agent_name}_raw_mean": float(policy_parameters["raw_mean"].detach().mean().cpu()),
                f"{self.agent_name}_raw_mean_state_std": float(policy_parameters["raw_mean"].detach().std(unbiased=False).cpu()),
                f"{self.agent_name}_raw_std": float(policy_parameters["raw_std"].detach().mean().cpu()),
                f"{self.agent_name}_raw_std_state_std": float(policy_parameters["raw_std"].detach().std(unbiased=False).cpu()),
                f"{self.agent_name}_fused_mean": float(policy_parameters["mean"].detach().mean().cpu()),
                f"{self.agent_name}_fused_mean_state_std": float(policy_parameters["mean"].detach().std(unbiased=False).cpu()),
                f"{self.agent_name}_fused_std": float(policy_parameters["std"].detach().mean().cpu()),
                f"{self.agent_name}_fused_std_state_std": float(policy_parameters["std"].detach().std(unbiased=False).cpu()),
                f"{self.agent_name}_sampled_action": float(policy_action.detach().mean().cpu()),
                f"{self.agent_name}_sampled_action_state_std": float(policy_action.detach().std(unbiased=False).cpu()),
                f"{self.agent_name}_actor_grad_norm": float(torch.as_tensor(actor_grad_norm).detach().cpu()),
            })
        return result

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
