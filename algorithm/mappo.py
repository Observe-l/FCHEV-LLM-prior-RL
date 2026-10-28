from __future__ import annotations

from importlib import import_module
import re

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from algorithm.networks import GaussianActor, mlp


class ValueNetwork(nn.Module):
    """Centralized team-value function over the complete environment state."""

    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        self.net = mlp(state_dim, 1, hidden_layers)

    def forward(self, state):
        return self.net(state)

    @torch.no_grad()
    def rescale_output(self, old_mean, old_scale, new_mean, new_scale):
        """Keep unnormalized value predictions fixed after ValueNorm changes."""
        output = self.net[-1]
        output.weight.mul_(old_scale / new_scale)
        output.bias.mul_(old_scale).add_(old_mean - new_mean).div_(new_scale)


class RunningValueNorm:
    """Running return normalization used by the upstream MAPPO defaults."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.m2 = float(epsilon)
        self.count = float(epsilon)

    @property
    def variance(self):
        return max(self.m2 / max(self.count, 1e-12), 1e-6)

    @property
    def scale(self):
        return float(np.sqrt(self.variance + 1e-5))

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not values.size:
            return self.mean, self.scale, self.mean, self.scale
        old_mean, old_scale = self.mean, self.scale
        batch_count = float(values.size)
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_var * batch_count + delta * delta * self.count * batch_count / total
        self.count = total
        return old_mean, old_scale, self.mean, self.scale

    def normalize(self, tensor):
        return (tensor - self.mean) / self.scale

    def denormalize(self, tensor):
        return tensor * self.scale + self.mean

    def state_dict(self):
        return {"mean": self.mean, "m2": self.m2, "count": self.count}

    def load_state_dict(self, state):
        self.mean = float(state["mean"])
        self.m2 = float(state["m2"])
        self.count = float(state["count"])


class MAPPO:
    """Two-agent MAPPO with decentralized Gaussian actors and a central critic.

    The driving and energy policies retain their existing local observation
    vectors. Both use the same cooperative reward, and a centralized value
    function receives the complete normalized 15-element state. Rollouts are
    on-policy: an episode is collected, GAE is computed, then PPO minibatch
    updates are applied before the next episode starts.

    Each actor samples a Gaussian action in normalized command coordinates.
    ``mappo_action_transform`` selects either tanh squashing (matching HASAC)
    or direct Gaussian sampling followed by clipping. The normalized command
    is then linearly mapped to physical acceleration and FCS power by train.py.
    """

    checkpoint_algorithm = "mappo"
    driving_state_dim = 11
    energy_state_dim = 6
    global_state_dim = 15
    action_dim_per_agent = 1
    driving_state_features = (
        "ego_acceleration_m_s2", "ego_speed_m_s", "lead_acceleration_m_s2",
        "lead_speed_m_s", "following_distance_m", "next_signal_state_normalized",
        "next_signal_distance_m", "signal_phase_remaining_s",
        "current_speed_limit_m_s", "next_speed_limit_m_s",
        "distance_to_next_speed_region_m",
    )
    energy_state_features = (
        "ego_acceleration_m_s2", "ego_speed_m_s", "soc", "battery_soh",
        "fcs_soh", "edu_soh",
    )

    def __init__(self, config, seed=0, prior_provider=None):
        self.cfg = dict(config)
        requested_device = config.get("device", "auto")
        self.device = torch.device(
            "cuda" if requested_device == "auto" and torch.cuda.is_available()
            else ("cpu" if requested_device == "auto" else requested_device)
        )
        hidden = tuple(config.get("hidden_layers", [256, 128, 64]))
        self.actors = nn.ModuleDict({
            "driving": GaussianActor(self.driving_state_dim, 1, hidden),
            "energy": GaussianActor(self.energy_state_dim, 1, hidden),
        }).to(self.device)
        self.critic = ValueNetwork(self.global_state_dim, hidden).to(self.device)
        self.value_normalizer = RunningValueNorm()

        actor_lr = float(config.get("actor_learning_rate", 3e-4))
        critic_lr = float(config.get("critic_learning_rate", actor_lr))
        adam_eps = float(config.get("mappo_adam_eps", 1e-5))
        self.actor_optimizers = {
            name: torch.optim.Adam(actor.parameters(), lr=actor_lr, eps=adam_eps)
            for name, actor in self.actors.items()
        }
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, eps=adam_eps
        )

        self.gamma = float(config.get("gamma", 0.99))
        self.gae_lambda = float(config.get("mappo_gae_lambda", 0.95))
        self.clip_param = float(config.get("mappo_clip_param", 0.2))
        self.ppo_epochs = int(config.get("mappo_ppo_epochs", 10))
        self.num_minibatches = int(config.get("mappo_num_minibatches", 8))
        self.value_loss_coef = float(config.get("mappo_value_loss_coef", 1.0))
        self.entropy_coef = float(config.get("mappo_entropy_coef", 0.01))
        self.max_grad_norm = float(config.get("mappo_max_grad_norm", 10.0))
        self.value_clip = float(config.get("mappo_value_clip", 0.2))
        self.huber_delta = float(config.get("mappo_huber_delta", 10.0))
        self.min_policy_std = float(config.get("mappo_min_policy_std", 1e-6))
        self.action_transform = str(
            config.get("mappo_action_transform", "gaussian_clip")
        ).strip().lower()
        self.action_clip_low = float(config.get("mappo_action_clip_low", -1.0))
        self.action_clip_high = float(config.get("mappo_action_clip_high", 1.0))
        self.normalize_advantage = bool(config.get("mappo_normalize_advantage", True))
        if self.ppo_epochs <= 0 or self.num_minibatches <= 0:
            raise ValueError("MAPPO PPO epochs and minibatches must be positive")
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("MAPPO gamma and GAE lambda must be in [0, 1]")
        if self.min_policy_std <= 0.0:
            raise ValueError("mappo_min_policy_std must be positive")
        if not self.action_clip_low < self.action_clip_high:
            raise ValueError("mappo_action_clip_low must be less than mappo_action_clip_high")
        if self.action_transform not in {"gaussian_clip", "tanh"}:
            raise ValueError(
                "mappo_action_transform must be 'gaussian_clip' or 'tanh'"
            )

        self.prior_cfg = dict(config.get("llm_prior", {}))
        self._external_prior_provider = prior_provider is not None
        self.prior_provider = prior_provider or self._build_prior_provider()
        self.lambda_mean = float(self.prior_cfg.get("lambda_0", 0.5))
        self.lambda_std = float(self.prior_cfg.get("lambda_1", 0.5))
        self._validate_prior_lambdas()

        self.rng = np.random.default_rng(int(seed) + 104729)
        self.rollout = []
        self.update_count = 0

    def _action_log_prob_and_entropy(self, distribution, latent):
        """Score a sampled action under the selected normalized action map.

        For tanh, apply the change-of-variables correction used by HASAC.
        The stored latent remains the pre-tanh Gaussian sample, so PPO can
        recompute the same transformed-action likelihood during its updates.
        """
        log_prob = distribution.log_prob(latent).sum(dim=-1, keepdim=True)
        if self.action_transform == "tanh":
            action = torch.tanh(latent)
            log_det = torch.log(1.0 - action.pow(2) + 1e-6).sum(
                dim=-1, keepdim=True
            )
            log_prob = log_prob - log_det
            # Monte Carlo estimate of the squashed policy's differential entropy.
            entropy = -log_prob.squeeze(-1)
        else:
            entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def _current_policy_entropy(self, distribution):
        """Estimate entropy using samples from the current policy itself."""
        if self.action_transform == "tanh":
            # Reusing a stored rollout action here would evaluate its likelihood
            # under the *updated* policy and can drive the mean away from the
            # data, making the entropy estimate and its gradient explode.
            fresh_latent = distribution.rsample()
            _, entropy = self._action_log_prob_and_entropy(
                distribution, fresh_latent
            )
            return entropy
        return distribution.entropy().sum(dim=-1)

    def _validate_prior_lambdas(self):
        if not 0.0 <= self.lambda_mean <= 1.0:
            raise ValueError("llm_prior.lambda_0 must be in [0, 1]")
        if not 0.0 <= self.lambda_std <= 1.0:
            raise ValueError("llm_prior.lambda_1 must be in [0, 1]")

    def _build_prior_provider(self):
        if not self.prior_cfg.get("enabled", False):
            return None
        program_name = self.prior_cfg.get(
            "program_name", self.prior_cfg.get("type", "speed_limit_rule")
        )
        if not isinstance(program_name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", program_name
        ):
            raise ValueError("llm_prior.program_name must be a Python module name")
        try:
            module = import_module(f"llm-prior.programs.{program_name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"llm-prior.programs.{program_name}":
                raise ValueError(f"Unknown prior program {program_name!r}") from exc
            raise
        provider = (
            module.build_prior(dict(self.prior_cfg))
            if callable(getattr(module, "build_prior", None)) else module
        )
        if not callable(getattr(provider, "predict", None)):
            raise TypeError(
                f"Prior program {program_name!r} must expose predict(observation)"
            )
        return provider

    def _predict_prior(self, full_observation):
        if self.prior_provider is None:
            return None
        if full_observation is None:
            raise ValueError("full_observation is required when the prior is enabled")
        full_numpy = np.asarray(full_observation, dtype=np.float32)
        if full_numpy.ndim == 1:
            full_numpy = full_numpy.reshape(1, -1)
        if full_numpy.shape != (1, self.global_state_dim):
            raise ValueError(
                f"Prior observation must have shape (1, {self.global_state_dim}); "
                f"received {full_numpy.shape}"
            )
        prior_output = self.prior_provider.predict(full_numpy)
        contract = import_module("llm-prior.contract")
        return contract.validate_prior_output(prior_output, batch_size=1)

    def _agent_lambda_weights(self, agent_name):
        overrides = self.prior_cfg.get("lambda_by_agent", {})
        value = overrides.get(agent_name, {}) if isinstance(overrides, dict) else {}
        if isinstance(value, (int, float)):
            lambda_mean = lambda_std = float(value)
        elif isinstance(value, dict):
            lambda_mean = float(value.get("lambda_0", self.lambda_mean))
            lambda_std = float(value.get("lambda_1", self.lambda_std))
        else:
            raise TypeError(f"lambda_by_agent.{agent_name} must be a number or mapping")
        if not 0.0 <= lambda_mean <= 1.0 or not 0.0 <= lambda_std <= 1.0:
            raise ValueError(f"{agent_name} prior lambdas must be in [0, 1]")
        return lambda_mean, lambda_std

    def _fuse_parameters(self, actor, state, agent_name, prior_data=None):
        raw_mean, log_std = actor(state)
        raw_std = log_std.exp()
        if prior_data is None:
            active = torch.zeros_like(raw_mean)
            prior_mean = torch.zeros_like(raw_mean)
            prior_std = torch.zeros_like(raw_std)
        else:
            prior_mean, prior_std, active = prior_data
            prior_mean = prior_mean.to(device=self.device, dtype=raw_mean.dtype)
            prior_std = prior_std.to(device=self.device, dtype=raw_std.dtype)
            active = active.to(device=self.device, dtype=raw_mean.dtype)
        lambda_mean, lambda_std = self._agent_lambda_weights(agent_name)
        mean_weight = active * lambda_mean
        std_weight = active * lambda_std
        mean = (1.0 - mean_weight) * raw_mean + mean_weight * prior_mean
        std = (1.0 - std_weight) * raw_std + std_weight * prior_std
        return mean, std.clamp_min(self.min_policy_std), raw_mean, raw_std

    @staticmethod
    def _prior_data(prior_output, agent_name):
        if prior_output is None:
            return None
        prior = prior_output.for_agent(agent_name)
        return (
            torch.as_tensor(prior.mean, dtype=torch.float32),
            torch.as_tensor(prior.std, dtype=torch.float32),
            torch.as_tensor(prior.active, dtype=torch.float32),
        )

    @torch.no_grad()
    def select_actions(self, driving_state, energy_state, global_state,
                       deterministic=False, full_observation=None):
        states = {
            "driving": np.asarray(driving_state, dtype=np.float32).reshape(1, -1),
            "energy": np.asarray(energy_state, dtype=np.float32).reshape(1, -1),
        }
        global_tensor = torch.as_tensor(
            np.asarray(global_state, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32, device=self.device,
        )
        prior_output = self._predict_prior(full_observation)
        record = {
            "global_state": global_tensor.cpu().numpy()[0].copy(),
            "value": float(
                self.value_normalizer.denormalize(self.critic(global_tensor)).item()
            ),
            "agents": {},
        }
        actions = {}
        for name in ("driving", "energy"):
            state = torch.as_tensor(states[name], dtype=torch.float32, device=self.device)
            prior_data = self._prior_data(prior_output, name)
            mean, std, raw_mean, raw_std = self._fuse_parameters(
                self.actors[name], state, name, prior_data
            )
            distribution = Normal(mean, std)
            latent = mean if deterministic else distribution.sample()
            if self.action_transform == "tanh":
                action = torch.tanh(latent)
            else:
                action = latent.clamp(self.action_clip_low, self.action_clip_high)
            log_prob, entropy = self._action_log_prob_and_entropy(
                distribution, latent
            )

            if prior_data is None:
                prior_mean = np.zeros((1,), dtype=np.float32)
                prior_std = np.zeros((1,), dtype=np.float32)
                active = np.zeros((1,), dtype=np.float32)
            else:
                prior_mean = prior_data[0].cpu().numpy()[0].copy()
                prior_std = prior_data[1].cpu().numpy()[0].copy()
                active = prior_data[2].cpu().numpy()[0].copy()
            record["agents"][name] = {
                "state": states[name][0].copy(),
                "latent": latent.cpu().numpy()[0].copy(),
                "policy_action": action.cpu().numpy()[0].copy(),
                # Kept as a compatibility alias for existing trace readers.
                "clipped_action": action.cpu().numpy()[0].copy(),
                "old_log_prob": float(log_prob.item()),
                "entropy": float(entropy.reshape(-1)[0].item()),
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "prior_active": active,
                "raw_mean": raw_mean.cpu().numpy()[0].copy(),
                "raw_std": raw_std.cpu().numpy()[0].copy(),
                "fused_mean": mean.cpu().numpy()[0].copy(),
                "fused_std": std.cpu().numpy()[0].copy(),
            }
            actions[name] = action.cpu().numpy()[0].astype(np.float32)
        return actions["driving"], actions["energy"], record, prior_output

    def store_transition(self, policy_record, team_reward, next_global_state,
                         terminated, truncated):
        if not np.isfinite(team_reward):
            raise ValueError("MAPPO reward must be finite")
        self.rollout.append({
            "record": policy_record,
            "reward": float(team_reward),
            "next_global_state": np.asarray(next_global_state, dtype=np.float32).copy(),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })

    def _prepare_rollout(self):
        if not self.rollout:
            return None
        global_states = np.stack([item["record"]["global_state"] for item in self.rollout])
        next_global_states = np.stack([item["next_global_state"] for item in self.rollout])
        rewards = np.asarray([item["reward"] for item in self.rollout], dtype=np.float32)
        values = np.asarray([item["record"]["value"] for item in self.rollout], dtype=np.float32)
        terminated = np.asarray([item["terminated"] for item in self.rollout], dtype=np.float32)
        done = np.asarray([
            item["terminated"] or item["truncated"] for item in self.rollout
        ], dtype=np.float32)
        with torch.no_grad():
            next_values_normalized = self.critic(torch.as_tensor(
                next_global_states, dtype=torch.float32, device=self.device
            ))
            next_values = self.value_normalizer.denormalize(
                next_values_normalized
            ).squeeze(-1).cpu().numpy()

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            bootstrap_mask = 1.0 - terminated[index]
            continuation_mask = 1.0 - done[index]
            delta = rewards[index] + self.gamma * next_values[index] * bootstrap_mask - values[index]
            gae = delta + self.gamma * self.gae_lambda * continuation_mask * gae
            advantages[index] = gae
        returns = advantages + values
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (
                (advantages - advantages.mean())
                / (advantages.std(ddof=0) + 1e-5)
            )

        prepared = {
            "global_state": global_states,
            "value_old": values,
            "return": returns,
            "advantage": advantages,
            "reward": rewards,
        }
        for name in ("driving", "energy"):
            records = [item["record"]["agents"][name] for item in self.rollout]
            for field, key in (("state", "state"), ("latent", "latent"),
                               ("old_log_prob", "old_log_prob"),
                               ("prior_mean", "prior_mean"),
                               ("prior_std", "prior_std"),
                               ("prior_active", "prior_active")):
                prepared[f"{name}_{field}"] = np.asarray(
                    [record[key] for record in records], dtype=np.float32
                )
        return prepared

    def update(self):
        """Apply on-policy clipped PPO updates to the completed episode rollout."""
        batch = self._prepare_rollout()
        if batch is None:
            return {}
        count = len(batch["return"])
        tensor = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
        }
        old_mean, old_scale, new_mean, new_scale = self.value_normalizer.update(
            batch["return"]
        )
        self.critic.rescale_output(old_mean, old_scale, new_mean, new_scale)
        tensor["return_normalized"] = self.value_normalizer.normalize(tensor["return"])
        tensor["value_old_normalized"] = self.value_normalizer.normalize(
            tensor["value_old"]
        )
        metrics = {
            "value_loss": 0.0, "driving_policy_loss": 0.0,
            "energy_policy_loss": 0.0, "driving_entropy": 0.0,
            "energy_entropy": 0.0, "driving_ratio": 0.0,
            "energy_ratio": 0.0, "driving_clip_fraction": 0.0,
            "energy_clip_fraction": 0.0,
            "driving_actor_grad_norm": 0.0,
            "energy_actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
        }
        num_updates = 0
        for _ in range(self.ppo_epochs):
            permutation = self.rng.permutation(count)
            minibatches = [indices for indices in np.array_split(
                permutation, min(self.num_minibatches, count)
            ) if len(indices)]
            for indices_np in minibatches:
                indices = torch.as_tensor(indices_np, dtype=torch.long, device=self.device)
                advantages = tensor["advantage"][indices].unsqueeze(-1)
                for name in ("driving", "energy"):
                    actor = self.actors[name]
                    state = tensor[f"{name}_state"][indices]
                    prior_data = (
                        tensor[f"{name}_prior_mean"][indices],
                        tensor[f"{name}_prior_std"][indices],
                        tensor[f"{name}_prior_active"][indices],
                    )
                    mean, std, _, _ = self._fuse_parameters(actor, state, name, prior_data)
                    distribution = Normal(mean, std)
                    latent = tensor[f"{name}_latent"][indices]
                    log_prob, _ = self._action_log_prob_and_entropy(
                        distribution, latent
                    )
                    entropy = self._current_policy_entropy(distribution).mean()
                    log_ratio = (log_prob - tensor[f"{name}_old_log_prob"][indices].unsqueeze(-1))
                    ratio = torch.exp(log_ratio.clamp(-20.0, 20.0))
                    surrogate = ratio * advantages
                    clipped_surrogate = ratio.clamp(
                        1.0 - self.clip_param, 1.0 + self.clip_param
                    ) * advantages
                    policy_loss = -torch.minimum(surrogate, clipped_surrogate).mean()
                    actor_loss = policy_loss - self.entropy_coef * entropy
                    optimizer = self.actor_optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        actor.parameters(), self.max_grad_norm
                    )
                    optimizer.step()

                    metrics[f"{name}_policy_loss"] += float(policy_loss.detach())
                    metrics[f"{name}_entropy"] += float(entropy.detach())
                    metrics[f"{name}_ratio"] += float(ratio.detach().mean())
                    metrics[f"{name}_clip_fraction"] += float(
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().detach()
                    )
                    metrics[f"{name}_actor_grad_norm"] += float(actor_grad_norm)

                global_state = tensor["global_state"][indices]
                value = self.critic(global_state)
                value_old = tensor["value_old_normalized"][indices].unsqueeze(-1)
                target_return = tensor["return_normalized"][indices].unsqueeze(-1)
                clipped_value = value_old + (value - value_old).clamp(
                    -self.value_clip, self.value_clip
                )
                loss_original = F.smooth_l1_loss(
                    value, target_return, reduction="none", beta=self.huber_delta
                )
                loss_clipped = F.smooth_l1_loss(
                    clipped_value, target_return, reduction="none", beta=self.huber_delta
                )
                value_loss = torch.maximum(loss_original, loss_clipped).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.value_loss_coef * value_loss).backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()
                metrics["value_loss"] += float(value_loss.detach())
                metrics["critic_grad_norm"] += float(critic_grad_norm)
                num_updates += 1

        for key in metrics:
            metrics[key] /= max(num_updates, 1)
        metrics["rollout_steps"] = count
        metrics["team_reward_mean"] = float(batch["reward"].mean())
        metrics["team_return"] = float(batch["reward"].sum())
        metrics["value_normalizer_mean"] = float(self.value_normalizer.mean)
        metrics["value_normalizer_std"] = float(self.value_normalizer.scale)
        metrics["update_count"] = self.update_count + 1
        self.update_count += 1
        self.rollout.clear()
        return metrics

    def save(self, path):
        torch.save(self.checkpoint_state(), path)

    def checkpoint_state(self):
        return {
            "algorithm": self.checkpoint_algorithm,
            "observation_schema": {
                "driving_dim": self.driving_state_dim,
                "energy_dim": self.energy_state_dim,
                "global_state_dim": self.global_state_dim,
                "driving_features": self.driving_state_features,
                "energy_features": self.energy_state_features,
            },
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
            "actor_optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.actor_optimizers.items()
            },
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "action_transform": self.action_transform,
            "update_count": self.update_count,
            "llm_prior": self.prior_cfg,
            "config": self.cfg,
        }

    def load(self, path, load_optimizers=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != self.checkpoint_algorithm:
            raise ValueError("Checkpoint is not a MAPPO checkpoint")
        self.actors.load_state_dict(checkpoint["actors"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "value_normalizer" in checkpoint:
            self.value_normalizer.load_state_dict(checkpoint["value_normalizer"])
        self.update_count = int(checkpoint.get("update_count", 0))
        self.gamma = float(checkpoint.get("gamma", self.gamma))
        saved_config = checkpoint.get("config", {})
        self.action_transform = str(checkpoint.get(
            "action_transform",
            saved_config.get("mappo_action_transform", self.action_transform),
        )).strip().lower()
        if self.action_transform not in {"gaussian_clip", "tanh"}:
            raise ValueError("Checkpoint contains an unknown MAPPO action transform")
        self.action_clip_low = float(saved_config.get(
            "mappo_action_clip_low", self.action_clip_low
        ))
        self.action_clip_high = float(saved_config.get(
            "mappo_action_clip_high", self.action_clip_high
        ))
        saved_prior_cfg = checkpoint.get("llm_prior")
        if saved_prior_cfg is not None:
            self.prior_cfg = dict(saved_prior_cfg)
            self.lambda_mean = float(self.prior_cfg.get("lambda_0", 0.5))
            self.lambda_std = float(self.prior_cfg.get("lambda_1", 0.5))
            self._validate_prior_lambdas()
            if not self._external_prior_provider:
                self.prior_provider = self._build_prior_provider()
        if load_optimizers:
            for name, optimizer in self.actor_optimizers.items():
                optimizer.load_state_dict(checkpoint["actor_optimizers"][name])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

