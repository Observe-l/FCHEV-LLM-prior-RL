from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithm.networks import Critic, DeterministicActor


class MADDPGReplayBuffer:
    """Joint replay for two actors with separate scalar rewards."""

    def __init__(self, state_dim, action_dim, reward_dim=2, capacity=100000,
                 seed=0, device="cpu"):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.reward_dim = int(reward_dim)
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.row_width = (
            2 * self.state_dim + self.action_dim + self.reward_dim + 1
            + 4 * self.action_dim
        )
        self.data = torch.empty(
            (self.capacity, self.row_width), dtype=torch.float32, device=self.device
        )
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def add(self, state, action, rewards, next_state, done,
            rule_values=None, rule_active=None,
            next_rule_values=None, next_rule_active=None):
        row = np.empty(self.row_width, dtype=np.float32)
        zeros = np.zeros(self.action_dim, dtype=np.float32)
        offset = 0
        for value, width in (
            (state, self.state_dim), (action, self.action_dim),
            (rewards, self.reward_dim), (next_state, self.state_dim),
            ([float(done)], 1),
            (zeros if rule_values is None else rule_values, self.action_dim),
            (zeros if rule_active is None else rule_active, self.action_dim),
            (zeros if next_rule_values is None else next_rule_values, self.action_dim),
            (zeros if next_rule_active is None else next_rule_active, self.action_dim),
        ):
            row[offset:offset + width] = value
            offset += width
        self.data[self.position].copy_(torch.as_tensor(row))
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(
            self.size, (int(batch_size),), generator=self.generator, device=self.device
        )
        batch = self.data.index_select(0, idx)
        s0 = self.state_dim
        a0 = s0 + self.action_dim
        r0 = a0 + self.reward_dim
        ns0 = r0 + self.state_dim
        # ns0 is already the end offset of next_state. The terminal flag is
        # the next single column; adding state_dim here shifts every rule
        # field beyond the replay row and returned incorrectly shaped slices.
        done0 = ns0
        rule0 = done0 + 1
        active0 = rule0 + self.action_dim
        next_rule0 = active0 + self.action_dim
        next_active0 = next_rule0 + self.action_dim
        return (
            batch[:, :s0], batch[:, s0:a0], batch[:, a0:r0],
            batch[:, r0:ns0], batch[:, done0:rule0],
            batch[:, rule0:active0], batch[:, active0:next_rule0],
            batch[:, next_rule0:next_active0], batch[:, next_active0:],
        )

    def __len__(self):
        return self.size


class MADDPG:
    """Two-agent MADDPG with centralized critics and separate rewards.

    Each actor uses its own observation. Each centralized critic sees the full
    normalized 15-D state and the joint two-dimensional action, but learns only
    its corresponding agent's reward (driving or energy management).
    """

    checkpoint_algorithm = "maddpg"
    driving_state_dim = 11
    energy_state_dim = 6
    global_state_dim = 15
    action_dim = 1
    joint_action_dim = 2
    driving_observation_indices = (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    energy_observation_indices = (4, 5, 0, 1, 2, 3)

    def __init__(self, config, seed=0):
        self.cfg = dict(config)
        requested_device = config.get("device", "auto")
        self.device = torch.device(
            "cuda" if requested_device == "auto" and torch.cuda.is_available()
            else ("cpu" if requested_device == "auto" else requested_device)
        )
        hidden = tuple(config.get("hidden_layers", [256, 128, 64]))
        self.actors = nn.ModuleDict({
            "driving": DeterministicActor(self.driving_state_dim, 1, hidden),
            "energy": DeterministicActor(self.energy_state_dim, 1, hidden),
        }).to(self.device)
        self.target_actors = copy.deepcopy(self.actors).to(self.device)
        self.critics = nn.ModuleDict({
            name: Critic(self.global_state_dim, self.joint_action_dim, hidden)
            for name in ("driving", "energy")
        }).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        actor_lr = float(config.get("actor_learning_rate", 3e-4))
        critic_lr = float(config.get("critic_learning_rate", actor_lr))
        self.actor_optimizers = {
            name: torch.optim.Adam(actor.parameters(), lr=actor_lr)
            for name, actor in self.actors.items()
        }
        self.critic_optimizers = {
            name: torch.optim.Adam(critic.parameters(), lr=critic_lr)
            for name, critic in self.critics.items()
        }
        self.gamma = float(config.get("gamma", 0.99))
        self.tau = float(config.get("tau", 0.005))
        self.rule_output_mode = str(config.get("maddpg_rule_output_mode", "none"))
        self.rule_lambda = float(config.get("maddpg_rule_lambda", 0.2))
        if self.rule_output_mode not in (
            "none", "physical_action", "normalized_action", "pre_tanh"
        ):
            raise ValueError(f"Unsupported MADDPG rule mode: {self.rule_output_mode!r}")
        if not 0.0 <= self.rule_lambda <= 1.0:
            raise ValueError("maddpg_rule_lambda must be in [0, 1]")
        self.action_low = np.asarray(
            config.get("action_low", [-1.5, 0.0]), dtype=np.float32
        ).reshape(2)
        self.action_high = np.asarray(
            config.get("action_high", [0.7, 50.0]), dtype=np.float32
        ).reshape(2)
        if np.any(self.action_high <= self.action_low):
            raise ValueError("MADDPG action_high must be greater than action_low")
        self.replay = MADDPGReplayBuffer(
            self.global_state_dim, self.joint_action_dim, reward_dim=2,
            capacity=config.get("replay_capacity", 100000), seed=seed,
            device=self.device,
        )
        self.update_count = 0

    def fuse_numpy_actions(self, actor_actions, rule_values=None, rule_active=None):
        """Return the normalized joint action after optional rule fusion."""
        actor = np.asarray(actor_actions, dtype=np.float32).reshape(2)
        if self.rule_output_mode == "none" or rule_values is None:
            return actor.copy()
        rule = np.asarray(rule_values, dtype=np.float32).reshape(2)
        active = np.ones(2, dtype=np.float32) if rule_active is None else np.asarray(
            rule_active, dtype=np.float32
        ).reshape(2)
        weight = self.rule_lambda * active
        if self.rule_output_mode == "physical_action":
            actor_physical = self._unit_to_physical_numpy(actor)
            mixed_physical = (1.0 - weight) * actor_physical + weight * rule
            return np.clip(self._physical_to_unit_numpy(mixed_physical), -1.0, 1.0)
        if self.rule_output_mode == "pre_tanh":
            # The actor API exposes normalized tanh actions. Invert that map
            # before blending so this mode truly mixes pre-tanh latents.
            actor_latent = np.arctanh(np.clip(actor, -0.999999, 0.999999))
            return np.tanh((1.0 - weight) * actor_latent + weight * rule)
        rule_unit = rule
        return np.clip((1.0 - weight) * actor + weight * rule_unit, -1.0, 1.0)

    def _unit_to_physical_numpy(self, unit):
        unit = np.asarray(unit, dtype=np.float32).reshape(2)
        acceleration = (
            unit[0] * abs(float(self.action_low[0]))
            if unit[0] < 0.0 else unit[0] * float(self.action_high[0])
        )
        power = self.action_low[1] + (unit[1] + 1.0) * 0.5 * (
            self.action_high[1] - self.action_low[1]
        )
        return np.asarray([acceleration, power], dtype=np.float32)

    def _physical_to_unit_numpy(self, physical):
        physical = np.asarray(physical, dtype=np.float32).reshape(2)
        acceleration = (
            physical[0] / abs(float(self.action_low[0]))
            if physical[0] < 0.0 else physical[0] / float(self.action_high[0])
        )
        power = 2.0 * (physical[1] - self.action_low[1]) / (
            self.action_high[1] - self.action_low[1]
        ) - 1.0
        return np.asarray([acceleration, power], dtype=np.float32)

    def _fuse_torch_actions(self, actor_actions, rule_values, rule_active,
                            actor_pre_tanh=False):
        if self.rule_output_mode == "none":
            return actor_actions
        active = rule_active.to(dtype=actor_actions.dtype)
        weight = self.rule_lambda * active
        if self.rule_output_mode == "physical_action":
            low = torch.as_tensor(self.action_low, dtype=actor_actions.dtype, device=self.device)
            high = torch.as_tensor(self.action_high, dtype=actor_actions.dtype, device=self.device)
            actor_acceleration = torch.where(
                actor_actions[:, :1] < 0.0,
                actor_actions[:, :1] * torch.abs(low[0]),
                actor_actions[:, :1] * high[0],
            )
            actor_power = low[1] + (actor_actions[:, 1:2] + 1.0) * 0.5 * (
                high[1] - low[1]
            )
            actor_physical = torch.cat((actor_acceleration, actor_power), dim=-1)
            mixed_physical = (1.0 - weight) * actor_physical + weight * rule_values
            normalized_acceleration = torch.where(
                mixed_physical[:, :1] < 0.0,
                mixed_physical[:, :1] / torch.abs(low[0]),
                mixed_physical[:, :1] / high[0],
            )
            normalized_power = 2.0 * (mixed_physical[:, 1:2] - low[1]) / (
                high[1] - low[1]
            ) - 1.0
            return torch.cat(
                (normalized_acceleration, normalized_power), dim=-1
            ).clamp(-1.0, 1.0)
        if self.rule_output_mode == "pre_tanh":
            actor_latent = (
                actor_actions if actor_pre_tanh
                else torch.atanh(actor_actions.clamp(-0.999999, 0.999999))
            )
            return torch.tanh((1.0 - weight) * actor_latent + weight * rule_values)
        rule_unit = rule_values
        return ((1.0 - weight) * actor_actions + weight * rule_unit).clamp(-1.0, 1.0)

    @torch.no_grad()
    def select_actions(self, driving_state, energy_state, deterministic=True):
        driving = torch.as_tensor(
            np.asarray(driving_state, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        energy = torch.as_tensor(
            np.asarray(energy_state, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        return (
            self.actors["driving"](driving).cpu().numpy()[0],
            self.actors["energy"](energy).cpu().numpy()[0],
        )

    def store_transition(self, state, joint_action, driving_reward, energy_reward,
                         next_state, done, rule_values=None, rule_active=None,
                         next_rule_values=None, next_rule_active=None):
        self.replay.add(
            state, joint_action, [driving_reward, energy_reward], next_state, done,
            rule_values, rule_active, next_rule_values, next_rule_active,
        )

    def update(self, batch_size):
        (
            state, joint_action, rewards, next_state, done,
            rule_values, rule_active, next_rule_values, next_rule_active,
        ) = self.replay.sample(batch_size)
        next_driving_state = next_state[:, self.driving_observation_indices]
        next_energy_state = next_state[:, self.energy_observation_indices]
        with torch.no_grad():
            if self.rule_output_mode == "pre_tanh":
                next_actor_action = torch.cat((
                    self.target_actors["driving"].pre_tanh(next_driving_state),
                    self.target_actors["energy"].pre_tanh(next_energy_state),
                ), dim=-1)
            else:
                next_actor_action = torch.cat((
                    self.target_actors["driving"](next_driving_state),
                    self.target_actors["energy"](next_energy_state),
                ), dim=-1)
            next_joint_action = self._fuse_torch_actions(
                next_actor_action, next_rule_values, next_rule_active,
                actor_pre_tanh=self.rule_output_mode == "pre_tanh",
            )

        critic_losses = {}
        for i, name in enumerate(("driving", "energy")):
            with torch.no_grad():
                target_q = self.target_critics[name](next_state, next_joint_action)
                target = rewards[:, i:i + 1] + self.gamma * (1.0 - done) * target_q
            q = self.critics[name](state, joint_action)
            loss = nn.functional.mse_loss(q, target)
            optimizer = self.critic_optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critics[name].parameters(), 10.0)
            optimizer.step()
            critic_losses[name] = loss.detach()

        actor_losses = {}
        for name in ("driving", "energy"):
            driving_state = state[:, self.driving_observation_indices]
            energy_state = state[:, self.energy_observation_indices]
            if self.rule_output_mode == "pre_tanh":
                driving_action = self.actors["driving"].pre_tanh(driving_state)
                energy_action = self.actors["energy"].pre_tanh(energy_state)
            else:
                driving_action = self.actors["driving"](driving_state)
                energy_action = self.actors["energy"](energy_state)
            if name == "driving":
                actor_joint_action = torch.cat((driving_action, energy_action.detach()), dim=-1)
            else:
                actor_joint_action = torch.cat((driving_action.detach(), energy_action), dim=-1)
            policy_joint_action = self._fuse_torch_actions(
                actor_joint_action, rule_values, rule_active,
                actor_pre_tanh=self.rule_output_mode == "pre_tanh",
            )
            critic = self.critics[name]
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            loss = -critic(state, policy_joint_action).mean()
            optimizer = self.actor_optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actors[name].parameters(), 10.0)
            optimizer.step()
            for parameter in critic.parameters():
                parameter.requires_grad_(True)
            actor_losses[name] = loss.detach()

        for name in ("driving", "energy"):
            self._soft_update(self.actors[name], self.target_actors[name])
            self._soft_update(self.critics[name], self.target_critics[name])
        self.update_count += 1
        return {
            "driving_actor_loss": float(actor_losses["driving"].cpu()),
            "energy_actor_loss": float(actor_losses["energy"].cpu()),
            "driving_critic_loss": float(critic_losses["driving"].cpu()),
            "energy_critic_loss": float(critic_losses["energy"].cpu()),
            "driving_q_mean": float(self.critics["driving"](state, joint_action).mean().detach().cpu()),
            "energy_q_mean": float(self.critics["energy"](state, joint_action).mean().detach().cpu()),
        }

    def _soft_update(self, source, target):
        with torch.no_grad():
            for parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "algorithm": self.checkpoint_algorithm,
            "actors": self.actors.state_dict(),
            "target_actors": self.target_actors.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.actor_optimizers.items()
            },
            "critic_optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in self.critic_optimizers.items()
            },
            "update_count": self.update_count,
            "config": self.cfg,
            "rule_output_mode": self.rule_output_mode,
            "rule_lambda": self.rule_lambda,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "reward_partition": {
                "driving_actor": "driving_reward",
                "energy_actor": "energy_reward",
            },
        }, path)

    def load(self, path, load_optimizers=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != self.checkpoint_algorithm:
            raise ValueError("Checkpoint is not a MADDPG checkpoint")
        self.actors.load_state_dict(checkpoint["actors"])
        self.target_actors.load_state_dict(checkpoint["target_actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.update_count = int(checkpoint.get("update_count", 0))
        if load_optimizers:
            for name, optimizer in self.actor_optimizers.items():
                optimizer.load_state_dict(checkpoint["actor_optimizers"][name])
            for name, optimizer in self.critic_optimizers.items():
                optimizer.load_state_dict(checkpoint["critic_optimizers"][name])
