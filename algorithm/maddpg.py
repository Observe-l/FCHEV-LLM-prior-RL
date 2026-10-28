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
        self.row_width = 2 * self.state_dim + self.action_dim + self.reward_dim + 1
        self.data = torch.empty(
            (self.capacity, self.row_width), dtype=torch.float32, device=self.device
        )
        self.position = 0
        self.size = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

    def add(self, state, action, rewards, next_state, done):
        row = np.empty(self.row_width, dtype=np.float32)
        offset = 0
        for value, width in (
            (state, self.state_dim), (action, self.action_dim),
            (rewards, self.reward_dim), (next_state, self.state_dim),
            ([float(done)], 1),
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
        return (
            batch[:, :s0], batch[:, s0:a0], batch[:, a0:r0],
            batch[:, r0:ns0], batch[:, ns0:],
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
        self.replay = MADDPGReplayBuffer(
            self.global_state_dim, self.joint_action_dim, reward_dim=2,
            capacity=config.get("replay_capacity", 100000), seed=seed,
            device=self.device,
        )
        self.update_count = 0

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
                         next_state, done):
        self.replay.add(
            state, joint_action, [driving_reward, energy_reward], next_state, done
        )

    def update(self, batch_size):
        state, joint_action, rewards, next_state, done = self.replay.sample(batch_size)
        next_driving_state = next_state[:, self.driving_observation_indices]
        next_energy_state = next_state[:, self.energy_observation_indices]
        with torch.no_grad():
            next_joint_action = torch.cat((
                self.target_actors["driving"](next_driving_state),
                self.target_actors["energy"](next_energy_state),
            ), dim=-1)

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
        for name, other_name in (("driving", "energy"), ("energy", "driving")):
            driving_state = state[:, self.driving_observation_indices]
            energy_state = state[:, self.energy_observation_indices]
            driving_action = self.actors["driving"](driving_state)
            energy_action = self.actors["energy"](energy_state)
            if name == "driving":
                policy_joint_action = torch.cat((driving_action, energy_action.detach()), dim=-1)
                own_action = driving_action
            else:
                policy_joint_action = torch.cat((driving_action.detach(), energy_action), dim=-1)
                own_action = energy_action
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
