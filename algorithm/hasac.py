from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithm.networks import Critic, GaussianActor
from algorithm.replay_buffer import ReplayBuffer


class HASAC:
    """Heterogeneous-Agent SAC with decentralized actors and twin joint critics.

    The driving and energy actors each consume their own observation. A shared
    replay buffer stores the concatenated joint state/action and common team
    reward. Each actor is updated in a randomly sampled sequential order using
    the centralized soft Q-function, following the HASAC policy-improvement
    procedure. Entropy temperature is fixed by ``hasac_alpha``.
    """

    checkpoint_algorithm = "hasac"
    driving_state_dim = 11
    energy_state_dim = 5
    action_dim_per_agent = 1

    def __init__(self, config, seed=0):
        self.cfg = dict(config)
        requested_device = config.get("device", "auto")
        self.device = torch.device(
            "cuda" if requested_device == "auto" and torch.cuda.is_available()
            else ("cpu" if requested_device == "auto" else requested_device)
        )
        hidden = tuple(config.get("hidden_layers", [256, 128, 64]))
        joint_state_dim = self.driving_state_dim + self.energy_state_dim
        joint_action_dim = 2 * self.action_dim_per_agent

        self.driving_actor = GaussianActor(
            self.driving_state_dim, self.action_dim_per_agent, hidden
        ).to(self.device)
        self.energy_actor = GaussianActor(
            self.energy_state_dim, self.action_dim_per_agent, hidden
        ).to(self.device)
        self.q1 = Critic(joint_state_dim, joint_action_dim, hidden).to(self.device)
        self.q2 = Critic(joint_state_dim, joint_action_dim, hidden).to(self.device)
        self.target_q1 = copy.deepcopy(self.q1).to(self.device)
        self.target_q2 = copy.deepcopy(self.q2).to(self.device)

        actor_lr = float(config.get("actor_learning_rate", 3e-4))
        critic_lr = float(config.get("critic_learning_rate", 3e-4))
        self.driving_actor_opt = torch.optim.Adam(self.driving_actor.parameters(), lr=actor_lr)
        self.energy_actor_opt = torch.optim.Adam(self.energy_actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.alpha = float(config.get("hasac_alpha", 1.0))
        self.gamma = float(config.get("gamma", 0.9999))
        self.tau = float(config.get("tau", 0.005))
        self.replay = ReplayBuffer(
            joint_state_dim, joint_action_dim,
            config.get("replay_capacity", 100000), seed, device=self.device,
        )
        self._permutation_rng = np.random.default_rng(int(seed) + 7919)
        self.update_count = 0

    @torch.no_grad()
    def select_actions(self, driving_state, energy_state, deterministic=False):
        driving_tensor = torch.as_tensor(
            np.asarray(driving_state, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        energy_tensor = torch.as_tensor(
            np.asarray(energy_state, dtype=np.float32), device=self.device
        ).reshape(1, -1)
        driving_action, _ = self.driving_actor.sample(
            driving_tensor, deterministic=deterministic
        )
        energy_action, _ = self.energy_actor.sample(
            energy_tensor, deterministic=deterministic
        )
        return driving_action.cpu().numpy()[0], energy_action.cpu().numpy()[0]

    def store_transition(self, driving_state, energy_state, driving_action,
                         energy_action, driving_reward, energy_reward,
                         next_driving_state, next_energy_state, done,
                         expected_team_reward=None):
        """Store one joint transition with the identical cooperative reward.

        The shared reward is explicitly the sum of the current driving and
        energy reward components. ``expected_team_reward`` checks that this
        decomposition matches the environment's reported total.
        """
        team_reward = float(driving_reward) + float(energy_reward)
        if expected_team_reward is not None and not np.isclose(
            team_reward, float(expected_team_reward), rtol=0.0, atol=1e-6
        ):
            raise ValueError(
                "HASAC shared reward must equal driving reward plus energy reward"
            )
        self.replay.add(
            np.concatenate((driving_state, energy_state)),
            np.asarray([driving_action, energy_action], dtype=np.float32).reshape(-1),
            team_reward,
            np.concatenate((next_driving_state, next_energy_state)),
            done,
        )
        return team_reward

    def update(self, batch_size):
        joint_state, joint_action, reward, next_joint_state, done = self.replay.sample(batch_size)
        driving_state = joint_state[:, :self.driving_state_dim]
        energy_state = joint_state[:, self.driving_state_dim:]
        next_driving_state = next_joint_state[:, :self.driving_state_dim]
        next_energy_state = next_joint_state[:, self.driving_state_dim:]

        with torch.no_grad():
            next_driving_action, driving_logp = self.driving_actor.sample(next_driving_state)
            next_energy_action, energy_logp = self.energy_actor.sample(next_energy_state)
            next_joint_action = torch.cat((next_driving_action, next_energy_action), dim=-1)
            soft_next_q = torch.minimum(
                self.target_q1(next_joint_state, next_joint_action),
                self.target_q2(next_joint_state, next_joint_action),
            ) - self.alpha * (driving_logp + energy_logp)
            target = reward + self.gamma * (1.0 - done) * soft_next_q

        q1 = self.q1(joint_state, joint_action)
        q2 = self.q2(joint_state, joint_action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)
        self.critic_opt.step()

        actor_losses = {}
        order = self._permutation_rng.permutation(2)
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        for agent_index in order:
            # Re-sample both policies at each sequential improvement step. The
            # other agent's sampled action is held fixed for this actor update.
            driving_action, driving_logp = self.driving_actor.sample(driving_state)
            energy_action, energy_logp = self.energy_actor.sample(energy_state)
            joint_policy_action = torch.cat((driving_action, energy_action), dim=-1)
            if agent_index == 0:
                joint_policy_action = torch.cat(
                    (driving_action, energy_action.detach()), dim=-1
                )
                own_logp = driving_logp
                actor = self.driving_actor
                optimizer = self.driving_actor_opt
                name = "driving"
            else:
                joint_policy_action = torch.cat(
                    (driving_action.detach(), energy_action), dim=-1
                )
                own_logp = energy_logp
                actor = self.energy_actor
                optimizer = self.energy_actor_opt
                name = "energy"

            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            q_policy = torch.minimum(
                self.q1(joint_state, joint_policy_action),
                self.q2(joint_state, joint_policy_action),
            )
            actor_loss = (self.alpha * own_logp - q_policy).mean()
            optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
            optimizer.step()
            for parameter in critic_parameters:
                parameter.requires_grad_(True)
            actor_losses[name] = float(actor_loss.detach().cpu())

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(np.mean(list(actor_losses.values()))),
            "driving_actor_loss": actor_losses["driving"],
            "energy_actor_loss": actor_losses["energy"],
            "alpha_loss": 0.0,
            "alpha": self.alpha,
        }

    def _soft_update(self, source, target):
        with torch.no_grad():
            for parameter, target_parameter in zip(source.parameters(), target.parameters()):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(), path)

    def checkpoint_state(self):
        return {
            "algorithm": self.checkpoint_algorithm,
            "observation_schema": {
                "driving_dim": self.driving_state_dim,
                "energy_dim": self.energy_state_dim,
            },
            "driving_actor": self.driving_actor.state_dict(),
            "energy_actor": self.energy_actor.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "driving_actor_opt": self.driving_actor_opt.state_dict(),
            "energy_actor_opt": self.energy_actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha": self.alpha, "gamma": self.gamma, "tau": self.tau,
            "update_count": self.update_count,
            "config": self.cfg,
        }

    def load(self, path, load_optimizers=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != self.checkpoint_algorithm:
            raise ValueError("Checkpoint is not a HASAC checkpoint")
        self.driving_actor.load_state_dict(checkpoint["driving_actor"])
        self.energy_actor.load_state_dict(checkpoint["energy_actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.target_q1.load_state_dict(checkpoint["target_q1"])
        self.target_q2.load_state_dict(checkpoint["target_q2"])
        self.alpha = float(checkpoint["alpha"])
        self.update_count = int(checkpoint.get("update_count", 0))
        if load_optimizers:
            self.driving_actor_opt.load_state_dict(checkpoint["driving_actor_opt"])
            self.energy_actor_opt.load_state_dict(checkpoint["energy_actor_opt"])
            self.critic_opt.load_state_dict(checkpoint["critic_opt"])
