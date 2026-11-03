from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from algorithm.sac import SAC
from importlib import import_module


RulePriorRouter = import_module("llm-prior.sac_rule_router").SACRulePriorRouter


class TwoAgentSAC:
    """Independent SAC policies for longitudinal driving and energy management."""

    checkpoint_algorithm = "two_agent_sac"

    def __init__(self, config, seed=0):
        self.rule_prior_config = dict(config.get("llm_prior", {}))
        self.rule_runtime = None
        if self.rule_prior_config.get("enabled", False):
            self.rule_runtime = RulePriorRouter(
                self.rule_prior_config.get("program_name", ""),
                selection_interval_steps=self.rule_prior_config.get(
                    "selection_interval_steps", 5
                ),
                legacy_lambda=self.rule_prior_config.get("legacy_lambda", 0.2),
            )
        sac_config = dict(config)
        sac_config["rule_prior_metadata_dim"] = 8 if self.rule_runtime is not None else 0
        self.driving = SAC(state_dim=11, action_dim=1, config=sac_config, seed=seed)
        # Match the current MADDPG energy observation: measured ego
        # acceleration, ego speed, SOC, battery SOH, FCS SOH, and EDU SOH.
        self.energy = SAC(state_dim=6, action_dim=1, config=sac_config, seed=seed + 1)

    @property
    def device(self):
        return self.driving.device

    @property
    def agents(self):
        return {"driving": self.driving, "energy": self.energy}

    def reset_rule_prior(self):
        if self.rule_runtime is not None:
            self.rule_runtime.reset()

    def rule_prior_for_step(self, observation, step_index):
        if self.rule_runtime is None:
            return None
        return self.rule_runtime.get(observation, step_index)

    @staticmethod
    def _prior_metadata(decision, agent_name):
        if decision is None:
            return np.zeros(4, dtype=np.float32)
        prior = decision.prior.for_agent(agent_name)
        return np.asarray([
            float(prior.mean[0, 0]),
            float(prior.std[0, 0]),
            float(prior.active[0, 0]),
            float(decision.lambda_value),
        ], dtype=np.float32)

    def store_transition(self, driving_state, energy_state,
                         driving_action, energy_action,
                         driving_reward, energy_reward,
                         next_driving_state, next_energy_state, done,
                         current_rule_decision=None, next_rule_decision=None):
        for name, replay, state, action, reward, next_state in (
            ("driving", self.driving.replay, driving_state, driving_action,
             driving_reward, next_driving_state),
            ("energy", self.energy.replay, energy_state, energy_action,
             energy_reward, next_energy_state),
        ):
            extra = None
            if replay.extra_dim:
                extra = np.concatenate((
                    self._prior_metadata(current_rule_decision, name),
                    self._prior_metadata(next_rule_decision, name),
                ))
            replay.add(state, action, reward, next_state, done, extra=extra)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "algorithm": self.checkpoint_algorithm,
            "observation_schema": {
                "driving": [
                    "ego_acceleration_m_s2", "ego_speed_m_s",
                    "lead_acceleration_m_s2", "lead_speed_m_s",
                    "following_distance_m", "next_signal_state",
                    "distance_to_next_signal_m", "signal_phase_remaining_s",
                    "current_speed_limit_m_s", "next_speed_limit_m_s",
                    "distance_to_next_speed_region_m",
                ],
                "energy": [
                    "measured_executed_acceleration_m_s2", "ego_speed_m_s",
                    "battery_soc", "battery_soh", "fcs_soh", "edu_soh",
                ],
            },
            "reward_partition": {
                "driving": [
                    "following", "comfort", "zero_speed_negative_accel_penalty",
                    "terminal_penalty",
                ],
                "energy": ["hydrogen", "fcs", "battery", "edu", "soc"],
            },
            "rule_prior": {
                "enabled": self.rule_runtime is not None,
                "program_name": self.rule_prior_config.get("program_name"),
                "selection_interval_steps": (
                    self.rule_runtime.selection_interval_steps
                    if self.rule_runtime is not None else None
                ),
                "registry": (
                    self.rule_runtime.registry()
                    if self.rule_runtime is not None else {}
                ),
            },
            "driving_agent": self.driving.checkpoint_state(),
            "energy_agent": self.energy.checkpoint_state(),
        }, path)

    def load(self, path, load_optimizers=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("algorithm") != self.checkpoint_algorithm:
            raise ValueError("Checkpoint is not a two-agent SAC checkpoint")
        self.driving.load_checkpoint_state(
            checkpoint["driving_agent"], load_optimizers=load_optimizers
        )
        self.energy.load_checkpoint_state(
            checkpoint["energy_agent"], load_optimizers=load_optimizers
        )
