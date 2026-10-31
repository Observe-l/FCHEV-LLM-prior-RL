from __future__ import annotations

from pathlib import Path

import torch

from algorithm.sac import SAC


class TwoAgentSAC:
    """Independent SAC policies for longitudinal driving and energy management."""

    checkpoint_algorithm = "two_agent_sac"

    def __init__(self, config, seed=0):
        self.driving = SAC(state_dim=11, action_dim=1, config=config, seed=seed)
        # Match the current MADDPG energy observation: measured ego
        # acceleration, ego speed, SOC, battery SOH, FCS SOH, and EDU SOH.
        self.energy = SAC(state_dim=6, action_dim=1, config=config, seed=seed + 1)

    @property
    def device(self):
        return self.driving.device

    @property
    def agents(self):
        return {"driving": self.driving, "energy": self.energy}

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
