from __future__ import annotations

from importlib import import_module
from pathlib import Path
import re

import torch

from algorithm.sac import SAC


class TwoAgentSAC:
    """Independent SAC policies for longitudinal driving and energy management."""

    checkpoint_algorithm = "two_agent_sac"

    def __init__(self, config, seed=0):
        self.prior_cfg = dict(config.get("llm_prior", {}))
        self.prior_provider = self._build_prior_provider()
        self.prior_program_source = None
        self.prior_program_version = 0
        self.driving = SAC(
            state_dim=11, action_dim=1, config=config, seed=seed,
            prior_provider=self.prior_provider, prior_config=self.prior_cfg,
            agent_name="driving",
        )
        # Match the current MADDPG energy observation: measured ego
        # acceleration, ego speed, SOC, battery SOH, FCS SOH, and EDU SOH.
        self.energy = SAC(
            state_dim=6, action_dim=1, config=config, seed=seed + 1,
            prior_provider=self.prior_provider, prior_config=self.prior_cfg,
            agent_name="energy",
        )

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
                raise ValueError(
                    f"Unknown prior program {program_name!r}; expected a module under llm-prior/programs"
                ) from exc
            raise
        provider = (
            module.build_prior(dict(self.prior_cfg))
            if callable(getattr(module, "build_prior", None))
            else module
        )
        if not callable(getattr(provider, "predict", None)):
            raise TypeError(
                f"Prior program {program_name!r} must expose predict(observation)"
            )
        return provider

    def _predict_prior(self, full_observation):
        if self.prior_provider is None:
            return None
        return self.driving._predict_prior(full_observation)

    def _agent_lambda_weights(self, agent_name):
        agent = self.agents[agent_name]
        return agent.lambda_mean, agent.lambda_std

    def set_runtime_prior(self, provider, *, source, version, lambda_value=None):
        """Install one episode-boundary rule version for both independent actors."""
        self.prior_provider = provider
        self.prior_program_source = source
        self.prior_program_version = int(version)
        self.prior_cfg["runtime_rule_program_version"] = int(version)
        for subagent in (self.driving, self.energy):
            subagent.prior_provider = provider
            subagent.prior_cfg["runtime_rule_program_version"] = int(version)
            if lambda_value is not None:
                value = float(lambda_value)
                if not 0.0 <= value <= 1.0:
                    raise ValueError("online rule lambda must be in [0, 1]")
                subagent.lambda_mean = value
                subagent.lambda_std = value
                subagent.prior_cfg["lambda_0"] = value
                subagent.prior_cfg["lambda_1"] = value
                subagent.prior_cfg["lambda_by_agent"] = {
                    "driving": {"lambda_0": value, "lambda_1": value},
                    "energy": {"lambda_0": value, "lambda_1": value},
                }
        if lambda_value is not None:
            value = float(lambda_value)
            self.prior_cfg["lambda_0"] = value
            self.prior_cfg["lambda_1"] = value
            self.prior_cfg["lambda_by_agent"] = {
                "driving": {"lambda_0": value, "lambda_1": value},
                "energy": {"lambda_0": value, "lambda_1": value},
            }

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
            "llm_prior": self.prior_cfg,
            "prior_program_source": self.prior_program_source,
            "prior_program_version": self.prior_program_version,
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
        saved_prior_cfg = checkpoint.get("llm_prior")
        if isinstance(saved_prior_cfg, dict):
            self.prior_cfg = dict(saved_prior_cfg)
            for subagent in (self.driving, self.energy):
                subagent.prior_cfg = dict(saved_prior_cfg)
                subagent.lambda_mean, subagent.lambda_std = subagent._agent_lambda_weights()
        source = checkpoint.get("prior_program_source")
        if source:
            compiler = import_module("llm-prior.online_rule_update").compile_rule_provider
            self.set_runtime_prior(
                compiler(source), source=source,
                version=checkpoint.get("prior_program_version", 0),
            )
