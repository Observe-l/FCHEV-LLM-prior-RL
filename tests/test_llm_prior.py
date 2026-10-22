from __future__ import annotations

import importlib
import json
import sys
import types
import unittest

import numpy as np
import torch

from algorithm.hasac import HASAC
from train import map_split_actions, split_agent_observations


def make_agent(prior_enabled: bool) -> HASAC:
    return HASAC({
        "device": "cpu",
        "hidden_layers": [16],
        "gamma": 0.99,
        "replay_capacity": 32,
        "batch_size": 2,
        "llm_prior": {
            "enabled": prior_enabled,
            "program_name": "speed_limit_rule",
            "lambda_0": 0.5,
            "lambda_1": 0.5,
            "target_normalized_action": -0.999,
        },
    }, seed=7)


def zero_actor_means_and_log_stds(agent: HASAC) -> None:
    for actor in (agent.driving_actor, agent.energy_actor):
        for parameter in actor.parameters():
            torch.nn.init.zeros_(parameter)


def full_observation(speed_m_s: float, signal_state: float = 0.0) -> np.ndarray:
    # Raw FCHEVEnv 15-element order, with the current posted road limit 60 km/h.
    return np.asarray([
        0.5, 1.0, 1.0, 1.0,
        0.0, speed_m_s, 0.0, 13.0, 10.0,
        signal_state, 100.0, 20.0,
        60.0 / 3.6, 30.0 / 3.6, 50.0,
    ], dtype=np.float32)


class LLMRulePriorSmokeTests(unittest.TestCase):
    def test_speed_rule_contract_and_activation(self):
        rule_module = importlib.import_module("llm-prior.programs.speed_limit_rule")
        rule = rule_module.SpeedLimitRulePrior()
        obs = np.stack((full_observation(10.0), full_observation(18.0)))
        output = rule.predict(obs)

        self.assertEqual(output.driving.mean.shape, (2, 1))
        self.assertEqual(output.driving.std.shape, (2, 1))
        self.assertEqual(output.driving.active[:, 0].tolist(), [False, True])
        self.assertAlmostEqual(float(output.driving.std[1, 0]), 0.0)
        self.assertFalse(bool(output.energy.active.any()))
        self.assertAlmostEqual(
            float(np.tanh(output.driving.mean[1, 0])), -0.999, places=6
        )

    def test_program_name_loads_python_module_and_mean_std_only_is_valid(self):
        agent = make_agent(prior_enabled=True)
        self.assertIsNotNone(agent.prior_provider)
        self.assertEqual(type(agent.prior_provider).__name__, "SpeedLimitRulePrior")

        contract = importlib.import_module("llm-prior.contract")
        checked = contract.validate_prior_output({
            "driving": {"mean": [-0.25], "std": [0.1]},
            "energy": {"mean": [0.0], "std": [0.2]},
        }, batch_size=1)
        self.assertTrue(bool(checked.driving.active[0, 0]))
        self.assertTrue(bool(checked.energy.active[0, 0]))

    def test_plain_predict_program_is_loaded_by_configured_name(self):
        module_name = "llm-prior.programs.test_predict_only_prior"
        program = types.ModuleType(module_name)

        def predict(observation):
            batch = np.asarray(observation).reshape(-1, 15).shape[0]
            return {
                "driving": {
                    "mean": np.full((batch, 1), -0.4, dtype=np.float32),
                    "std": np.full((batch, 1), 0.2, dtype=np.float32),
                },
                "energy": {
                    "mean": np.zeros((batch, 1), dtype=np.float32),
                    "std": np.full((batch, 1), 0.3, dtype=np.float32),
                },
            }

        program.predict = predict
        sys.modules[module_name] = program
        try:
            agent = HASAC({
                "device": "cpu",
                "hidden_layers": [16],
                "replay_capacity": 8,
                "llm_prior": {
                    "enabled": True,
                    "program_name": "test_predict_only_prior",
                    "lambda_0": 0.5,
                    "lambda_1": 0.5,
                },
            })
            output = agent._predict_prior(full_observation(10.0), batch_size=1)
            self.assertAlmostEqual(float(output.driving.mean[0, 0]), -0.4)
            self.assertTrue(bool(output.driving.active[0, 0]))
            self.assertAlmostEqual(float(output.energy.std[0, 0]), 0.3)
        finally:
            sys.modules.pop(module_name, None)

    def test_vendor_neutral_bridge_parses_and_caches_json_contract(self):
        bridge_module = importlib.import_module("llm-prior.bridge")
        calls = []

        def fake_llm(system_prompt, user_payload):
            payload = json.loads(user_payload)
            self.assertEqual(len(payload["observation"]), 15)
            self.assertIn("pre-tanh latent Gaussian space", system_prompt)
            calls.append(payload)
            return json.dumps({
                "driving": {"mean": [-3.8], "std": [0.0], "active": True},
                "energy": {"mean": [0.0], "std": [0.0], "active": False},
            })

        bridge = bridge_module.CallableLLMPriorBridge(fake_llm)
        obs = full_observation(18.0)
        first = bridge.predict(obs)
        second = bridge.predict(obs)
        self.assertEqual(len(calls), 1)
        self.assertTrue(bool(first.driving.active[0, 0]))
        np.testing.assert_array_equal(first.driving.mean, second.driving.mean)

    def test_prior_fusion_changes_only_overspeed_driving_action(self):
        actor_only = make_agent(prior_enabled=False)
        fused = make_agent(prior_enabled=True)
        zero_actor_means_and_log_stds(actor_only)
        zero_actor_means_and_log_stds(fused)

        obs = full_observation(18.0)
        driving_state, energy_state = split_agent_observations(obs, {})
        base_drive, base_energy = actor_only.select_actions(
            driving_state, energy_state, deterministic=True
        )
        prior_drive, prior_energy = fused.select_actions(
            driving_state, energy_state, deterministic=True,
            full_observation=obs,
        )

        self.assertAlmostEqual(float(base_drive[0]), 0.0, places=6)
        self.assertLess(float(prior_drive[0]), -0.9)
        self.assertAlmostEqual(float(prior_energy[0]), float(base_energy[0]), places=6)
        physical = map_split_actions(
            prior_drive, prior_energy,
            type("ActionSpace", (), {
                "low": np.asarray([-1.5, 0.0], dtype=np.float32),
                "high": np.asarray([0.7, 50.0], dtype=np.float32),
            })(),
        )
        self.assertLess(float(physical[0]), -1.3)

        # Below the current limit, the active mask makes fusion a strict no-op.
        below_limit = full_observation(10.0)
        low_drive, low_energy = split_agent_observations(below_limit, {})
        before = actor_only.select_actions(low_drive, low_energy, deterministic=True)
        after = fused.select_actions(
            low_drive, low_energy, deterministic=True,
            full_observation=below_limit,
        )
        np.testing.assert_allclose(after[0], before[0], atol=1e-7)
        np.testing.assert_allclose(after[1], before[1], atol=1e-7)

    def test_hasac_update_uses_reconstructed_full_observation(self):
        agent = make_agent(prior_enabled=True)
        zero_actor_means_and_log_stds(agent)
        for speed in (18.0, 10.0):
            obs = full_observation(speed)
            next_obs = full_observation(speed - 0.2)
            drive, energy = split_agent_observations(obs, {})
            next_drive, next_energy = split_agent_observations(next_obs, {})
            agent.store_transition(
                drive, energy, np.asarray([-0.5]), np.asarray([0.0]),
                0.0, 0.0, next_drive, next_energy, False,
                expected_team_reward=0.0,
            )
        result = agent.update(batch_size=2)
        for key in ("critic_loss", "driving_actor_loss", "energy_actor_loss"):
            self.assertTrue(np.isfinite(result[key]), msg=f"non-finite {key}: {result[key]}")


if __name__ == "__main__":
    unittest.main()
