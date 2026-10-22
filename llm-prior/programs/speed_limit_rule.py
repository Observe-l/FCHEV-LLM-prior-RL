from __future__ import annotations

import numpy as np

from ..contract import PriorOutput, observation_batch, validate_prior_output


class SpeedLimitRulePrior:
    """Example program: recommend braking when ego speed exceeds the limit."""

    def __init__(self, target_normalized_action: float = -0.999,
                 tolerance_m_s: float = 1e-6):
        target = float(target_normalized_action)
        if not -1.0 < target < 1.0:
            raise ValueError("target_normalized_action must be strictly between -1 and 1")
        self.target_normalized_action = target
        self.target_latent_mean = float(np.arctanh(target))
        self.tolerance_m_s = float(tolerance_m_s)

    def predict(self, observation) -> PriorOutput:
        full_obs, _ = observation_batch(observation)
        # Raw FCHEVEnv observation indices: ego speed=5 and current posted
        # speed limit=12. The next-region limit applies only after entering it.
        speed = full_obs[:, 5]
        current_limit = full_obs[:, 12]
        active_driving = (current_limit > 0.0) & (
            speed > current_limit + self.tolerance_m_s
        )
        batch = full_obs.shape[0]
        zeros = np.zeros((batch, 1), dtype=np.float32)
        driving_mean = zeros.copy()
        driving_mean[active_driving, 0] = self.target_latent_mean
        return validate_prior_output({
            "driving": {
                "mean": driving_mean,
                "std": zeros,
                "active": active_driving.reshape(batch, 1),
            },
            "energy": {
                "mean": zeros.copy(),
                "std": zeros.copy(),
                "active": np.zeros((batch, 1), dtype=bool),
            },
        }, batch_size=batch)


def build_prior(config):
    """Optional factory used to pass rule settings from experiment config."""
    return SpeedLimitRulePrior(
        target_normalized_action=config.get("target_normalized_action", -0.999),
        tolerance_m_s=config.get("tolerance_m_s", 1e-6),
    )


_default_prior = SpeedLimitRulePrior()


def predict(observation):
    """Standard program entry point: raw observation -> Gaussian parameters."""
    return _default_prior.predict(observation)
