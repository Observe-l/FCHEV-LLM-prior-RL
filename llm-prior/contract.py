from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


OBSERVATION_FIELDS = (
    "battery_soc",
    "battery_soh",
    "fcs_soh",
    "edu_soh",
    "ego_acceleration_m_s2",
    "ego_speed_m_s",
    "lead_acceleration_m_s2",
    "lead_speed_m_s",
    "following_distance_m",
    "next_signal_state",  # raw environment encoding: red=0, green=1, yellow=2
    "next_signal_distance_m",
    "next_signal_phase_remaining_s",
    "current_speed_limit_m_s",
    "next_speed_limit_m_s",
    "distance_to_next_speed_region_m",
)
FULL_OBSERVATION_DIM = len(OBSERVATION_FIELDS)
AGENT_ACTION_DIMS = {"driving": 1, "energy": 1}


@dataclass(frozen=True)
class GaussianActionPrior:
    """One agent's prior distribution in the actor's pre-tanh latent space.

    Arrays use shape ``(batch, action_dim)``. ``active`` has the same shape
    and masks individual dimensions where a rule supplies a prior.
    """

    mean: np.ndarray
    std: np.ndarray
    active: np.ndarray


@dataclass(frozen=True)
class PriorOutput:
    """Per-agent Gaussian priors returned by every prior provider."""

    driving: GaussianActionPrior
    energy: GaussianActionPrior

    def for_agent(self, name: str) -> GaussianActionPrior:
        if name not in AGENT_ACTION_DIMS:
            raise KeyError(f"Unknown HASAC agent {name!r}")
        return getattr(self, name)


def observation_batch(observation) -> tuple[np.ndarray, bool]:
    """Validate a raw full environment observation and return a batch view."""
    values = np.asarray(observation, dtype=np.float32)
    single = values.ndim == 1
    if single:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != FULL_OBSERVATION_DIM:
        raise ValueError(
            "Prior input must be the full raw environment observation with "
            f"shape ({FULL_OBSERVATION_DIM},) or (batch, {FULL_OBSERVATION_DIM}); "
            f"received {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Prior input observation must contain only finite values")
    return values, single


def validate_prior_output(output, batch_size: int) -> PriorOutput:
    """Convert a provider mapping into validated, batch-shaped NumPy arrays."""
    if isinstance(output, PriorOutput):
        raw = {name: getattr(output, name) for name in AGENT_ACTION_DIMS}
    elif isinstance(output, Mapping):
        raw = output
    else:
        raise TypeError("Prior provider must return PriorOutput or an agent mapping")

    checked = {}
    for name, action_dim in AGENT_ACTION_DIMS.items():
        if name not in raw:
            raise ValueError(f"Prior output is missing the {name!r} agent")
        item = raw[name]
        if isinstance(item, GaussianActionPrior):
            mean, std, active = item.mean, item.std, item.active
        elif isinstance(item, Mapping):
            # A program that always supplies a prior only needs to return
            # mean/std. Conditional programs may include ``active`` to mask
            # dimensions where their prior should not be fused.
            mean, std, active = item.get("mean"), item.get("std"), item.get("active", True)
        else:
            raise TypeError(f"Prior output for {name!r} must be a mapping or GaussianActionPrior")

        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        active = np.asarray(active, dtype=bool)
        if mean.ndim == 0:
            mean = mean.reshape(1, 1)
        elif mean.ndim == 1:
            mean = mean.reshape(-1, action_dim)
        if std.ndim == 0:
            std = std.reshape(1, 1)
        elif std.ndim == 1:
            std = std.reshape(-1, action_dim)
        if active.ndim == 0:
            active = np.full((batch_size, action_dim), bool(active), dtype=bool)
        elif active.ndim == 1:
            if active.size == action_dim:
                active = np.broadcast_to(active.reshape(1, action_dim), (batch_size, action_dim))
            else:
                active = active.reshape(-1, action_dim)

        expected = (batch_size, action_dim)
        if mean.shape != expected or std.shape != expected or active.shape != expected:
            raise ValueError(
                f"Prior output for {name!r} must have mean/std/active shape {expected}; "
                f"received {mean.shape}, {std.shape}, {active.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError(f"Prior output for {name!r} must be finite")
        if np.any(std < 0.0):
            raise ValueError(f"Prior std for {name!r} cannot be negative")
        checked[name] = GaussianActionPrior(mean=mean, std=std, active=active)

    return PriorOutput(**checked)
