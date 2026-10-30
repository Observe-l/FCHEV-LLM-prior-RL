from __future__ import annotations

from importlib import import_module
from collections.abc import Mapping
import re

import numpy as np


MADDPG_RULE_MODES = ("physical_action", "normalized_action", "pre_tanh")
OBSERVATION_DIM = 15


def _as_batch(observation):
    values = np.asarray(observation, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != OBSERVATION_DIM:
        raise ValueError(
            "MADDPG rule input must have shape (15,) or (batch, 15); "
            f"received {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("MADDPG rule input must contain only finite values")
    return values


def _scalar_batch(value, batch_size, label, *, boolean=False):
    dtype = bool if boolean else np.float32
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full((batch_size,), array.item(), dtype=dtype)
    elif array.ndim == 2 and array.shape == (batch_size, 1):
        array = array[:, 0]
    elif array.ndim == 1 and array.shape == (1,) and batch_size > 1:
        array = np.full((batch_size,), array[0], dtype=dtype)
    elif array.ndim != 1 or array.shape[0] != batch_size:
        raise ValueError(
            f"MADDPG rule field {label!r} must be a scalar, (batch,), or "
            f"(batch, 1); received {array.shape} for batch {batch_size}"
        )
    if not boolean and not np.all(np.isfinite(array)):
        raise ValueError(f"MADDPG rule field {label!r} must be finite")
    return array


class MADDPGRuleAdapter:
    """Load and validate a deterministic two-actor MADDPG rule program.

    The program receives the full raw environment observation. ``predict``
    always returns ``values`` and ``active`` arrays with shape ``(batch, 2)``,
    ordered as driving then energy. The meaning of values is selected by mode.
    """

    def __init__(self, program_name, mode, action_low, action_high, config=None):
        if mode not in MADDPG_RULE_MODES:
            raise ValueError(
                f"MADDPG rule mode must be one of {MADDPG_RULE_MODES}; got {mode!r}"
            )
        if not isinstance(program_name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", program_name
        ):
            raise ValueError("Rule program name must be a Python module name")
        self.mode = mode
        self.action_low = np.asarray(action_low, dtype=np.float32).reshape(2)
        self.action_high = np.asarray(action_high, dtype=np.float32).reshape(2)
        if not np.all(np.isfinite(self.action_low)) or not np.all(
            np.isfinite(self.action_high)
        ) or np.any(self.action_high <= self.action_low):
            raise ValueError("MADDPG physical action bounds must be finite and ordered")

        module = import_module(f"llm-prior.programs.{program_name}")
        factory = getattr(module, "build_prior", None)
        self.program = factory(dict(config or {})) if callable(factory) else module
        if not callable(getattr(self.program, "predict", None)):
            raise TypeError(
                f"Rule program {program_name!r} must expose predict(observation) "
                "or build_prior(config) returning an object with predict(observation)"
            )
        self.program_name = program_name

    def predict(self, observation):
        values = _as_batch(observation)
        output = self.program.predict(values.copy())
        if not isinstance(output, Mapping):
            raise TypeError("MADDPG rule predict() must return an agent mapping")

        action_values = []
        active_values = []
        for index, name in enumerate(("driving", "energy")):
            if name not in output or not isinstance(output[name], Mapping):
                raise ValueError(f"MADDPG rule output must contain a {name!r} mapping")
            item = output[name]
            if self.mode == "pre_tanh":
                # MADDPG has no Gaussian distribution, so generated rules can
                # use the explicit `prior` name. Accept `mean` as a migration
                # path for the existing HASAC rule module; its std is ignored.
                field = "prior" if "prior" in item else "mean"
            else:
                field = "action"
            if field not in item:
                raise ValueError(
                    f"Rule mode {self.mode!r} requires output field {name}.{field}"
                )
            action_value = _scalar_batch(
                item[field], values.shape[0], f"{name}.{field}"
            )
            active = _scalar_batch(
                item.get("active", True), values.shape[0], f"{name}.active",
                boolean=True,
            )
            if self.mode == "physical_action":
                low, high = self.action_low[index], self.action_high[index]
                if np.any(active & ((action_value < low - 1e-5) | (action_value > high + 1e-5))):
                    unit = "m/s^2" if name == "driving" else "kW"
                    raise ValueError(
                        f"Active physical {name} rule action must be in "
                        f"[{low}, {high}] {unit}"
                    )
            elif self.mode == "normalized_action":
                if np.any(active & ((action_value < -1.0 - 1e-6) | (action_value > 1.0 + 1e-6))):
                    raise ValueError(
                        f"Active normalized {name} rule action must be in [-1, 1]"
                    )
            action_values.append(action_value)
            active_values.append(active)

        return {
            "values": np.stack(action_values, axis=1).astype(np.float32),
            "active": np.stack(active_values, axis=1).astype(bool),
        }
