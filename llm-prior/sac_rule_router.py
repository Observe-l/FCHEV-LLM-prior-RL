from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import re

import numpy as np

from .contract import PriorOutput, validate_prior_output


@dataclass(frozen=True)
class RulePriorDecision:
    rule_name: str | None
    lambda_value: float
    prior: PriorOutput
    selection_updated: bool


class SACRulePriorRouter:
    """Select and evaluate one generated SAC rule over a configurable interval.

    The selected rule name and lambda are held for ``selection_interval_steps``.
    The selected rule function itself is evaluated on the fresh raw observation
    at every environment step.
    """

    def __init__(self, program_name: str, selection_interval_steps: int = 5,
                 legacy_lambda: float = 0.2):
        if not isinstance(program_name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", program_name
        ):
            raise ValueError("Rule program name must be a Python module name")
        self.program_name = program_name
        self.selection_interval_steps = int(selection_interval_steps)
        if self.selection_interval_steps < 1:
            raise ValueError("selection_interval_steps must be at least 1")

        self.module = import_module(f"llm-prior.programs.{program_name}")
        self.rules = getattr(self.module, "RULES", None)
        self.legacy = False
        if not isinstance(self.rules, dict) or not self.rules:
            # Backward-compatible adapter for older single-rule SAC programs.
            # New prompts should emit RULES + select_rule instead.
            predict = getattr(self.module, "predict", None)
            if not callable(predict):
                raise TypeError(
                    f"{program_name} must define RULES and select_rule(observation)"
                )
            self.rules = {"legacy_predict": predict}
            self.selector = None
            self.legacy = True
            self.legacy_lambda = float(legacy_lambda)
            if not 0.0 <= self.legacy_lambda <= 1.0:
                raise ValueError("legacy_lambda must be in [0, 1]")
        else:
            self.selector = getattr(self.module, "select_rule", None)
            if not callable(self.selector):
                raise TypeError(f"{program_name} must define select_rule(observation)")
            if len(self.rules) < 2:
                raise ValueError("Multi-rule SAC prior programs must define at least two rules")

        for name, function in self.rules.items():
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*", name
            ):
                raise ValueError(f"Invalid rule name {name!r}")
            if not callable(function):
                raise TypeError(f"Rule {name!r} must be callable")

        self.reset()

    def reset(self):
        self._selected_rule_name = None
        self._lambda_value = 0.0
        self._last_selection_step = None
        self._last_step_index = -1
        self._has_selected = False

    def _select(self, observation: np.ndarray):
        if self.legacy:
            return "legacy_predict", self.legacy_lambda
        selection = self.selector(observation.copy())
        if not isinstance(selection, dict):
            raise TypeError("select_rule must return a mapping")
        name = selection.get("rule_name")
        value = selection.get("lambda")
        if name in (None, "none"):
            name = None
            if value is None:
                value = 0.0
        elif not isinstance(name, str) or name not in self.rules:
            raise ValueError(f"select_rule returned unknown rule name {name!r}")
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("The rule lambda must be a numeric scalar, not bool")
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("The selected rule lambda must be finite and in [0, 1]")
        if name is None and value != 0.0:
            raise ValueError("select_rule must return lambda=0 when rule_name is None")
        return name, value

    @staticmethod
    def _no_prior(batch_size: int) -> PriorOutput:
        zeros = np.zeros((batch_size, 1), dtype=np.float32)
        inactive = np.zeros((batch_size, 1), dtype=bool)
        return validate_prior_output({
            "driving": {"mean": zeros.copy(), "std": zeros.copy(), "active": inactive.copy()},
            "energy": {"mean": zeros.copy(), "std": zeros.copy(), "active": inactive.copy()},
        }, batch_size=batch_size)

    def get(self, observation, step_index: int) -> RulePriorDecision:
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim == 1:
            if values.shape != (15,):
                raise ValueError("Rule router input must have shape (15,)")
            batch = values.reshape(1, 15)
        elif values.ndim == 2 and values.shape[1] == 15 and values.shape[0] == 1:
            batch = values
        else:
            raise ValueError("Rule router selects one environment state at a time")
        if not np.all(np.isfinite(batch)):
            raise ValueError("Rule router observation must contain only finite values")
        step_index = int(step_index)
        if step_index < 0 or step_index < self._last_step_index:
            raise ValueError("step_index must be nonnegative and monotonic within an episode")

        # A next-state prior is requested while storing the preceding replay
        # transition and then requested again for the actual action. Reuse the
        # selection when both requests refer to the same environment step.
        should_select = (
            not self._has_selected
            or (
                step_index > self._last_step_index
                and step_index % self.selection_interval_steps == 0
            )
        )
        if should_select:
            self._selected_rule_name, self._lambda_value = self._select(batch[0])
            self._last_selection_step = step_index
            self._has_selected = True
        self._last_step_index = step_index

        if self._selected_rule_name is None:
            prior = self._no_prior(batch.shape[0])
        else:
            output = self.rules[self._selected_rule_name](batch.copy())
            prior = validate_prior_output(output, batch_size=batch.shape[0])
        return RulePriorDecision(
            rule_name=self._selected_rule_name,
            lambda_value=self._lambda_value,
            prior=prior,
            selection_updated=should_select,
        )

    def registry(self):
        return list(self.rules)
