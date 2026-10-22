from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contract import OBSERVATION_FIELDS, observation_batch, validate_prior_output


class CallableLLMPriorBridge:
    """Adapter for a caller-supplied LLM function using the shared JSON contract.

    ``llm_call`` receives ``(system_prompt: str, user_payload: str)`` and must
    return a JSON string or mapping with ``driving`` and ``energy`` Gaussian
    outputs. A small in-memory cache avoids repeated calls for identical states
    during off-policy actor updates. This module intentionally does not bind to
    a vendor SDK or require credentials.
    """

    def __init__(self, llm_call, prompt_path=None, cache=True):
        if not callable(llm_call):
            raise TypeError("llm_call must be callable")
        self.llm_call = llm_call
        path = Path(prompt_path) if prompt_path else Path(__file__).parent / "prompts" / "gaussian_rule_prior.md"
        self.system_prompt = path.read_text(encoding="utf-8")
        self.cache_enabled = bool(cache)
        self._cache = {}

    def _single(self, row):
        key = np.asarray(row, dtype=np.float32).tobytes()
        if self.cache_enabled and key in self._cache:
            return self._cache[key]
        payload = json.dumps(
            {"observation": dict(zip(OBSERVATION_FIELDS, map(float, row)))},
            ensure_ascii=False,
        )
        response = self.llm_call(self.system_prompt, payload)
        if isinstance(response, str):
            response = json.loads(response)
        checked = validate_prior_output(response, batch_size=1)
        if self.cache_enabled:
            self._cache[key] = checked
        return checked

    def predict(self, observation):
        batch, _ = observation_batch(observation)
        outputs = [self._single(row) for row in batch]
        merged = {}
        for name in ("driving", "energy"):
            merged[name] = {
                field: np.concatenate([
                    getattr(output.for_agent(name), field) for output in outputs
                ], axis=0)
                for field in ("mean", "std", "active")
            }
        return validate_prior_output(merged, batch_size=batch.shape[0])
