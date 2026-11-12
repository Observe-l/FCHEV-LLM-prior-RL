"""Episode-boundary LLM rule updates over Strata's OpenAI-compatible API.

This module is opt-in. It does not make per-step LLM calls: the current Python
rule evaluates locally at each step, and the LLM is contacted only after an
episode has completed. It supports both SAC rule-prior updates and rule-only
controller updates.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace
import urllib.error
import urllib.request

import numpy as np

safe_math = math


RULE_UPDATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "change_rule": {"type": "boolean"},
        "rule_program": {"type": ["string", "null"]},
        "change_lambda": {"type": "boolean"},
        "lambda": {
            "type": ["object", "null"],
            "properties": {
                "driving": {"type": "number"},
                "energy": {"type": "number"},
            },
            "required": ["driving", "energy"],
            "additionalProperties": False,
        },
    },
    "required": ["change_rule", "rule_program", "change_lambda", "lambda"],
    "additionalProperties": False,
}

RULE_ONLY_UPDATE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "change_rule": {"type": "boolean"},
        "rule_program": {"type": ["string", "null"]},
    },
    "required": ["change_rule", "rule_program"],
    "additionalProperties": False,
}


OBSERVATION_FIELDS = (
    "battery_soc", "battery_soh", "fcs_soh", "edu_soh",
    "ego_acceleration_m_s2", "ego_speed_m_s",
    "lead_acceleration_m_s2", "lead_speed_m_s", "following_distance_m",
    "next_signal_state", "next_signal_distance_m",
    "next_signal_phase_remaining_s", "current_speed_limit_m_s",
    "next_speed_limit_m_s", "distance_to_next_speed_region_m",
)

TRACE_FIELDS_FOR_UPDATE = (
    # Compact state/action trace: physical observation, physical commands,
    # agent rewards, and actual safety/termination outcomes only.
    "time_s", "driving_agent_reward", "energy_agent_reward",
    "policy_obs_soc", "policy_obs_battery_soh", "policy_obs_fcs_soh",
    "policy_obs_edu_soh", "policy_obs_ego_acceleration_m_s2",
    "policy_obs_ego_speed_m_s", "policy_obs_lead_acceleration_m_s2",
    "policy_obs_lead_speed_m_s", "policy_obs_following_distance_m",
    "policy_obs_next_signal_state", "policy_obs_next_signal_distance_m",
    "policy_obs_signal_phase_remaining_s", "policy_obs_current_speed_limit_m_s",
    "policy_obs_next_speed_limit_m_s", "policy_obs_distance_to_next_speed_region_m",
    "requested_acceleration_m_s2", "action_executed_acceleration_m_s2",
    "action_requested_fcs_power_raw_kw", "action_executed_fcs_power_kw",
    "traffic_light_state_at_crossing", "traffic_light_crossing",
    "red_light_violation", "speeding_violation", "collision_event",
    "gap_overlap", "terminated_reason",
)

EPISODE_HISTORY_FIELDS = (
    "episode", "return", "driving_agent_return", "energy_agent_return",
    "steps", "termination_reason", "truncation_reason",
    "active_rule_version", "active_rule_sha256",
)

CURRENT_EPISODE_FIELDS = (
    "episode", "return", "driving_agent_return", "energy_agent_return", "steps",
    "termination_reason", "truncation_reason", "hydrogen_g", "collisions",
    "collision_events", "red_light_violations", "speeding_violations",
    "traffic_lights_passed", "lead_traffic_lights_passed", "terminal_penalty",
    "safety_violations", "constraint_clipping_steps", "minimum_gap_m",
    "final_soc", "final_battery_soh", "final_fcs_soh", "final_edu_soh",
    "final_core_temperature_c", "final_surface_temperature_c",
    "active_rule_version", "active_rule_sha256",
)

SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "iter", "len", "list", "map", "max", "min",
    "next", "range", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "zip", "ValueError", "TypeError", "RuntimeError",
    "AssertionError", "IndexError", "KeyError", "OverflowError", "ZeroDivisionError",
)
FORBIDDEN_CALL_NAMES = {
    "eval", "exec", "compile", "open", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint", "__import__",
}
SAFE_NUMPY_NAMES = (
    "abs", "all", "any", "arange", "arctan", "arctanh", "array", "asarray",
    "broadcast_to", "clip", "column_stack", "concatenate", "cos", "exp",
    "expm1", "errstate", "bool_", "float32", "float64", "int32", "int64",
    "full", "inf", "isfinite",
    "isnan", "linspace", "log", "maximum", "mean", "minimum", "nan_to_num",
    "nan", "ones", "ones_like", "pi", "sign", "sin", "sqrt", "stack", "sum",
    "tanh", "where", "zeros", "zeros_like",
)
SAFE_NUMPY = SimpleNamespace(
    **{name: getattr(np, name) for name in SAFE_NUMPY_NAMES},
    linalg=SimpleNamespace(norm=np.linalg.norm),
)


class RuleProgramError(ValueError):
    """A generated rule failed static or interface validation."""


class StrataRequestError(RuntimeError):
    """A Strata request failed while retaining response details for diagnostics."""

    def __init__(self, message, *, response_text=None, response_metadata=None):
        super().__init__(message)
        self.response_text = response_text
        self.response_metadata = response_metadata or {}


class CompiledRuleProvider:
    """A validated local rule module executed with a small builtin surface."""

    def __init__(self, source: str):
        tree = _validate_rule_ast(source)
        namespace = {
            "__builtins__": _safe_builtins(),
            "__name__": "fchev_runtime_rule",
            "np": SAFE_NUMPY,
        }
        try:
            code = compile(tree, "<llm-generated-rule>", "exec")
            exec(code, namespace, namespace)
        except Exception as exc:
            raise RuleProgramError(f"rule module failed to load: {exc}") from exc
        predict = namespace.get("predict")
        if not callable(predict):
            raise RuleProgramError("rule module must define callable predict(observation)")
        self._predict = predict

    def predict(self, observation):
        return self._predict(observation)


def _safe_builtins():
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTIN_NAMES}

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name == "numpy":
            return SAFE_NUMPY
        if level == 0 and name == "math":
            return safe_math
        if level == 0 and name == "__future__":
            import __future__
            return __future__
        raise ImportError(f"generated rules may import only numpy; rejected {name!r}")

    safe["__import__"] = safe_import
    return safe


def _validate_rule_ast(source: str):
    if not isinstance(source, str) or not source.strip():
        raise RuleProgramError("rule program is empty")
    if len(source.encode("utf-8")) > 250_000:
        raise RuleProgramError("rule program exceeds 250 kB")
    try:
        tree = ast.parse(source, filename="<llm-generated-rule>", mode="exec")
    except SyntaxError as exc:
        raise RuleProgramError(f"rule program syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ("numpy", "math"):
                    raise RuleProgramError(f"unsupported import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module not in ("numpy", "math") and not (
                node.module == "__future__"
                and all(alias.name == "annotations" for alias in node.names)
            ):
                raise RuleProgramError(f"unsupported import from: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_NAMES:
                raise RuleProgramError(f"unsupported function call: {node.func.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise RuleProgramError("private/dunder attribute access is not allowed in generated rules")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise RuleProgramError("dunder names are not allowed in generated rules")

    return tree


def compile_rule_provider(source: str):
    """Load a candidate and verify its output against the shared prior contract."""
    provider = CompiledRuleProvider(source)
    raw_obs = np.asarray(
        [0.5, 1.0, 1.0, 1.0, 0.0, 8.0, 0.0, 8.0, 30.0,
         0.0, 80.0, 20.0, 8.3333333, 16.6666667, 80.0],
        dtype=np.float32,
    )
    contract = import_module("llm-prior.contract")
    try:
        contract.validate_prior_output(provider.predict(raw_obs), batch_size=1)
        # SAC evaluates the rule over replay batches during gradient updates.
        # A candidate that works for one observation can still broadcast a
        # (batch,) vector against a (batch, 1) column into (batch, batch).
        batch_obs = np.repeat(raw_obs.reshape(1, -1), 64, axis=0)
        contract.validate_prior_output(provider.predict(batch_obs), batch_size=64)
    except Exception as exc:
        raise RuleProgramError(
            "predict(observation) violates the single-row or batch prior contract: "
            f"{exc}"
        ) from exc
    return provider


def _to_json_value(value):
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        if not math.isfinite(number):
            return None
        if number.is_integer():
            return int(number)
        return number
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _select_fields(row, fields):
    return {key: _to_json_value(row.get(key)) for key in fields if key in row}


def _episode_history(rows):
    return [_select_fields(row, EPISODE_HISTORY_FIELDS) for row in rows]


def _current_episode(row):
    result = _select_fields(row, CURRENT_EPISODE_FIELDS)
    result["reward_component_totals"] = {
        key.removeprefix("reward_component_"): _to_json_value(value)
        for key, value in row.items()
        if key.startswith("reward_component_") and key != "reward_component_total"
    }
    result["reward_component_total"] = _to_json_value(row.get("reward_component_total"))
    return result


def _final_trace_window(trace, window=20):
    selected = []
    for index, row in enumerate(trace[-window:], start=max(1, len(trace) - window + 1)):
        selected.append({"step": index, **_select_fields(row, TRACE_FIELDS_FOR_UPDATE)})
    return selected


def build_episode_context(history_rows, current_row, trace, current_lambdas):
    """Compact complete episode history plus the current episode's final 20 steps."""
    if not isinstance(current_lambdas, dict) or set(current_lambdas) != {"driving", "energy"}:
        raise ValueError("current_lambdas must contain driving and energy values")
    return {
        "current_lambda": {
            name: float(current_lambdas[name]) for name in ("driving", "energy")
        },
        "lambda_scope": (
            "one independently adjustable value per agent; each agent's value is shared "
            "by that agent's mean and std fusion weights"
        ),
        "episode_history": _episode_history(history_rows),
        "current_episode": _current_episode(current_row),
        "final_20_steps_before_episode_end": _final_trace_window(trace, 20),
    }


def build_rule_only_episode_context(history_rows, current_row, trace):
    """Episode history and a compact trace for rule-only online updates."""
    return {
        "episode_history": _episode_history(history_rows),
        "current_episode": _current_episode(current_row),
        "final_20_steps_before_episode_end": _final_trace_window(trace, 20),
    }


def _parse_llm_json(content):
    if not isinstance(content, str):
        raise ValueError("chat completion content is not text")
    text = content.strip()
    # Accept a fenced JSON response while still requiring one JSON object.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response does not contain a JSON object")
    value, end = json.JSONDecoder().raw_decode(text[start:])
    if text[start + end:].strip():
        raise ValueError("LLM response must contain only one JSON object")
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON must be an object")
    required = {"change_rule", "rule_program", "change_lambda", "lambda"}
    if set(value) != required:
        raise ValueError(f"LLM response keys must be exactly {sorted(required)}")
    for key in ("change_rule", "change_lambda"):
        if not isinstance(value[key], bool):
            raise ValueError(f"{key} must be a JSON boolean")
    if value["change_rule"]:
        if not isinstance(value["rule_program"], str) or not value["rule_program"].strip():
            raise ValueError("rule_program must contain the full Python source when change_rule=true")
    elif value["rule_program"] is not None:
        raise ValueError("rule_program must be null when change_rule=false")
    if value["change_lambda"]:
        lam = value["lambda"]
        if not isinstance(lam, dict) or set(lam) != {"driving", "energy"}:
            raise ValueError("lambda must contain exactly driving and energy numbers")
        for name in ("driving", "energy"):
            agent_lambda = lam[name]
            if isinstance(agent_lambda, bool) or not isinstance(agent_lambda, (int, float)):
                raise ValueError(f"lambda.{name} must be a number")
            if not math.isfinite(float(agent_lambda)) or not 0.0 <= float(agent_lambda) <= 1.0:
                raise ValueError(f"lambda.{name} must be finite and lie in [0, 1]")
            lam[name] = float(agent_lambda)
    elif value["lambda"] is not None:
        raise ValueError("lambda must be null when change_lambda=false")
    return value


def _parse_rule_only_llm_json(content):
    if not isinstance(content, str):
        raise ValueError("chat completion content is not text")
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response does not contain a JSON object")
    value, end = json.JSONDecoder().raw_decode(text[start:])
    if text[start + end:].strip():
        raise ValueError("LLM response must contain only one JSON object")
    if not isinstance(value, dict) or set(value) != {"change_rule", "rule_program"}:
        raise ValueError("rule-only response must contain exactly change_rule and rule_program")
    if not isinstance(value["change_rule"], bool):
        raise ValueError("change_rule must be a JSON boolean")
    if value["change_rule"]:
        if not isinstance(value["rule_program"], str) or not value["rule_program"].strip():
            raise ValueError("rule_program must contain the full Python source when change_rule=true")
    elif value["rule_program"] is not None:
        raise ValueError("rule_program must be null when change_rule=false")
    return value


def _normalize_rule_source(source):
    """Remove common transport wrappers without evaluating generated code."""
    candidate = source.strip()
    repairs = []
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
            repairs.append("removed_outer_markdown_fence")
    try:
        decoded = ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        decoded = None
    if isinstance(decoded, str) and "def predict" in decoded:
        candidate = decoded.strip()
        repairs.append("unwrapped_python_string_literal")
    return candidate, repairs


class StrataEpisodeRuleUpdater:
    """Calls Strata once at episode end and validates the returned rule update."""

    def __init__(
        self, *, base_url="http://127.0.0.1:8080/v1", model="auto",
        api_key_env="STRATA_API_KEY", timeout_s=900.0, max_tokens=None,
        temperature=1.0, top_p=0.95, top_k=20,
        reasoning_effort="low", enable_thinking=True, mode="sac",
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self.max_tokens = (
            None if max_tokens is None or int(max_tokens) in (0, -1)
            else int(max_tokens)
        )
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.reasoning_effort = str(reasoning_effort)
        self.enable_thinking = bool(enable_thinking)
        if mode not in ("sac", "rule_only"):
            raise ValueError("online rule update mode must be 'sac' or 'rule_only'")
        self.mode = mode
        template_name = (
            "fchev_llm_rule_only_episode_update_en.md"
            if mode == "rule_only"
            else "fchev_llm_rule_prior_episode_update_en.md"
        )
        self.template_path = Path(__file__).parent / "prompts" / template_name
        self.template = self.template_path.read_text(encoding="utf-8")

    @property
    def parameters(self):
        parameters = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "reasoning_effort": self.reasoning_effort,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": (
                        "fchev_rule_only_update"
                        if self.mode == "rule_only"
                        else "fchev_rule_update"
                    ),
                    "strict": True,
                    "schema": (
                        RULE_ONLY_UPDATE_RESPONSE_SCHEMA
                        if self.mode == "rule_only"
                        else RULE_UPDATE_RESPONSE_SCHEMA
                    ),
                },
            },
        }
        # Omit max_tokens to leave output uncapped by a separate completion
        # limit; Strata will stop at the remaining model context window.
        if self.max_tokens is not None:
            parameters["max_tokens"] = self.max_tokens
        return parameters

    def check_server(self):
        """Fail early if the configured local Strata endpoint is unreachable."""
        request = urllib.request.Request(f"{self.base_url}/models", method="GET")
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            request.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(request, timeout=min(self.timeout_s, 10.0)) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach Strata at {self.base_url}: {exc}"
            ) from exc
        models = [item.get("id") for item in payload.get("data", []) if item.get("id")]
        if self.model == "auto":
            if len(models) == 1:
                self.model = models[0]
            else:
                qwen_models = [name for name in models if "qwen3.8" in name.lower()]
                if len(qwen_models) == 1:
                    self.model = qwen_models[0]
                else:
                    raise RuntimeError(
                        "Strata model auto-detection needs exactly one loaded model or one "
                        f"Qwen3.8 model; /v1/models returned {models}"
                    )
        return payload

    def render_prompt(self, *, current_rule_source, context):
        prompt = self.template.replace("<<<CURRENT_RULE_PROGRAM>>>", current_rule_source)
        prompt = prompt.replace(
            "<<<EPISODE_CONTEXT_JSON>>>",
            json.dumps(context, ensure_ascii=False, indent=2, allow_nan=False),
        )
        if "<<<" in prompt:
            raise ValueError("unfilled placeholder remains in online rule update prompt")
        return prompt

    def _chat(self, prompt):
        # Each episode is a fresh, stateless request: no conversation ID,
        # earlier user message, or earlier assistant response is sent. The
        # prompt itself contains the requested training-log summary only.
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **self.parameters,
        }
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        if key:
            request.add_header("Authorization", f"Bearer {key}")
        request_metadata = {
            "prompt_chars": len(prompt),
            "prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "max_output_tokens": self.max_tokens,
            "output_cap": "unset" if self.max_tokens is None else "configured",
            "response_format": "json_schema",
            "reasoning_effort": self.reasoning_effort,
            "enable_thinking": self.enable_thinking,
        }
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            request_metadata["http_status"] = int(exc.code)
            raise StrataRequestError(
                f"Strata returned HTTP {exc.code}: {detail}",
                response_text=detail,
                response_metadata=request_metadata,
            ) from exc
        except json.JSONDecodeError as exc:
            raise StrataRequestError(
                f"Strata response was not valid JSON: {exc}",
                response_text=exc.doc[:4000],
                response_metadata=request_metadata,
            ) from exc
        choices = payload.get("choices", [])
        if not choices:
            raise StrataRequestError(
                "Strata chat-completions response has no choices",
                response_text=json.dumps(payload, ensure_ascii=False)[:4000],
                response_metadata={**request_metadata, "usage": payload.get("usage")},
            )
        choice = choices[0]
        request_metadata.update({
            "response_id": payload.get("id"),
            "response_model": payload.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": payload.get("usage"),
        })
        message = choices[0].get("message", {})
        return {
            "content": message.get("content"),
            "metadata": request_metadata,
        }

    def request_update(self, *, current_rule_source, context):
        prompt = self.render_prompt(current_rule_source=current_rule_source, context=context)
        try:
            completion = self._chat(prompt)
        except StrataRequestError as exc:
            return {
                "decision": None,
                "response_text": exc.response_text,
                "response_metadata": exc.response_metadata,
                "response_error": str(exc),
                "rendered_prompt": prompt,
                "candidate_provider": None,
                "candidate_source": current_rule_source,
                "candidate_error": None,
            }
        except Exception as exc:
            return {
                "decision": None,
                "response_text": None,
                "response_metadata": {
                    "prompt_chars": len(prompt),
                    "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                    "max_output_tokens": self.max_tokens,
                    "output_cap": "unset" if self.max_tokens is None else "configured",
                },
                "response_error": f"{type(exc).__name__}: {exc}",
                "rendered_prompt": prompt,
                "candidate_provider": None,
                "candidate_source": current_rule_source,
                "candidate_error": None,
            }
        response_content = completion["content"]
        response_text = (
            response_content
            if isinstance(response_content, str)
            else json.dumps(response_content, ensure_ascii=False)
        )
        response_metadata = completion["metadata"]
        finish_reason = response_metadata.get("finish_reason")
        usage = response_metadata.get("usage") or {}
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        try:
            completion_token_count = int(completion_tokens)
        except (TypeError, ValueError):
            completion_token_count = None
        output_limit_reached = finish_reason in {"length", "max_tokens"} or (
            self.max_tokens is not None
            and completion_token_count is not None
            and completion_token_count >= self.max_tokens
        )
        if output_limit_reached:
            if self.max_tokens is None:
                limit_message = (
                    "LLM output ended at the model/context limit despite no separate "
                    f"max_tokens cap (finish_reason={finish_reason!r}, "
                    f"completion_tokens={completion_token_count!r})"
                )
            else:
                limit_message = (
                    f"LLM output reached max_tokens={self.max_tokens} "
                    f"(finish_reason={finish_reason!r}, "
                    f"completion_tokens={completion_token_count!r})"
                )
            return {
                "decision": None,
                "response_text": response_text,
                "response_metadata": response_metadata,
                "response_error": limit_message,
                "rendered_prompt": prompt,
                "candidate_provider": None,
                "candidate_source": current_rule_source,
                "candidate_error": None,
            }
        try:
            decision = (
                _parse_rule_only_llm_json(response_text)
                if self.mode == "rule_only"
                else _parse_llm_json(response_text)
            )
        except Exception as exc:
            return {
                "decision": None,
                "response_text": response_text,
                "response_metadata": response_metadata,
                "response_error": f"{type(exc).__name__}: {exc}",
                "rendered_prompt": prompt,
                "candidate_provider": None,
                "candidate_source": current_rule_source,
                "candidate_error": None,
            }
        candidate_provider = None
        candidate_error = None
        candidate_normalization = []
        candidate_source = current_rule_source
        if decision["change_rule"]:
            candidate_source, candidate_normalization = _normalize_rule_source(
                decision["rule_program"]
            )
            decision["rule_program"] = candidate_source
            try:
                candidate_provider = compile_rule_provider(candidate_source)
            except Exception as exc:
                candidate_error = str(exc)
                candidate_source = current_rule_source
        return {
            "decision": decision,
            "response_text": response_text,
            "response_metadata": response_metadata,
            "response_error": None,
            "rendered_prompt": prompt,
            "candidate_provider": candidate_provider,
            "candidate_source": candidate_source,
            "candidate_error": candidate_error,
            "candidate_normalization": candidate_normalization,
        }

    @staticmethod
    def source_sha256(source):
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    def save_initial_artifacts(self, run_dir, source):
        updates_dir = Path(run_dir) / "online_rule_updates"
        updates_dir.mkdir(parents=True, exist_ok=True)
        (updates_dir / "prompt_template.md").write_text(self.template, encoding="utf-8")
        initial_path = updates_dir / "rule_v000_episode_0000.py"
        initial_path.write_text(source, encoding="utf-8")
        return {
            "directory": str(updates_dir),
            "rule_version": 0,
            "rule_source_path": str(initial_path),
            "rule_source_sha256": self.source_sha256(source),
        }

    def write_episode_record(
        self, *, run_dir, episode, context, result, previous_version,
        current_version, current_source, accepted_rule, accepted_lambda=False,
        previous_lambdas=None, current_lambdas=None, error=None,
    ):
        updates_dir = Path(run_dir) / "online_rule_updates"
        updates_dir.mkdir(parents=True, exist_ok=True)
        rule_path = updates_dir / f"rule_v{current_version:03d}_after_episode_{episode:04d}.py"
        rule_path.write_text(current_source, encoding="utf-8")
        prompt_path = updates_dir / f"episode_{episode:04d}_prompt.txt"
        response_path = updates_dir / f"episode_{episode:04d}_response.txt"
        if result is not None:
            prompt_path.write_text(result["rendered_prompt"], encoding="utf-8")
            if result.get("response_text") is not None:
                response_path.write_text(result["response_text"], encoding="utf-8")
        record = {
            "episode": int(episode),
            "controller_mode": self.mode,
            "conversation_mode": "stateless single-turn; no previous LLM messages are sent",
            "context": context,
            "request_parameters": self.parameters,
            "prompt_path": str(prompt_path) if result is not None else None,
            "prompt_chars": len(result["rendered_prompt"]) if result is not None else None,
            "prompt_utf8_bytes": (
                len(result["rendered_prompt"].encode("utf-8")) if result is not None else None
            ),
            "response_path": str(response_path) if result and result.get("response_text") is not None else None,
            "response_text": result.get("response_text") if result else None,
            "response_metadata": result.get("response_metadata") if result else None,
            "response_error": result.get("response_error") if result else None,
            "decision": result.get("decision") if result else None,
            "error": error,
            "candidate_rule_error": result.get("candidate_error") if result else None,
            "candidate_rule_normalization": (
                result.get("candidate_normalization") if result else None
            ),
            "rule_updated": bool(accepted_rule),
            "rule_version_before": int(previous_version),
            "rule_version_after": int(current_version),
            "rule_source_sha256": self.source_sha256(current_source),
            "rule_source_path": str(rule_path),
        }
        if self.mode == "sac":
            if previous_lambdas is None or current_lambdas is None:
                raise ValueError("SAC update records require before/after lambdas")
            record.update({
                "lambda_updated": bool(accepted_lambda),
                "lambda_before": {
                    name: float(previous_lambdas[name]) for name in ("driving", "energy")
                },
                "lambda_after": {
                    name: float(current_lambdas[name]) for name in ("driving", "energy")
                },
            })
        record_path = updates_dir / f"episode_{episode:04d}.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        return record_path


def agent_lambdas(agent):
    """Return each agent's scalar lambda, requiring mean/std weights to match per agent."""
    result = {}
    for name in ("driving", "energy"):
        lambda_mean, lambda_std = map(float, agent._agent_lambda_weights(name))
        if abs(lambda_mean - lambda_std) > 1e-9:
            raise ValueError(
                f"online rule updates require {name} lambda_0 and lambda_1 to match"
            )
        result[name] = lambda_mean
    return result

