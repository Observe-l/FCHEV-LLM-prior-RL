# Episode-Level LLM Rule Updates

The online updater is optional and currently applies to the independent two-agent SAC run (`--algorithm sac`). The initial rule is always loaded from `llm-prior/programs/fchev_rule_prior.py`; that source file is never overwritten.

## Strata connection

Strata exposes an OpenAI-compatible Chat Completions endpoint. By default, this project connects to `http://127.0.0.1:8080/v1` and discovers the loaded model ID from `/v1/models`. The server currently reports `qwen3.8-flash-next-iq3_xxs`. If Strata is configured with an API key, put the key in the environment variable `STRATA_API_KEY`; do not put it in a config file. For a different address or model ID, pass `--llm-base-url` or `--llm-model`.

Check the local server and its loaded model from a terminal:

```bash
curl http://127.0.0.1:8080/v1/models
```

The updater enables thinking at the `low` reasoning effort for each request. Strata documents `reasoning_effort` values including `low`; this keeps reasoning enabled but short. It is a reasoning preference, not an output-token cap.

Example direct request using the same low-thinking and sampling settings as the training updater:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-flash-next-iq3_xxs",
    "messages": [{"role": "user", "content": "Reply with the single word ready."}],
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "reasoning_effort": "low",
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 64
  }'
```

Equivalent call using the OpenAI Python client (install `openai` only if you want to run this standalone snippet; the project updater itself does not need it):

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key=os.getenv("STRATA_API_KEY", "local"),
)
response = client.chat.completions.create(
    model="qwen3.8-flash-next-iq3_xxs",
    messages=[{"role": "user", "content": "Reply with the single word ready."}],
    temperature=1.0,
    top_p=0.95,
    extra_body={
        "top_k": 20,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"enable_thinking": True},
    },
    max_tokens=64,
)
print(response.choices[0].message.content)
```

The Python updater uses the standard library `urllib` client, so no OpenAI Python package is required. Every call is a fresh, stateless, single-turn request with one user message; it does not send prior LLM prompts or replies or use a conversation ID. The episode reward/steps/end-reason history is included as compact training data, as required by the update prompt. The LLM is called once after each completed episode and never from the per-step rule evaluation path.

The updater requests a strict JSON-schema response and does not set a separate output-token cap by default; generation is bounded by the remaining model context window. Set `--llm-max-tokens N` only when an explicit output cap is desired (`0` or `-1` also means no separate cap). The local model reported 131,072 context tokens. Strata validates the response structure but does not guarantee that a string field is executable Python; the project compiles each candidate and validates its rule interface before installation.

## Run with online updates

```bash
conda run -n default python train.py \
  --config config/default.json \
  --algorithm sac \
  --episodes 200 \
  --seed 7 \
  --prior-program-name fchev_rule_prior \
  --enable-rule-prior \
  --prior-lambda 0.2 \
  --enable-online-rule-updates \
  --experiment-name sac_online_rule_updates
```

The LLM receives the original rule prompt's environment/task/observation sections, the complete active rule source, all episode-level returns/steps/end reasons, current episode reward-component totals and metrics, and a compact trace of up to 20 steps. Each trace row contains raw physical observations, requested/executed physical actions, per-agent rewards, and key traffic/safety outcomes; normalized observations, prior-distribution details, duplicate action fields, and per-step reward-component breakdowns are omitted. A returned rule is compiled, checked against a restricted import/builtin surface, and validated against the prior output contract before it becomes active. A failed HTTP request or rejected rule leaves the prior rule and lambda active and is logged; training continues.

Every completed call also saves the raw response as `episode_NNNN_response.txt`. Its JSON record stores `finish_reason`, token usage, prompt size, parse errors, and rule validation errors so output truncation can be distinguished from malformed JSON or an invalid Python program.

The returned `lambda` is a single scalar in `[0,1]`. It updates both agents' `lambda_0` (mean) and `lambda_1` (standard-deviation) fusion weights. The update is installed after the current episode and affects the next episode.

## Run artifacts

Each run stores online-update artifacts in `results/<run>/online_rule_updates/`:

- `prompt_template.md`: exact update prompt template used by the run.
- `rule_v000_episode_0000.py`: untouched initial rule snapshot.
- `episode_NNNN_prompt.txt`: complete prompt sent after that episode.
- `episode_NNNN_response.txt`: raw Strata response, including malformed or truncated output when returned.
- `episode_NNNN.json`: episode context, sampling settings, model response, validation result, lambda transition, and active source hash/path.
- `rule_vNNN_after_episode_NNNN.py`: active complete source snapshot after the update decision.

The final and rolling checkpoints also retain the active rule source and version.
