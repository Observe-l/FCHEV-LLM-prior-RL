# Episode-updated LLM rule-only experiment

`train_rule_only.py` runs the FCHEV SUMO environment using only the active rule
program. It creates no SAC/MAPPO/MADDPG actor or critic, replay buffer, gradient
update, or RL checkpoint. The initial source is
`llm-prior/programs/fchev_rule_prior.py`.

At each 1 s environment step, the rule receives the current raw 15-value
observation and returns driving and energy Gaussian parameters. The runner
samples each rule distribution, applies `tanh`, maps the normalized values to
acceleration and FCS output power, and lets the environment apply its usual
physical constraints. It calls the local Strata model once after each episode.
An accepted complete rule replacement becomes active in the next episode. Each
LLM request is stateless and includes the full current rule, episode history,
reward totals, metrics, and a compact final trajectory window.

There is no lambda field or actor/rule fusion in this experiment. The legacy
`active` field is ignored because there is no RL policy to use as a fallback;
the returned mean and standard deviation directly control both action channels
at every step. The initial `fchev_rule_prior.py` emits zero mean and zero
standard deviation for dimensions it marks inactive, so those initial outputs
map to zero requested acceleration and 25 kW FCS power until the LLM replaces
the rule with an explicit all-state policy.

Example command using the local Conda `ray` environment and Strata on port
8080:

```bash
conda run -n ray python -u train_rule_only.py \
  --config config/default.json \
  --episodes 200 \
  --seed 7 \
  --llm-base-url http://127.0.0.1:8080/v1 \
  --experiment-name online_llm_rule_only_200ep
```

Each run gets a unique directory under `results/`. It records `episodes.csv`,
per-episode trajectories, reward/metrics plots, `summary.json`, and
`online_rule_updates/` containing the prompt and response for each update, the
accepted rule versions, and JSON update records. `--config`, `--episodes`,
`--seed`, `--max-steps`, and the Strata connection can be overridden on the
command line.

The episode update prompt is
`llm-prior/prompts/fchev_llm_rule_only_episode_update_en.md`. Its response
schema contains only `change_rule` and `rule_program`; it has no lambda output.
