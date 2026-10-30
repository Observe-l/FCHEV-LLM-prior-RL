# LLM/rule policy priors

This folder documents both HASAC Gaussian priors and deterministic MADDPG
rule actions. The interfaces are different: HASAC rules return mean/std;
MADDPG rules return one deterministic value per actor and do not use a std.

## MADDPG rule modes

MADDPG rules receive the full raw 15-value environment observation and return
both `driving` and `energy` entries. Each may include `active` (default
`True`). The output mode determines the field and units:

| CLI mode | Rule field | Units | Fusion |
|---|---|---|---|
| `physical_action` | `action` | Driving: m/s²; energy: kW | Convert actor to physical units, blend, convert back |
| `normalized_action` | `action` | Each scalar in `[-1,1]` | Blend normalized actions, then map to physical units |
| `pre_tanh` | `prior` (legacy `mean` accepted; `std` ignored) | Deterministic latent value | Blend actor/rule latent values, apply tanh, then map |

Fusion uses one lambda in `[0,1]` for both channels. The blend happens in the
selected mode's space: physical units for `physical_action`, normalized units
for `normalized_action`, and pre-tanh latent values for `pre_tanh`.

```text
physical_action:  a_fused = (1 - λ) * a_actor_physical + λ * a_rule_physical
normalized_action: u_fused = (1 - λ) * u_actor + λ * u_rule
pre_tanh:          z_fused = (1 - λ) * z_actor + λ * z_rule
                   u_fused = tanh(z_fused)
```

`active=False` bypasses fusion for that actor. Training adds the configured
Gaussian exploration noise to the actor's normalized action before fusion. In
`pre_tanh` mode, the behavior action is numerically mapped back with `atanh`;
actor-gradient and target updates use the network latent directly.
Fresh runs have 1,000 random warm-up steps without the prior; resumed runs
skip that warm-up.

Example:

```bash
python train.py \
  --config config/experiments/maddpg_individual_reward_g099_500ep.json \
  --algorithm maddpg \
  --prior-program-name fchev_rule_prior \
  --maddpg-rule-output-mode pre_tanh \
  --maddpg-rule-lambda 0.2
```

Three standalone English prompts for generating MADDPG rules are:

- `prompts/fchev_maddpg_rule_prior_physical_action_en.md`
- `prompts/fchev_maddpg_rule_prior_normalized_action_en.md`
- `prompts/fchev_maddpg_rule_prior_pre_tanh_en.md`

The environment's driving action is acceleration, not target speed. The
reward and observation facts checked against the implementation are recorded
in `prompts/MADDPG_RULE_PROMPTS_AUDIT.md`.

## HASAC Gaussian policy prior

The existing HASAC interface remains separate. A selected HASAC program
receives the complete raw 15-element environment observation and returns
Gaussian prior parameters for both agents in their pre-`tanh` latent space.
It does not return a physical acceleration or power command.

## Selectable Python program interface

The program name is set externally with `llm_prior.program_name` in the
experiment JSON, or overridden on the command line with
`--prior-program-name PROGRAM_NAME`. The prior must also have
`llm_prior.enabled: true` in the config. For example:

```json
"llm_prior": {
  "enabled": true,
  "program_name": "speed_limit_rule",
  "lambda_0": 0.5,
  "lambda_1": 0.5
}
```

HASAC imports `llm-prior/programs/<program_name>.py` and calls its
`predict(observation)` function. Add or replace a module in that directory;
the generated program does not need to change HASAC, `train.py`, or the
fusion code. `observation` is a NumPy `float32` array shaped `(15,)` for a
single state or `(batch, 15)` for a batch. The function returns a mapping for
both agents:

```python
{
    "driving": {"mean": [0.0], "std": [0.1], "active": True},
    "energy":  {"mean": [0.0], "std": [0.1], "active": True},
}
```

The required outputs are `mean` and nonnegative `std`, each with one value per
agent action dimension. The optional `active` mask defaults to `True`; return
`False` for dimensions/states where the prior should not be fused. A single
program handles the full observation and can give a prior to either or both
agents. Output values are latent Gaussian parameters, before `tanh`, not
normalized actions or physical controls. An optional `build_prior(config)`
factory can be used for configurable example programs; plain LLM-generated
modules can just expose `predict(observation)`.

The currently selected smoke-test program is
`llm-prior/programs/speed_limit_rule.py`. That is the program to replace with
an LLM-generated Python module for **HASAC**, or select another module by
changing `program_name`. It is not a MADDPG rule module. `bridge.py` is a
separate adapter for calling a hosted/runtime LLM that returns JSON; it is not
required when the LLM has already generated a Python program.

## Contract

The raw observation order is declared in `contract.py`:

| Index | Value | Unit / encoding |
|---:|---|---|
| 0–3 | battery SOC, battery SOH, FCS SOH, EDU SOH | fraction |
| 4–7 | ego acceleration, ego speed, lead acceleration, lead speed | m/s², m/s, m/s², m/s |
| 8 | front gap | m |
| 9 | next signal state | red 0, green 1, yellow 2 |
| 10–11 | signal distance, phase remaining | m, s |
| 12–14 | current limit, next limit, distance to next speed region | m/s, m/s, m |

Each agent output has `mean` and `std`, with one scalar per agent action;
`active` optionally masks prior application. The mean and standard deviation
are fused independently:

\[
\mu_f=(1-\lambda_0)\mu_{actor}+\lambda_0\mu_{prior},\qquad
\sigma_f=(1-\lambda_1)\sigma_{actor}+\lambda_1\sigma_{prior}.
\]

The current example uses `lambda_0=lambda_1=0.5`. `active=false` masks out a
prior for that agent/state, leaving both actor parameters unchanged. Sampling
then follows `z = mu_f + sigma_f * epsilon`, `epsilon ~ Normal(0,1)`, followed
by `tanh(z)`. HASAC's entropy log probability is computed from this fused
distribution.

## Included rule and bridge

`programs/speed_limit_rule.py` implements the smoke-test prior. When ego speed exceeds
the **current** posted speed limit (raw observation indices 5 and 12), it
activates only the driving prior. The desired normalized action is `-0.999`,
represented in latent space as `atanh(-0.999)`; its prior std is zero. This is
near the maximum-deceleration action, while avoiding an infinite latent mean.
The energy prior is inactive because this rule makes no FCS recommendation.

`bridge.py` provides a vendor-neutral `CallableLLMPriorBridge`: provide a
callable that accepts the system prompt and a JSON observation payload and
returns JSON matching the contract. Identical observations are cached so an
off-policy update does not repeatedly query an LLM. The prompt is in
`prompts/gaussian_rule_prior.md`. No LLM service or credentials are required by
the included deterministic rule.

The runnable HASAC example configuration is
`config/experiments/hasac_zone_enabled_road60_max70_tlgreen5_pass50_speedbeta100_gamma0p99_llm_rule_prior.json`.
