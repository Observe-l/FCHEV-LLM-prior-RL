# MADDPG prompt facts checked against the implementation

The three `fchev_maddpg_rule_prior_*_en.md` prompts are based on the current
MADDPG experiment configurations and environment code, not the older HASAC
prompt. The older HASAC prompt describes Gaussian mean/std priors and should
not be used as the MADDPG action contract.

## Sources checked

- `config/experiments/maddpg_individual_reward_g099_500ep.json`
- `config/experiments/maddpg_individual_reward_g0999_500ep.json`
- `environment/fchev_env.py` — observation construction, speed-region logic,
  action bounds, reward assembly, feasibility, and termination
- `environment/sumo_interface.py` — lead speed-limit handling and ego SUMO
  speed mode
- `environment/scenario/build_scenario.py` — route, traffic signals, lead IDM,
  and reproducible stop plans
- `util/observation.py` and `train.py` — observation scaling and actor input
  splits
- `util/reward.py` and `train.py` — reward formulas and MADDPG reward split
- `algorithm/networks.py` and `algorithm/maddpg.py` — deterministic tanh
  actors, centralized critics, separate reward targets

## Key consistency points

| Topic | Current implementation/config |
|---|---|
| Policy | Two deterministic actors. Each network returns `tanh(net(state))`; there is no actor `std`. MADDPG exploration noise is trainer-side. |
| Actor inputs | Driving actor: normalized raw features 4–14 (11 values). Energy actor: normalized features `[4,5,0,1,2,3]` (6 values). Rule receives all 15 raw values. |
| Critics/rewards | Two centralized critics see normalized global state (15) plus the joint normalized action (2); each critic learns its own agent's reward. |
| Physical action | Acceleration `[-1.5,0.7] m/s²`; FCS requested power `[0,50] kW`. The environment action is not target speed. |
| Signal timing | 40 s green + 3 s yellow + 47 s red = 90 s; seeded signal phases differ. |
| Speed limits | Ordinary road 60 km/h; ego's explicit intersection zone is 30 km/h within 100 m before the stop line and through the junction. Lead speed is separately capped at 30 km/h within 200 m before a light and inside the junction. |
| Following | `D_safe=2+0.5v+0.0625v²`; `D_max=10+v+0.0825v²`; `d_ref=200 m`; `C0=5,C1=500,C2=5,C3=5`. Far-gap reward is `C3*(forward distance/current active posted speed limit)` in seconds. |
| Signal reward | Prediction term is `+5` if projected arrival precedes phase change and current phase is green, `-5` if yellow/red, multiplied by beta7=1. Green line crossing adds +50 once. |
| Speeding | Continuous penalty is `-100*max(v-limit,0)/limit`. Reaching 120% of active limit terminates: 36 km/h in the 30 zone; 72 km/h on ordinary road, but the hard 70 km/h cap makes the latter unreachable. |
| Termination penalty | Collision/overlap, red-light, and speeding fixed penalties are all configured as 0 in these MADDPG runs. They still end the episode. |
| Speed-entry shaping | beta8=500, assumed braking deceleration=0.75 m/s². It is active only before a lower limit; context-boundary positive jumps and positive collision/overlap terminal-step jumps are suppressed. |
| Episode end | Lead terminal stop or ego route completion terminates normally. 6,000 s limit truncates. 300 m / 1,000 m gap thresholds only log; no stuck termination is active. |

The g099 and g0999 configurations have the same environment/reward settings;
their gamma values differ (0.99 vs 0.999). Prompts intentionally treat gamma
as a training hyperparameter rather than a physical environment constant.

## MADDPG rule CLI

```bash
python train.py \
  --config config/experiments/maddpg_individual_reward_g099_500ep.json \
  --algorithm maddpg \
  --prior-program-name fchev_rule_prior \
  --maddpg-rule-output-mode pre_tanh \
  --maddpg-rule-lambda 0.2
```

This command uses the existing HASAC rule module as a compatibility example;
the MADDPG adapter reads its latent `mean` and ignores its `std`. For newly
generated MADDPG programs, choose the matching mode and output schema from the
three prompts. `normalized_action` and `pre_tanh` are the other mode choices.
