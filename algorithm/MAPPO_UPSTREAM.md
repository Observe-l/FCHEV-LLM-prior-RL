# MAPPO implementation notes

The default training algorithm is MAPPO. Its clipped policy objective, on-policy rollout/update cycle, centralized value function, and generalized advantage estimation follow the public MAPPO implementation maintained at [marlbenchmark/on-policy](https://github.com/marlbenchmark/on-policy). Useful upstream files are:

- [`r_mappo.py`](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py): clipped PPO policy/value update and advantage handling.
- [`rMAPPOPolicy.py`](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py): decentralized actor and centralized critic policy interface.
- [Upstream license](https://github.com/marlbenchmark/on-policy/blob/main/LICENSE): MIT License.

`algorithm/mappo.py` is the FCHEV-specific adaptation. It uses the same MAPPO update structure while fitting this repository's single Gymnasium environment and two heterogeneous continuous-action actors:

- Driving actor: 11 normalized observation values, scalar acceleration action.
- Energy actor: 6 normalized observation values, scalar FCS power action.
- Central value function: complete normalized 15-value environment observation.
- Value normalization: running return statistics normalize critic targets; the critic output is rescaled when those statistics change, keeping its unnormalized predictions continuous.
- Cooperative reward: the environment reward, checked as driving reward plus energy reward.
- On-policy data: one episode is collected and consumed for PPO updates; there is no replay buffer.
- Action transform: configure `training.mappo_action_transform` as `gaussian_clip` or `tanh`. `gaussian_clip` samples an ordinary Gaussian command `z`, clips it to `[-1, 1]`, and stores/scores the original `z` for PPO. `tanh` computes `a=tanh(z)` and applies the change-of-variables correction to the Gaussian log probability, matching HASAC's normalized action transform. In either case, the normalized command is mapped to physical acceleration and FCS power by `train.py` using the same mapping as HASAC.
- LLM rule prior: when enabled, each actor's prior mean/std is mixed with its policy mean/std before sampling. The same stored prior parameters are used when recomputing PPO likelihoods during updates.

Default MAPPO parameters are in `config/default.json`: gamma 0.9999, GAE lambda 0.95, clip parameter 0.2, 10 PPO epochs and 8 minibatches per episode, entropy coefficient 0.01, value-loss coefficient 1.0, and maximum gradient norm 10.0. `--algorithm hasac` remains available for existing HASAC experiments and checkpoints; checkpoints cannot be loaded across algorithms.
