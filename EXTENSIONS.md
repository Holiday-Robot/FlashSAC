# Project extensions (16-831)

Additions to FlashSAC for the course project. With the default config the training behaviour is unchanged.

## Config keys

All keys live in `configs/agent/flashSAC.yaml` and are set with `--overrides agent.<key>=<value>`.

| Key | Values (default first) | Meaning |
|---|---|---|
| `sampler` | `uniform`, `swd`, `laber` | Replay sampling method |
| `swd_decay_steps` | `2000` | SWD decay horizon, in interaction steps |
| `swd_min_weight` | `0.1` | SWD weight floor |
| `laber_multiplier` | `4` | LaBER candidate batch = multiplier × `sample_batch_size` |
| `actor_delay` | `fixed`, `ema` | Actor delay rule |
| `actor_update_period` | `2` | Delay `d` for `actor_delay=fixed` |
| `actor_delay_ema_interval` | `100` | `k`: critic updates between delay recomputations |
| `actor_delay_ema_beta` | `0.9` | EMA decay, applied once per interval |
| `actor_delay_ema_scale` | `2.0` | Delay when the TD error equals its EMA |
| `actor_delay_ema_min` / `_max` | `1.0` / `8.0` | Clip range of the delay |
| `actor_lr_schedule` | `actor_step`, `global_step` | What drives the actor/temperature learning-rate schedule |

`log_csv` (default `true`, in `configs/flashSAC_base.yaml`) turns the CSV log on or off.

## Methods

**SWD** (`flash_rl/buffers/torch_buffer.py`). WarpSAC's Sample Weight Decay: a transition added `age` interaction
steps ago is sampled with probability ∝ `max(swd_min_weight, 1 - age / swd_decay_steps)`. No importance correction,
as in WarpSAC: both the loss and the BatchNorm statistics follow the recency-biased batch.
This is FlashSAC + SWD, not full WarpSAC (which also changes the critic).

**LaBER** (`FlashSACAgent._select_laber_batch`). Draw `laber_multiplier × B` transitions uniformly, score each with

    surprise_i = mean over the 2 critics of ‖softmax(logits(s_i, a_i)) − target_probs_i‖₂

and keep `B` of them with probability ∝ surprise (with replacement). The importance weights
`w_i = mean(surprise) / surprise_i` correct the critic update for this selection in two places: the loss terms, and the
BatchNorm batch statistics of the critic and target critic (weighted mean and variance). Both corrections are
self-normalised estimates of their uniform-sampling counterparts, so the update is approximately, not exactly,
unbiased. Scoring runs with BatchNorm in eval mode. The actor is updated on a uniform slice of the candidates, not on
the high-surprise selection.

**EMA actor delay** (`flash_rl/agents/flashSAC/actor_delay.py`). Every `interval` critic updates, with `e` the mean
absolute TD error over those updates:

    ema   = beta * ema + (1 - beta) * e
    delay = clip(scale * e / ema, min, max)

The delay is real-valued: each critic update earns `1/delay` credit and an actor update costs 1.

**Actor LR schedule.** In FlashSAC the actor's cosine decay advances once per actor update, so its learning-rate
schedule depends on the delay. `actor_lr_schedule=global_step` ties it to the number of critic updates instead. Use it
for all runs of a delay comparison (it also changes the `d=2` baseline).

## Running

    scripts/run_humanoid.sh <exp_name> [agent overrides...]

runs the four IsaacLab humanoid tasks for each seed; see the examples at the top of the script.

## Logs

Each run writes `logs/<group_name>/<exp_name>/<env_name>/seed<seed>_<time>/`:

- `config.yaml`: the resolved config of the run.
- `metrics.csv`: one row per logging interval (1% of training). Metrics are averaged over the interval unless noted.
  A cell is empty when the metric was not produced in that interval.

| Columns | Meaning |
|---|---|
| `env_step`, `wall_time` | Row index: environment steps, and seconds since the logger was created |
| `avg_return`, `avg_length`, `avg_success_*` | Evaluation results (10 evaluations per run, plus one at step 0) |
| `eval_env_step`, `eval_train_time`, `eval_num_updates` | Exactly when that evaluation ran. Use these for return curves |
| `time/train_time` | Seconds of training, excluding evaluation and video recording |
| `time/collect_time`, `time/update_time`, `time/eval_time` | Cumulative seconds in env stepping, gradient updates, evaluation |
| `time/env_steps_per_sec`, `time/gpu_max_memory_gb` | Throughput and peak GPU memory |
| `train/num_updates`, `train/num_actor_updates` | Cumulative critic and actor updates |
| `critic/td_abs` | Mean abs. TD error between distribution means (importance-weighted under LaBER). Drives the EMA delay |
| `critic/surprise` | Mean surprise, same weighting |
| `critic/loss`, `critic/q_mean`, `critic/learning_rate` | Critic loss, mean Q of the batch, learning rate |
| `actor/loss`, `actor/entropy`, `actor/learning_rate`, `temperature/value` | Actor and temperature state |
| `replay/age_mean`, `_median`, `_p90`, `_max` | Staleness of the critic's batch, in interaction steps |
| `replay/batch_td_abs`, `replay/batch_surprise` | Unweighted error of the critic's batch |
| `replay/age_surprise_corr` | Pearson correlation between age and surprise in the critic's batch |
| `replay/size` | Transitions in the buffer |
| `actor_delay/delay`, `td_ratio`, `td_ema` | Current delay, `e / ema`, and the EMA |
| `actor_delay/actor_updated` | Fraction of critic updates that also updated the actor |
| `laber/candidate_*` | Surprise, TD error, age and their correlation over the uniform candidates |
| `laber/effective_sample_frac`, `weight_max`, `unique_frac` | How concentrated the LaBER selection is |

## Videos

Checkpoints are saved at the end of training to `models/<group_name>/<exp_name>/<env_name>/seed<seed>-<time>/step<N>/`.
To record a trained policy to an mp4 without a display:

    uv run python play_isaaclab.py \
        --checkpoint_path 'models/humanoid/<exp_name>/Isaac-Velocity-Flat-G1-v0/seed0-<time>/step<N>' \
        --video --video_length 500 --follow_robot \
        --overrides env=isaaclab \
        --overrides env.env_name='Isaac-Velocity-Flat-G1-v0' \
        --overrides agent.buffer_max_length=1

The video (`rl-video-step-0.mp4`, 50 steps per second) goes to `<checkpoint_path>/videos` unless `--video_dir` is set.
`--follow_robot` keeps the camera on the robot of environment 0; without it the camera is fixed. Pass any overrides
that change the network (e.g. `agent.critic_hidden_dim`) exactly as in training. Without `--video` the script opens
the Isaac Sim viewport as before.

## Tests

    python -m pytest tests/test_extensions.py

CPU-only; covers the SWD sampling distribution, ages, the delay rules, LaBER weights, the LR schedule and the CSV log.
