# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlashSAC is a PyTorch implementation of a fast, stable SAC variant for high-dimensional robot control. Core innovations: distributional categorical critic (C51-style), unit weight normalization + RMSNorm (from LLM architectures), residual MLP blocks, zeta-distribution noise repetition for temporally correlated exploration, and `torch.compile` + AMP for GPU throughput. Supports 100+ tasks across 9 simulators (MuJoCo, DMC, IsaacLab, ManiSkill, Genesis, MuJoCo Playground, HumanoidBench, MyoSuite, MetaWorld).

## Development Commands

**Package manager:** `uv` — no `setup.py`, `Makefile`, or `requirements.txt`.

```bash
# Install base dependencies
uv sync

# Install with optional simulator environments
uv sync --extra <env>    # isaaclab | mujoco-playground | maniskill | genesis | humanoid-bench | myosuite | metaworld | all
# Note: 'all' excludes isaaclab (conflicts with genesis and humanoid-bench — use a separate venv)

# Install dev tools (black, ruff, mypy, ipdb)
uv sync --dev

# Lint + type-check (runs black → ruff check --fix → mypy)
./bin/lint

# Run training
uv run python train.py
uv run python train.py --overrides env=dmc --overrides env.env_name='humanoid-walk'

# Batch experiments
bash scripts/run_mujoco.sh

# Visualize IsaacLab agent
uv run python play_isaaclab.py --checkpoint_path 'models/.../step<N>' --overrides env=isaaclab ...
```

There is **no test suite** — CI only runs `./bin/lint` (static analysis).

Linting config (`pyproject.toml`): ruff line-length 120, mypy strict mode. Black 120-char lines.

## Configuration System

**Hydra** with OmegaConf. Base config: `configs/flashSAC_base.yaml`. Defaults pull in `agent: flashSAC` and `env: mujoco`. Override via repeatable `--overrides key=value` flags. A custom `eval` resolver is registered for expressions like `${eval:'int(${num_interaction_steps} / 10)'}`.

## Architecture

```
train.py
  ├── create_envs()  →  train_env / eval_env / record_env  (Gymnasium 1.1 VectorEnv)
  │     ├── CPU sims: SyncVectorEnv / AsyncVectorEnv (multiprocessing spawn)
  │     └── GPU sims: native parallel wrappers (IsaacLabVectorEnv, etc.)
  │
  ├── FlashSACAgent (flash_rl/agents/flashSAC/)
  │     ├── FlashSACActor          — Embedder → N×FlashSACBlock → UnitRMSNorm → NormalTanhPolicy
  │     ├── FlashSACDoubleCritic   — (N, B, D) ensemble layout; outputs categorical Q over num_bins bins
  │     ├── FlashSACDoubleCritic   — target critic, EMA-updated
  │     ├── FlashSACTemperature    — log-alpha parameter
  │     ├── TorchUniformBuffer     — pre-allocated tensors, n-step returns
  │     └── RewardNormalizer       — running return variance
  │
  └── Training loop:
        1. sample_actions()  →  zeta noise repeat → actor forward → tanh(mean + std*noise)
        2. env.step() → transitions
        3. buffer.add() + reward_normalizer.update()
        4. agent.update(): update_actor → update_temperature → update_critic (distributional Bellman) → EMA target
        5. Periodic: evaluate(), record_video(), log_metric(), agent.save()
```

Key source files:
- `train.py` — main loop; sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, JAX vars **before all imports** — keep it that way
- `flash_rl/agents/flashSAC/layer.py` — `UnitLinear`, `UnitRMSNorm`, `FlashSACBlock`, `NormalTanhPolicy`; ensemble variants use `(N, B, D)` einsum layout
- `flash_rl/agents/flashSAC/update.py` — actor + target critic forward passes are batched together (concatenated obs) to halve overhead; `_compute_categorical_td_target` and `_select_min_q_log_probs` are `@torch.compile`d
- `flash_rl/agents/utils/network.py` — `Network` wrapper for compile + EMA + weight norm; use `.apply(method, ...)` to call named methods on compiled modules
- `flash_rl/buffers/torch_buffer.py` — pre-allocated tensors, pin_memory on CPU, n-step via rolling deque
- `flash_rl/envs/isaaclab.py` — `IsaacLabVectorEnv`; `ACTION_BOUNDS` dict, asymmetric obs concatenation

## Important Constraints and Gotchas

**`compile_mode: 'auto'` (default):** On PyTorch ≥ 2.9 (Python 3.11), `reduce-overhead` causes 5–10× slowdowns; `auto` selects `max-autotune` instead. Don't override manually.

**AMP:** Disabled for CPU simulators (small batch, sync overhead dominates). Enabled for GPU simulators.

**Buffer device:** `cpu` for CPU simulators (1 env), `cuda` for GPU simulators (1024 envs).

**IsaacLab:** Only one `SimulationApp` per process — `eval_env = record_env = train_env`. Cannot be co-installed with genesis or humanoid-bench.

**Asymmetric observations (IsaacLab):** Actor obs = first `actor_observation_size[-1]` dims of the concatenated obs; critic sees the full obs. Populated via `env_info["actor_observation_size"]` in `IsaacLabVectorEnv.reset()`.

**`updates_per_interaction_step`:** Supports fractional values (e.g. 0.5 = one update every 2 steps) via a float accumulator.

**n-step returns:** At `n_step=3` (IsaacLab default), `gamma**3` is used in the TD target. Transitions are only finalized after `n_step` steps accumulate in the buffer's deque.
