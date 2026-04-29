#!/bin/bash
##################################################################################
# FlashSAC RMA Stage 1 — Sharpa Wave In-Hand Rotate (IsaacLab)
# Actor encodes priv_info via env_mlp; critic sees raw priv_info (asymmetric).
##################################################################################

uv run --frozen python train.py \
    --config_name flashSAC_base \
    --overrides seed=0 \
    `#=== Environment ===#` \
    --overrides env=sharpa \
    --overrides num_env_steps=50_000_896 \
    --overrides num_train_envs=1024 \
    --overrides num_eval_envs=null \
    --overrides num_record_envs=null \
    --overrides num_eval_episodes=1024 \
    --overrides num_record_episodes=0 \
    `#=== Agent ===#` \
    --overrides agent=flashSAC_rma \
    --overrides agent.buffer_max_length=10_000_000 \
    --overrides agent.buffer_min_length=100_000 \
    --overrides agent.buffer_device_type='cuda' \
    --overrides agent.use_amp=true \
    --overrides updates_per_interaction_step=2 \
    --overrides agent.asymmetric_observation=false \
    --overrides gamma=0.99 \
    --overrides n_step=3
