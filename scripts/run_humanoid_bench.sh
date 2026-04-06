#!/bin/bash
##################################################################################
# Humanoid Bench (CPU Simulator)
##################################################################################

env_names=(
    "h1-walk-v0"
    "h1-stand-v0"
    "h1-run-v0"
    "h1-reach-v0"
    "h1-hurdle-v0"
    "h1-crawl-v0"
    "h1-maze-v0"
    "h1-sit_simple-v0"
    "h1-sit_hard-v0"
    "h1-balance_simple-v0"
    "h1-balance_hard-v0"
    "h1-stair-v0"
    "h1-slide-v0"
    "h1-pole-v0"
)
seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides seed=${seed} \
            `#=== Environment (CPU sim) ===#` \
            --overrides env=humanoid_bench \
            --overrides env.env_name=${env_name} \
            --overrides num_env_steps=1_000_000 \
            --overrides num_train_envs=1 \
            --overrides num_eval_envs=1 \
            --overrides num_record_envs=1 \
            --overrides num_eval_episodes=50 \
            --overrides num_record_episodes=1 \
            `#=== Agent (CPU sim) ===#` \
            --overrides agent=flashSAC \
            --overrides agent.buffer_max_length=1_000_000 \
            --overrides agent.buffer_min_length=10_000 \
            --overrides agent.buffer_device_type='cpu' \
            --overrides agent.sample_batch_size=512 \
            --overrides agent.use_amp=false \
            --overrides updates_per_interaction_step=1 \
            `#=== Benchmark default ===#` \
            --overrides agent.asymmetric_observation=false \
            --overrides gamma=0.99 \
            --overrides n_step=1
    done
done
