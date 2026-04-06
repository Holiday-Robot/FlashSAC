#!/bin/bash
##################################################################################
# MyoSuite (CPU Simulator)
##################################################################################

env_names=(
    "myo-reach"
    "myo-reach-hard"
    "myo-pose"
    "myo-pose-hard"
    "myo-obj-hold"
    "myo-obj-hold-hard"
    "myo-key-turn"
    "myo-key-turn-hard"
    "myo-pen-twirl"
    "myo-pen-twirl-hard"
)
seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides seed=${seed} \
            `#=== Environment (CPU sim) ===#` \
            --overrides env=myosuite \
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
            --overrides gamma=0.95 \
            --overrides n_step=1
    done
done
