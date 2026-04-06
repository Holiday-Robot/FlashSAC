#!/bin/bash
##################################################################################
# ManiSkill (GPU Simulator)
##################################################################################

env_names=(
    "PickSingleYCB-v1"
    "PegInsertionSide-v1"
    "UnitreeG1TransportBox-v1"
    "UnitreeG1PlaceAppleInBowl-v1"
    "LiftPegUpright-v1"
    "PokeCube-v1"
    "PullCube-v1"
    "RollBall-v1"
)
seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides seed=${seed} \
            `#=== Environment (GPU sim) ===#` \
            --overrides env=maniskill \
            --overrides env.env_name=${env_name} \
            --overrides num_env_steps=50_000_896 \
            --overrides num_train_envs=1024 \
            --overrides num_eval_envs=50 \
            --overrides num_record_envs=1 \
            --overrides num_eval_episodes=50 \
            --overrides num_record_episodes=1 \
            `#=== Agent (GPU sim) ===#` \
            --overrides agent=flashSAC \
            --overrides agent.buffer_max_length=10_000_000 \
            --overrides agent.buffer_min_length=100_000 \
            --overrides agent.buffer_device_type='cuda' \
            --overrides agent.sample_batch_size=2048 \
            --overrides agent.use_amp=true \
            --overrides updates_per_interaction_step=2 \
            `#=== Benchmark default ===#` \
            --overrides agent.asymmetric_observation=false \
            --overrides gamma=0.9 \
            --overrides n_step=1
    done
done
