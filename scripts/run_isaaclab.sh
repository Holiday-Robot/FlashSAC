#!/bin/bash
##################################################################################
# IsaacLab (GPU Simulator)
##################################################################################

env_names=(
    "Isaac-Repose-Cube-Shadow-Direct-v0"
    "Isaac-Repose-Cube-Allegro-Direct-v0"
    "Isaac-Velocity-Flat-G1-v0"
    "Isaac-Velocity-Rough-G1-v0"
    "Isaac-Velocity-Flat-H1-v0"
    "Isaac-Velocity-Rough-H1-v0"
    "Isaac-Lift-Cube-Franka-v0"
    "Isaac-Open-Drawer-Franka-v0"
    "Isaac-Velocity-Flat-Anymal-C-v0"
    "Isaac-Velocity-Rough-Anymal-C-v0"
    "Isaac-Velocity-Flat-Anymal-D-v0"
    "Isaac-Velocity-Rough-Anymal-D-v0"
)
seeds=( 0 1000 2000 3000 4000 )

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides seed=${seed} \
            `#=== Environment (GPU sim) ===#` \
            --overrides env=isaaclab \
            --overrides env.env_name=${env_name} \
            --overrides num_env_steps=50_000_896 \
            --overrides num_train_envs=1024 \
            --overrides num_eval_envs=null \
            --overrides num_record_envs=null \
            --overrides num_eval_episodes=1024 \
            --overrides num_record_episodes=0 \
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
            --overrides gamma=0.99 \
            --overrides n_step=3
    done
done
