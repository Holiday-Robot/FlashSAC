#!/bin/bash
##################################################################################
# Project experiments: IsaacLab humanoid locomotion (G1 / H1, flat / rough)
#
# usage: scripts/run_humanoid.sh <exp_name> [agent overrides...]
#
#   scripts/run_humanoid.sh uniform_d2
#   scripts/run_humanoid.sh swd_d2      agent.sampler=swd
#   scripts/run_humanoid.sh laber_d2    agent.sampler=laber
#   scripts/run_humanoid.sh uniform_d4  agent.actor_update_period=4 agent.actor_lr_schedule=global_step
#   scripts/run_humanoid.sh uniform_ema agent.actor_delay=ema agent.actor_delay_ema_interval=100 \
#                                       agent.actor_lr_schedule=global_step
#
# SEEDS and ENVS can be set in the environment, e.g. SEEDS="0 1000" ENVS="Isaac-Velocity-Flat-G1-v0".
# Metrics are written to logs/humanoid/<exp_name>/<env_name>/seed<seed>_<time>/metrics.csv
##################################################################################

exp_name=${1:?usage: scripts/run_humanoid.sh <exp_name> [agent overrides...]}
shift

extra_overrides=()
for override in "$@"; do
    extra_overrides+=(--overrides "$override")
done

read -r -a seeds <<< "${SEEDS:-0 1000 2000}"
read -r -a env_names <<< "${ENVS:-Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Rough-G1-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Velocity-Rough-H1-v0}"

for seed in "${seeds[@]}"; do
    for env_name in "${env_names[@]}"; do
        echo "$exp_name, $env_name, $seed"
        uv run python train.py \
            --config_name flashSAC_base \
            --overrides group_name=humanoid \
            --overrides exp_name=${exp_name} \
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
            --overrides n_step=3 \
            `#=== Experiment ===#` \
            "${extra_overrides[@]}"
    done
done
