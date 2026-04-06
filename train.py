import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import argparse
import random
import sys
from datetime import datetime
from typing import Optional

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.common import create_logger
from flash_rl.envs import create_envs
from flash_rl.evaluation import evaluate, record_video
from flash_rl.types import Tensor


def run(args: argparse.Namespace) -> None:
    ###############################
    # configs
    ###############################

    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides

    # eval resolver
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))

    # initialize config
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    ###############################
    # seeding / configuration
    ###############################

    # Set random seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    #############################
    # envs
    #############################
    train_env, eval_env, record_env = create_envs(**cfg.env)

    observation_space = train_env.observation_space
    action_space = train_env.action_space

    #############################
    # agent
    #############################

    # Since the network architecture is typically tied to the learning algorithm,
    #   we opted not to fully modularize the network for the sake of readability.
    # Therefore, for each algorithm, the network is implemented within its respective directory.

    _, env_info = train_env.reset()
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    #############################
    # train
    #############################

    logger = create_logger(cfg)

    # load model if given
    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    save_path_resolved = cfg.save_path.replace("TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S"))
    save_path_base = script_dir + "/" + save_path_resolved
    if cfg.agent_load_path is not None:
        load_path = os.path.join(script_dir, cfg.agent_load_path)
        agent.load(load_path)
    if cfg.buffer_load_path is not None:
        load_path = os.path.join(script_dir, cfg.buffer_load_path)
        agent.load_replay_buffer(load_path)

    # initial evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
    logger.update_metric(**eval_info)
    logger.update_metric(**video_info)
    logger.log_metric(step=0)
    logger.reset()

    # start training
    observations, env_infos = train_env.reset()
    actions: Optional[Tensor] = None
    transition: Optional[dict[str, Tensor]] = None
    update_counter = 0
    update_info = {}

    for interaction_step in tqdm.tqdm(range(1, int(cfg.num_interaction_steps + 1)), smoothing=0.1, mininterval=0.5):
        # using env steps simplifies the comparison with the performance reported in the paper.
        env_step = interaction_step * cfg.num_train_envs

        # collect data. use random actions until agent.can_start_training()
        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            actions = train_env.action_space.sample()

        assert actions is not None
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        if "episode_info" in env_infos:
            logger.update_metric(**env_infos["episode_info"])

        transition = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        agent.process_transition(transition)
        transition["next_observation"] = next_observations
        observations = next_observations

        if agent.can_start_training():
            # update network
            # updates_per_interaction_step can be below 1.0
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                update_info = agent.update()
                logger.update_metric(**update_info)
                update_counter -= 1

            # evaluation
            if cfg.evaluation_per_interaction_step and interaction_step % cfg.evaluation_per_interaction_step == 0:
                eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
                logger.update_metric(**eval_info)

            # metrics
            if cfg.metrics_per_interaction_step and interaction_step % cfg.metrics_per_interaction_step == 0:
                metrics_info = agent.get_metrics()
                logger.update_metric(**metrics_info)

            # video recording
            if cfg.recording_per_interaction_step and interaction_step % cfg.recording_per_interaction_step == 0:
                video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
                logger.update_metric(**video_info)

            # logging
            if cfg.logging_per_interaction_step and interaction_step % cfg.logging_per_interaction_step == 0:
                logger.log_metric(step=env_step)
                logger.reset()

            # checkpointing
            if (
                cfg.save_checkpoint_per_interaction_step
                and interaction_step % cfg.save_checkpoint_per_interaction_step == 0
            ):
                save_path = os.path.join(save_path_base, f"step{interaction_step}")
                agent.save(save_path)

            # save buffer
            if cfg.save_buffer_per_interaction_step and interaction_step % cfg.save_buffer_per_interaction_step == 0:
                save_path = os.path.join(save_path_base, f"step{interaction_step}")
                agent.save_replay_buffer(save_path)

    # final evaluation
    eval_info = evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
    logger.update_metric(**eval_info)
    logger.update_metric(**video_info)
    logger.log_metric(step=env_step)
    logger.reset()

    train_env.close()
    eval_env.close()
    record_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    args = parser.parse_args()
    run(args)
