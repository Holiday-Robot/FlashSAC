import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import random
from typing import MutableMapping, Optional

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.isaaclab import make_isaaclab_env
from flash_rl.types import Tensor


def play(args: argparse.Namespace) -> None:
    config_path = args.config_path
    config_name = args.config_name
    overrides = args.overrides
    checkpoint_path = args.checkpoint_path
    num_envs = args.num_envs
    num_episodes = args.num_episodes

    # Load config (same as train.py)
    OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    hydra.initialize(version_base=None, config_path=config_path)
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Create environment with rendering (headless=False)
    env = make_isaaclab_env(
        env_name=cfg.env.env_name,
        num_envs=num_envs,
        seed=cfg.seed,
        headless=False,
    )

    # Create agent using config (same as train.py)
    _, env_info = env.reset(random_start_init=False)
    agent = create_agent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    # Load checkpoint
    agent.load(checkpoint_path)

    # Play loop
    observations, _ = env.reset(random_start_init=False)
    prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
    completed_episodes = 0
    episode_returns = np.zeros(num_envs)

    while completed_episodes < num_episodes:
        actions = agent.sample_actions(interaction_step=0, prev_transition=prev_transition, training=False)
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

        episode_returns += rewards
        episode_dones = np.logical_or(terminateds, truncateds)

        for idx in range(num_envs):
            if episode_dones[idx]:
                completed_episodes += 1
                print(f"Episode {completed_episodes}: return = {episode_returns[idx]:.2f}")
                episode_returns[idx] = 0.0
                if completed_episodes >= num_episodes:
                    break

        observations = next_observations
        prev_transition = {"next_observation": observations}

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent in IsaacLab")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments for visualization")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play")
    args = parser.parse_args()
    play(args)
