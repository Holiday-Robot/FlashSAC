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

    # Create environment: Isaac Sim viewport, or headless recording to an mp4 file with --video
    video_dir = None
    if args.video:
        video_dir = args.video_dir or os.path.join(checkpoint_path, "videos")
    env = make_isaaclab_env(
        env_name=cfg.env.env_name,
        num_envs=num_envs,
        seed=cfg.seed,
        headless=args.video,
        video_dir=video_dir,
        video_length=args.video_length,
        follow_robot=args.follow_robot,
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
    num_steps = 0

    # with --video, play for the length of the video instead of a number of episodes
    while num_steps < args.video_length if args.video else completed_episodes < num_episodes:
        num_steps += 1
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
                if not args.video and completed_episodes >= num_episodes:
                    break

        observations = next_observations
        prev_transition = {"next_observation": observations}

    if args.video:
        env.finish_video()
        print(f"Saved video to {video_dir}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent in IsaacLab")
    parser.add_argument("--config_path", type=str, default="./configs")
    parser.add_argument("--config_name", type=str, default="flashSAC_base")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to agent checkpoint directory")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments for visualization")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to play")
    parser.add_argument("--video", action="store_true", help="Record an mp4 headless instead of opening the viewport")
    parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record (50 steps = 1 s)")
    parser.add_argument("--video_dir", type=str, default=None, help="Default: <checkpoint_path>/videos")
    parser.add_argument("--follow_robot", action="store_true", help="Camera follows the robot of environment 0")
    args = parser.parse_args()
    play(args)
