from typing import Any, MutableMapping

import numpy as np
from gymnasium.vector import VectorEnv

from .agents.base_agent import BaseAgent
from .types import NDArray, Tensor


def evaluate(
    agent: BaseAgent[Any],
    env: VectorEnv[NDArray, NDArray, NDArray],
    num_episodes: int,
    env_type: str,
) -> dict[str, float]:
    num_envs = env.num_envs

    assert num_episodes % num_envs == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = num_episodes // num_envs

    total_return_list = []
    total_success_once_list = []
    total_success_end_list = []
    total_length_list = []

    for _ in range(num_eval_episodes_per_env):
        returns = np.zeros(num_envs)
        lengths = np.zeros(num_envs)
        success_once = np.zeros(num_envs)
        success_end = np.zeros(num_envs)
        if env_type == "isaaclab":
            observations, infos = env.reset(random_start_init=False)  # type: ignore[call-arg]
        else:
            observations, infos = env.reset()

        prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
        dones = np.zeros(num_envs)
        while np.sum(dones) < num_envs:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_transition=prev_transition,
                training=False,
            )
            actions = np.array(actions)
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

            prev_transition = {"next_observation": next_observations}

            returns += rewards * (1 - dones)
            lengths += 1 - dones

            if "success" in infos:
                success = infos["success"].astype("float") * (1 - dones)
                success_once = np.logical_or(success_once, success)

            if "final_info" in infos:
                for idx in range(num_envs):
                    final_info = infos["final_info"]
                    if "success" in final_info:
                        final_success = final_info["success"][idx].astype("float") * (1 - dones[idx])
                        success_end[idx] = final_success
            else:
                pass

            # once an episode is done in a sub-environment, we assume it to be done.
            # also, we assume to be done whether it is terminated or truncated during evaluation.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            observations = next_observations

        for env_idx in range(num_envs):
            total_return_list.append(returns[env_idx])
            total_length_list.append(lengths[env_idx])
            total_success_once_list.append(success_once[env_idx].astype("bool").astype("float"))
            total_success_end_list.append(success_end[env_idx].astype("bool").astype("float"))

    eval_info = {
        "episode_rewards": float(np.mean(total_return_list)),
        "avg_length": float(np.mean(total_length_list)),
        "avg_success_once": float(np.mean(total_success_once_list)),
        "avg_success_end": float(np.mean(total_success_end_list)),
    }

    env.reset()

    return eval_info


def record_video(
    agent: BaseAgent[Any],
    env: VectorEnv[NDArray, NDArray, NDArray],
    num_episodes: int,
    env_type: str,
    video_length: int = 1000,
) -> dict[str, Any]:
    if num_episodes == 0:
        return {}
    num_envs = env.num_envs
    # assert num_episodes % num_envs == 0, "num_episodes must be divisible by env.num_envs"
    num_eval_episodes_per_env = max(num_episodes // num_envs, 1)

    total_videos = []

    for _ in range(num_eval_episodes_per_env):
        videos: list[NDArray] = []

        if env_type == "isaaclab":
            observations, infos = env.reset(random_start_init=False)  # type: ignore[call-arg]
        else:
            observations, infos = env.reset()
        prev_transition: MutableMapping[str, Tensor] = {"next_observation": observations}
        images = env.render()  # type: ignore
        dones = np.zeros(num_envs)
        while np.sum(dones) < num_envs:
            actions = agent.sample_actions(
                interaction_step=0,
                prev_transition=prev_transition,
                training=False,
            )
            actions = np.array(actions)
            next_observations, rewards, terminateds, truncateds, infos = env.step(actions)

            prev_transition = {"next_observation": next_observations}

            # once an episode is done in a sub-environment, we assume it to be done.
            dones = np.maximum(dones, terminateds)
            dones = np.maximum(dones, truncateds)

            # proceed
            videos.append(images)  # type: ignore
            images = env.render()
            observations = next_observations

        total_videos.append(np.stack(videos, axis=1))  # (num_envs, t, c, h, w)

    # TODO: if there is termination, video length can be different
    # maybe add zero-padding depending on the max length
    total_videos = np.concatenate(total_videos, axis=0)  # (b, t, h, w, c)
    total_videos = total_videos[:, :video_length]
    total_videos = total_videos.transpose(0, 1, 4, 2, 3)  # (b, t, c, h, w)

    video_info = {"video": total_videos}

    env.reset()

    return video_info
