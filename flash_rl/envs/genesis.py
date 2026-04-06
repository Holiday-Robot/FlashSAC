from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import NDArray
from .genesis_envs import get_genesis_env


def recursive_to_numpy(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data


class GenesisVectorEnv(VectorEnv[NDArray, NDArray, NDArray]):
    """
    Gymnasium "SyncVectorEnv" implementation for Genesis example envs.
    """

    def __init__(
        self,
        env: Any,
        rescale_action: bool = True,
        to_numpy: bool = False,
        **kwargs: Any,
    ):
        self.envs = env
        self.num_envs = self.envs.num_envs
        self.rescale_action = rescale_action
        self.to_numpy = to_numpy

        # Observation/action spaces
        assert "num_obs" in self.envs.obs_cfg, "num_obs must be defined in obs_cfg."
        assert "num_priv_obs" in self.envs.obs_cfg, "num_priv_obs must be defined in obs_cfg (can be None)."
        if "num_history_obs" in self.envs.obs_cfg:
            assert (
                self.envs.obs_cfg["num_history_obs"] == 1
            ), "num_history_obs in obs_cfg is assumed unused in this version."
        if self.envs.obs_cfg["num_priv_obs"] is not None:
            self.asymmetric_obs = True
            self.obs_size = self.envs.obs_cfg["num_obs"]
            privileged_obs_size = self.envs.obs_cfg["num_priv_obs"]
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=(privileged_obs_size,), dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.asymmetric_obs = False
            self.obs_size = self.envs.obs_cfg["num_obs"]
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=(self.obs_size,), dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        assert "num_actions" in self.envs.env_cfg, "num_actions must be defined in env_cfg."
        assert "action_range" in self.envs.env_cfg, "action_range must be defined in env_cfg."
        action_size = self.envs.env_cfg["num_actions"]
        self.action_range = self.envs.env_cfg["action_range"]
        action_limit = 1.0 if self.rescale_action else self.action_range
        self.single_action_space = gym.spaces.Box(
            low=-action_limit, high=action_limit, shape=(action_size,), dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.metadata = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}

        # For convenience, we assume privileged obs to contain the exact same information
        # as regular obs, even the indices (i.e., priv_obs[:self.obs_size] == obs).
        # Below is a sanity check to make sure the environment follows this.
        if self.asymmetric_obs:
            self.envs.reset()
            if hasattr(self.envs, "compute_observations"):
                self.envs.compute_observations()
            obs, _ = self.envs.get_observations()
            priv_obs, _ = self.envs.get_privileged_observations()
            sanity_check_err = ((obs - priv_obs[:, : self.obs_size]) ** 2).sum().item()
            assert sanity_check_err < 1e-8, "Privileged observation sanity check failed in Genesis."

    @property
    def device(self) -> torch.device:
        return self.base_env.device  # type: ignore

    @property
    def base_env(self) -> Any:
        return self.envs

    @property
    def unwrapped(self) -> VectorEnv[NDArray, NDArray, NDArray]:
        return self.base_env  # type: ignore

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[NDArray, dict[str, Any]]:
        self.envs.reset()
        if self.asymmetric_obs:
            obs, _ = self.envs.get_privileged_observations()
        else:
            obs, _ = self.envs.get_observations()
        info = {
            "non_privileged_obs_size": self.obs_size,
            "actor_observation_size": [self.obs_size],
        }
        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            info = recursive_to_numpy(info)  # type: ignore

        return obs, info

    def step(
        self, actions: Union[NDArray, dict[str, NDArray]]
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
        actions = torch.tensor(actions, device=self.device)  # type: ignore
        if self.rescale_action:
            # rescale back to intended range
            actions *= self.action_range
        obs, rew, dones, infos = self.envs.step(actions)
        if self.asymmetric_obs:
            obs, _ = self.envs.get_privileged_observations()
        truncations = infos["time_outs"].to(dtype=torch.bool)
        terminations = torch.logical_and(dones, torch.logical_not(truncations))

        infos["final_obs"] = obs.detach().clone()
        if dones.any():
            infos["episode_info"] = {}
            episode_info = infos.pop("episode")
            for rew_name, rew_mean in episode_info.items():
                infos["episode_info"][f"Reward/{rew_name}"] = rew_mean
            infos["episode_info"]["episode_length"] = infos["episode_length"]
            if self.asymmetric_obs:
                infos["final_obs"] = infos["final_privileged_observations"].detach().clone()
            else:
                infos["final_obs"] = infos["final_observations"].detach().clone()
            # no final_info available in Genesis
            infos["final_info"] = {}
            # NOTE (stao): that adding masks like below is a bit redundant and not necessary
            # but this is to follow the standard gymnasium API
            infos["_final_info"] = dones
            infos["_final_obs"] = dones
            infos["_elapsed_steps"] = dones
            # NOTE (stao): Unlike gymnasium, the code here does not add masks for every key in the info object.

        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            rew = recursive_to_numpy(rew)
            terminations = recursive_to_numpy(terminations)  # type: ignore
            truncations = recursive_to_numpy(truncations)
            infos = recursive_to_numpy(infos)

        return obs, rew, terminations, truncations, infos  # type: ignore

    def close(self, **kwargs: Any) -> None:
        return

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        function = getattr(self.envs, name)
        return function(*args, **kwargs)

    def render(self) -> Optional[tuple[NDArray, ...]]:  # type: ignore
        # NOTE: Genesis seemingly renders only one environment, even when num_envs>1
        image = self.base_env.render()
        if self.to_numpy:
            image = recursive_to_numpy(image)
        image = image[np.newaxis, ...]
        return image  # type: ignore


def make_genesis_env(
    env_name: str,
    num_envs: int,
    rescale_action: bool,
    eval_mode: bool,
) -> VectorEnv[NDArray, NDArray, NDArray]:
    env = get_genesis_env(
        env_name=env_name,
        num_envs=num_envs,
        eval_mode=eval_mode,
    )
    env = GenesisVectorEnv(
        env,
        rescale_action=rescale_action,
        to_numpy=True,
    )
    return env
