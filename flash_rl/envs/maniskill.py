from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.common import torch_clone_dict
from mani_skill.utils.structs.types import Array

from ..types import NDArray


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


class ManiSkillVectorEnv(VectorEnv[Array, Array, Array]):
    """
    Gymnasium "SyncVectorEnv" implementation for ManiSkill environments running on the GPU for parallel simulation.

    Args:
        env: The environment created via gym.make / after wrappers are applied. If a string is given, we use
            gym.make(env) to create an environment
        num_envs: The number of parallel environments. This is only used if the env argument is a string
        env_kwargs: Environment kwargs to pass to gym.make. This is only used if the env argument is a string
        auto_reset (bool): Whether this wrapper will auto reset the environment (following the same API/conventions as
            Gymnasium). Default is True (recommended as most ML/RL libraries use auto reset)
        ignore_terminations (bool): Whether this wrapper ignores terminations when deciding when to auto reset.
            Terminations can be caused by the task reaching a success or fail state as defined in a task's evaluation
            function. Default is False, meaning there is early stop in episode rollouts. If set to True, this would
            generally for situations where you may want to model a task as infinite horizon where a task
            stops only due to the timelimit.
    """

    def __init__(
        self,
        env: Union[BaseEnv, str],
        num_envs: Optional[int] = None,
        auto_reset: bool = True,
        ignore_terminations: bool = False,
        to_numpy: bool = False,
        **kwargs: Any,
    ):
        if isinstance(env, str):
            assert num_envs, "num_envs must be provided."
            self.envs = gym.make(env, num_envs=num_envs, **kwargs)
        else:
            self.envs = env
            num_envs = self.base_env.num_envs
        self.num_envs = num_envs
        self.auto_reset = auto_reset
        self.ignore_terminations = ignore_terminations
        self.spec = self.envs.spec
        self.to_numpy = to_numpy

        self.single_observation_space = self.envs.get_wrapper_attr("single_observation_space")
        self.single_observation_space = self.envs.get_wrapper_attr("single_observation_space")
        self.single_action_space = self.envs.get_wrapper_attr("single_action_space")
        self.action_space = batch_space(self.single_action_space, self.num_envs)
        self.observation_space = self.envs.get_wrapper_attr("observation_space")
        self.metadata = self.envs.metadata
        # hardcoded SAME STEP reset mode for now. Trying to support others with backwards compatibility with gym < 1.0
        # might be too much of a hassle.
        self.metadata.update(autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    @property
    def device(self) -> torch.device:
        return self.base_env.device  # type: ignore

    @property
    def base_env(self) -> BaseEnv:
        return self.envs.unwrapped

    @property
    def unwrapped(self) -> VectorEnv[Array, Array, Array]:
        return self.base_env  # type: ignore

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[Array, dict[str, Any]]:
        if options is None:
            options = {}
        obs, info = self.envs.reset(seed=seed, options=options)  # type: ignore
        if "env_idx" in options:
            env_idx = options["env_idx"]
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.base_env.device)
            mask[env_idx] = True

        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            info = recursive_to_numpy(info)  # type: ignore

        return obs, info

    def step(self, actions: Union[Array, dict[str, Array]]) -> tuple[Array, Array, Array, Array, dict[str, Any]]:
        obs, rew, _terminations, _truncations, infos = self.envs.step(actions)

        if isinstance(_terminations, bool):
            terminations = torch.tensor([_terminations], device=self.device)
        else:
            assert isinstance(_terminations, torch.Tensor)
            terminations = _terminations
        if isinstance(_truncations, bool):
            truncations = torch.tensor([_truncations], device=self.device)
        else:
            assert isinstance(_truncations, torch.Tensor)
            truncations = _truncations
        if self.ignore_terminations:
            terminations.fill_(False)

        dones = torch.logical_or(terminations, truncations)

        if dones.any() and self.auto_reset:
            final_obs = torch_clone_dict(obs)
            env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
            final_info = torch_clone_dict(infos)
            obs, infos = self.reset(options=dict(env_idx=env_idx))
            # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
            infos["final_obs"] = final_obs
            infos["final_info"] = final_info
            # NOTE (stao): that adding masks like below is a bit redundant and not necessary
            # but this is to follow the standard gymnasium API
            infos["_final_info"] = dones
            infos["_final_obs"] = dones
            infos["_elapsed_steps"] = dones
            # NOTE (stao): Unlike gymnasium, the code here does not add masks for every key in the info object.

        if self.to_numpy:
            obs = recursive_to_numpy(obs)
            rew = recursive_to_numpy(rew)  # type: ignore
            terminations = recursive_to_numpy(terminations)  # type: ignore
            truncations = recursive_to_numpy(truncations)  # type: ignore
            infos = recursive_to_numpy(infos)  # type: ignore

        return obs, rew, terminations, truncations, infos

    def close(self, **kwargs: Any) -> None:
        self.envs.close(**kwargs)  # type: ignore

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        function = getattr(self.envs, name)
        return function(*args, **kwargs)

    def render(self) -> Optional[tuple[NDArray, ...]]:  # type: ignore
        image = self.base_env.render()
        if self.to_numpy:
            image = recursive_to_numpy(image)
        return image  # type: ignore


def make_maniskill_env(
    env_name: str,
    num_envs: int,
    width: int = 224,
    height: int = 224,
) -> gym.Env[Array, Array]:
    env = gym.make(
        env_name,
        num_envs=num_envs,
        render_mode="rgb_array",
        human_render_camera_configs={
            "width": width,
            "height": height,
        },
        reward_mode="normalized_dense",
    )
    env = ManiSkillVectorEnv(env, to_numpy=True, ignore_terminations=True)  # type: ignore
    return env
