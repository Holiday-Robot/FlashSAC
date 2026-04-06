import os
from typing import Any, Optional, Union

import gymnasium as gym
from gymnasium.utils.step_api_compatibility import convert_to_terminated_truncated_step_api
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

from ..types import F32NDArray, NDArray


class MetaWorldtoGymnasium(gym.Env[F32NDArray, F32NDArray]):
    """
    Convert MetaWorld `SawyerXYZEnv` env type to `gymnasium.Env`
    """

    def __init__(
        self,
        env: type[SawyerXYZEnv],
        seed: int,
        device_id: Union[int, str],
        sparse: bool,
        width: int,
        height: int,
    ):
        self.env = env
        self.sparse = sparse
        self.metadata = getattr(self.env, "metadata", {"render_modes": []})
        self.reward_range = getattr(self.env, "reward_range", None)
        self.spec = getattr(self.env, "spec", None)

        # rendering information
        self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.height = height
        self.width = width
        self.camera_name = "corner2"
        self.render_mode = "rgb_array"
        self.env._freeze_rand_vec = False
        self.device_id = device_id  # for GPU rendering

        # set random Seed
        self.seed = seed
        self.env.seed(seed)

        # space definition
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low,
            high=self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype,
        )
        self.action_space = gym.spaces.Box(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
            shape=self.env.action_space.shape,
            dtype=self.env.action_space.dtype,
        )

    @property
    def device(self) -> Union[int, str]:
        return self.device_id

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[F32NDArray, dict[str, Any]]:
        reset_info: dict[str, Any] = {}
        if seed is not None:
            self.env.seed(seed)
        state = self.env.reset()
        return state, reset_info

    def step(self, action: F32NDArray) -> tuple[F32NDArray, float, bool, bool, dict[str, Any]]:
        state, reward, done, info = self.env.step(action.copy())
        if self.sparse:
            assert "success" in info
            reward = float(info["success"])

        return convert_to_terminated_truncated_step_api((state, reward, done, info))  # type: ignore

    def render(self, *args: Any, **kwargs: Any) -> Union[Any, tuple[NDArray, ...]]:  # type: ignore
        return self.env.sim.render(
            width=self.width,
            height=self.height,
            mode="offscreen",
            camera_name=self.camera_name,
            device_id=self.device_id,
        ).copy()

    def close(self) -> None:
        self.env.close()


def make_metaworld_env(
    env_name: str,
    seed: int,
    width: int = 224,
    height: int = 224,
) -> gym.Env[F32NDArray, F32NDArray]:
    if "_sparse" in env_name:
        env_name = env_name.split("_")[0]
        sparse = True
    else:
        sparse = False
    env_id = env_name.split("-", 1)[-1] + "-v2-goal-observable"
    # `SawyerXYZEnv` type
    env_cls: type[SawyerXYZEnv] = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed)
    env = MetaWorldtoGymnasium(
        env_cls,
        seed=seed,
        device_id=int(os.environ["MUJOCO_EGL_DEVICE_ID"]),
        sparse=sparse,
        width=width,
        height=height,
    )
    # Convert `max_path_length` of `SawyerXYZEnv` to `max_epsiode_steps` of `gym.Env`
    if hasattr(env_cls, "max_path_length") and env_cls.max_path_length > 0:
        env = gym.wrappers.TimeLimit(env, env_cls.max_path_length)  # type: ignore
    return env
