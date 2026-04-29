from typing import Any, Union, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray

# NOTE: There is no way to get the action bounds from the env, so we hardcode them here following FastTD3
ACTION_BOUNDS = {
    "Isaac-Inhand-Rotate-Sharpa-Wave-v0": 1.0,
    "Isaac-Repose-Cube-Shadow-Direct-v0": 1.0,
    "Isaac-Repose-Cube-Allegro-Direct-v0": 1.0,
    "Isaac-Velocity-Flat-G1-v0": 1.0,
    "Isaac-Velocity-Rough-G1-v0": 1.0,
    "Isaac-Velocity-Flat-H1-v0": 1.0,
    "Isaac-Velocity-Rough-H1-v0": 1.0,
    "Isaac-Lift-Cube-Franka-v0": 3.0,
    "Isaac-Open-Drawer-Franka-v0": 3.0,
    "Isaac-Velocity-Flat-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-C-v0": 1.0,
    "Isaac-Velocity-Flat-Anymal-D-v0": 1.0,
    "Isaac-Velocity-Rough-Anymal-D-v0": 1.0,
}


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


class IsaacLabVectorEnv(
    VectorEnv[Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray]]
):
    """
    Gymnasium "SyncVectorEnv" implementation for IsaacLab environments.

    As all jax-based env does, IsaacLab does not internally store the 'state' of the env.

    Args:
        env_name (str): The environment name registered in IsaacLab.
        num_envs (int): The number of parallel environments. This is only used if the env argument is a string
        device (str):
        seed (int):
        action_bounds (float):
        to_numpy (bool): If True, will convert all outputs from jnp.ndarray to np.array.
    """

    def __init__(
        self,
        env_name: str,
        num_envs: int,
        seed: int,
        device: str,
        action_bounds: float,
        to_numpy: bool = True,
        headless: bool = True,
        use_priv_info: bool = False,
        env_cfg_overrides: dict[str, Any] | None = None,
    ):
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=headless, device=device, enable_cameras=not headless)
        self.simulation_app = app_launcher.app

        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # Register any extra task packages that may be installed alongside FlashSAC
        try:
            import rl_isaaclab.tasks.inhand_rotate  # noqa: F401
        except ImportError:
            pass

        env_cfg = parse_env_cfg(
            env_name,
            device=device,
            num_envs=num_envs,
        )
        env_cfg.seed = seed
        if env_cfg_overrides:
            for k, v in env_cfg_overrides.items():
                setattr(env_cfg, k, v)
        self.seed = seed
        self.device = device
        self.envs = gym.make(env_name, cfg=env_cfg, render_mode=None)

        self.num_envs = cast(Any, self.envs.unwrapped).num_envs
        self.max_episode_steps = cast(Any, self.envs.unwrapped).max_episode_length
        self.to_numpy = to_numpy

        # Get observation/action spaces
        # NOTE: Action range: [-1, 1] * action_bounds (https://github.com/google-deepmind/mujoco_playground/issues/19)
        self.obs_size = cast(Any, self.envs.unwrapped).single_observation_space["policy"].shape
        self.use_priv_info = use_priv_info
        unwrapped_cfg = cast(Any, self.envs.unwrapped).cfg
        self.priv_info_dim = int(getattr(unwrapped_cfg, "priv_info_dim", 0)) if use_priv_info else 0
        self.asymmetric_obs = "critic" in cast(Any, self.envs.unwrapped).single_observation_space
        if self.asymmetric_obs:
            # NOTE: Env will treat concatenate actor & critic states as the observation,
            # but will give 'actual' observation size in the info.
            self.critic_obs_size = cast(Any, self.envs.unwrapped).single_observation_space["critic"].shape
            # NOTE: setting to [0, 0] since we only need the shape and dtype
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=self.obs_size + self.critic_obs_size, dtype=np.float32
            )
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.critic_obs_size = 0
            obs_shape = (self.obs_size[0] + self.priv_info_dim,)
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=obs_shape, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_bounds = action_bounds
        self.action_size = cast(Any, self.envs.unwrapped).single_action_space.shape
        self.single_action_space = gym.spaces.Box(
            low=-1.0 * self.action_bounds, high=1.0 * self.action_bounds, shape=self.action_size, dtype=np.float32
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[Union[torch.Tensor, F32NDArray], dict[str, Any]]:
        obs_dict, infos = self.envs.reset()
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        if self.use_priv_info:
            obs = torch.cat((obs, obs_dict["priv_info"]), dim=-1)
        # NOTE: decorrelate episode horizons like RSL‑RL
        # In IsaacLab, `dones` is computed as follows:
        # `time_out = self.episode_length_buf >= self.max_episode_length - 1`
        # While training, this code spreads out the resets to avoid spikes
        # when many environments reset at a similar time.
        if random_start_init:
            # step in current episode (per env)
            cast(Any, self.envs.unwrapped).episode_length_buf = torch.randint_like(
                cast(Any, self.envs.unwrapped).episode_length_buf, high=int(self.max_episode_steps)
            )
        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = recursive_to_numpy(infos)  # type: ignore
        infos.update(
            {
                "actor_observation_size": self.obs_size,
                "asymmetric_obs": self.asymmetric_obs,
                "priv_info_dim": self.priv_info_dim,
            }
        )
        return obs, infos

    def step(self, actions: Union[torch.Tensor, F32NDArray]) -> tuple[
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        Union[torch.Tensor, F32NDArray],
        dict[str, Any],
    ]:
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(actions).to(self.device)

        if self.action_bounds is not None:
            torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminations, truncations, infos = cast(Any, self.envs.step(torch_actions))
        obs = obs_dict["policy"]
        if self.asymmetric_obs:
            critic_obs = obs_dict["critic"]
            obs = torch.cat((obs, critic_obs), dim=-1)
        else:
            critic_obs = None
        if self.use_priv_info:
            obs = torch.cat((obs, obs_dict["priv_info"]), dim=-1)
        episode_info = {
            k: float(v)
            for k, v in infos.items()
            if isinstance(v, (float, int)) or (isinstance(v, torch.Tensor) and v.numel() == 1)
        }
        infos = {"time_outs": truncations, "observations": {"critic": critic_obs}}
        if episode_info:
            infos["episode_info"] = episode_info
        # NOTE: There's really no way to get the raw observations from IsaacLab
        # We just use the 'reset_obs' as next_obs, unfortunately.
        # See https://github.com/isaac-sim/IsaacLab/issues/1362
        infos["final_obs"] = obs

        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            infos = recursive_to_numpy(infos)
        return obs, rew, terminations, truncations, infos

    def close(self, **kwargs: Any) -> None:
        # self.envs.close(**kwargs)
        # self.simulation_app.close()
        return

    def render(self) -> None:
        raise NotImplementedError("We don't support rendering for IsaacLab environments")


def make_isaaclab_env(
    env_name: str,
    num_envs: int,
    seed: int,
    headless: bool = True,
    use_priv_info: bool = False,
    env_cfg_overrides: dict[str, Any] | None = None,
) -> IsaacLabVectorEnv:
    if env_name not in ACTION_BOUNDS:
        print(f"Action bounds not defined for {env_name}; using default value 1.0.")
    action_bounds = ACTION_BOUNDS.get(env_name, 1.0)
    env = IsaacLabVectorEnv(
        env_name=env_name,
        num_envs=num_envs,
        seed=seed,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        action_bounds=action_bounds,
        to_numpy=True,
        headless=headless,
        use_priv_info=use_priv_info,
        env_cfg_overrides=env_cfg_overrides,
    )
    return env
