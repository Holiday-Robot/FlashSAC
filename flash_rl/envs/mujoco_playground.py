from typing import Any, Union

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
from mujoco_playground import registry
from mujoco_playground._src.mjx_env import MjxEnv, State
from mujoco_playground._src.wrapper import Wrapper

from flash_rl.types import NDArray

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")  # type: ignore
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)  # type: ignore
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # type: ignore

MUJOCO_PLAYGROUND_HUMANOID_ENVS = [
    "G1JoystickRoughTerrain",
    "G1JoystickFlatTerrain",
    "T1JoystickRoughTerrain",
    "T1JoystickFlatTerrain",
]


def recursive_to_numpy(
    data: Union[jnp.ndarray, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    if isinstance(data, jnp.ndarray):
        return np.array(data)
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)
    else:
        return data


class AutoResetWrapper(Wrapper):  # type: ignore
    """
    Modified version of mujoco_playground._src.wrapper.BraxAutoResetWrapper.
    Additionally saved `final_obs` for SAME_STEP auto reset (https://farama.org/Vector-Autoreset-Mode).
    """

    def reset(self, rng: jnp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["first_state"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: State, action: jnp.ndarray) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            done = state.done
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jnp.where(done, x, y)

        # Save final observation by simple copy
        state.info["final_obs"] = state.obs

        data = jax.tree.map(where_done, state.info["first_state"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(data=data, obs=obs)


class MujocoPlaygroundEnv(VectorEnv[NDArray, NDArray, NDArray]):
    """
    Gymnasium "SyncVectorEnv" implementation for MujocoPlayground environments.

    As all jax-based env does, MujocoPlayground does not internally store the 'state' of the env.

    Args:
        env: The environment created via mujoco_playground.registry.load(), and after wrappers are applied.
        num_envs (int): The number of parallel environments. This is only used if the env argument is a string
        seed (int): Seed for PRNGKey.
        to_numpy (bool): If True, will convert all outputs from jnp.ndarray to np.array.
        render_height (int): Render resolution. Currently works only when `num_envs=1`.
        render_width (int)
    """

    def __init__(
        self,
        env: MjxEnv,
        num_envs: int = 1,
        seed: int = 0,
        to_numpy: bool = False,
        render_height: int = 240,
        render_width: int = 320,
    ):
        self.envs = env
        self.num_envs = num_envs
        self.rng = jax.random.PRNGKey(seed)
        self.to_numpy = to_numpy
        self.render_height = render_height
        self.render_width = render_width

        self.reset_fn = jax.jit(self.envs.reset)
        self.step_fn = jax.jit(self.envs.step)
        self.states = None

        # Get observation/action spaces
        # - Obs range: Unknown (setting to [0, 0] since we only need the shape and dtype)
        # - Action range: [-1, 1] (https://github.com/google-deepmind/mujoco_playground/issues/19)
        if isinstance(self.envs.unwrapped.observation_size, dict):
            # Env will treat privileged state as the observation,
            # but will give 'actual' observation size in the info.
            self.asymmetric_obs = True
            self.obs_size = self.envs.observation_size["state"]
            priv_obs_size = self.envs.observation_size["privileged_state"]
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=priv_obs_size, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        else:
            self.asymmetric_obs = False
            self.obs_size = self.envs.observation_size
            self.single_observation_space = gym.spaces.Box(low=0.0, high=0.0, shape=self.obs_size, dtype=np.float32)
            self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        action_size = (self.envs.action_size,)
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=action_size, dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Metadata (hard-coded to SAME_STEP mode)
        self.metadata = {}
        self.metadata.update(autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)

    @property
    def unwrapped(self) -> MjxEnv:
        return self.envs.unwrapped

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray, dict[str, Any]]:
        self.rng, reset_rng = jax.random.split(self.rng, 2)
        reset_rngs = jax.random.split(reset_rng, self.num_envs)
        self.states = self.reset_fn(reset_rngs)
        obs, _, _, _, info = self.extract_infos(self.states)
        return obs, info

    def step(self, actions: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
        self.states = self.step_fn(self.states, actions)
        obs, rew, terminated, truncated, info = self.extract_infos(self.states)
        return obs, rew, terminated, truncated, info

    def extract_infos(self, state: State) -> tuple[NDArray, NDArray, NDArray, NDArray, dict[str, Any]]:
        """
        Helper function to extract infos from `state`.
        Note: `state.done` counts both terminated and truncated cases.
        """

        if self.asymmetric_obs:
            observation = state.obs["privileged_state"]
        else:
            observation = state.obs
        reward = state.reward
        truncation = state.info["truncation"]
        termination = jnp.logical_and(state.done, jnp.logical_not(truncation))

        info = {
            **state.info,
            **state.metrics,
            "actor_observation_size": self.obs_size,
        }
        if self.asymmetric_obs and "final_obs" in info:
            info["final_obs"] = info["final_obs"]["privileged_state"]

        if self.to_numpy:
            observation = np.array(observation)
            reward = np.array(reward)
            termination = np.array(termination)  # type: ignore
            truncation = np.array(truncation)
            info = recursive_to_numpy(info)  # type: ignore

        return observation, reward, termination, truncation, info  # type: ignore

    def close(self, **kwargs: Any) -> Any:
        return

    def render(self) -> Any:
        assert self.num_envs == 1
        is_joystick = self.envs.unwrapped.__class__.__name__ == "Joystick"
        img = self.envs.render(
            trajectory=self.states,
            height=self.render_height,
            width=self.render_width,
            camera="track" if is_joystick else None,
            scene_option=None,
            modify_scene_fns=None,
        )
        return [img]


def make_mujoco_playground_env(
    env_name: str,
    seed: int,
    num_envs: int,
    max_episode_steps: int,
    use_domain_randomization: bool,
    use_push_randomization: bool,
    height: int = 240,
    width: int = 320,
    **kwargs: Any,
) -> MujocoPlaygroundEnv:
    cfg = registry.get_default_config(env_name)
    is_humanoid_task = env_name in MUJOCO_PLAYGROUND_HUMANOID_ENVS

    # Randomizations
    if use_domain_randomization:
        raise NotImplementedError
    if is_humanoid_task and not use_push_randomization:
        cfg.push_config.enable = False
        cfg.push_config.magnitude_range = [0.0, 0.0]

    # Raw env
    env = registry.load(env_name, config=cfg)

    # Wrappers (order following `wrap_for_brax_training`: https://tinyurl.com/3az279ww)
    env = VmapWrapper(env)
    if max_episode_steps is not None:
        env = EpisodeWrapper(env, max_episode_steps, action_repeat=1)
    env = AutoResetWrapper(env)

    # MjxEnv-to-Gymnasium wrapper
    vec_env = MujocoPlaygroundEnv(
        env=env,
        num_envs=num_envs,
        seed=seed,
        to_numpy=True,
        render_height=height,
        render_width=width,
    )

    return vec_env
