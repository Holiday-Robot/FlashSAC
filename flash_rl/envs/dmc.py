import gymnasium as gym
from dm_control import suite
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium

from ..types import F32NDArray


def make_dmc_env(
    env_name: str,
    seed: int,
    flatten: bool = True,
) -> gym.Env[F32NDArray, F32NDArray]:
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode="rgb_array")
    if flatten and isinstance(env.observation_space, spaces.Dict):
        env = FlattenObservation(env)

    return env  # type: ignore
