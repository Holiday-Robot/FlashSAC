from typing import Any, SupportsFloat, Union

import gymnasium as gym
from gymnasium.core import RenderFrame

from ..types import F32NDArray

MYOSUITE_TASKS_DICT = {
    "myo-reach": "myoHandReachFixed-v0",
    "myo-reach-hard": "myoHandReachRandom-v0",
    "myo-pose": "myoHandPoseFixed-v0",
    "myo-pose-hard": "myoHandPoseRandom-v0",
    "myo-obj-hold": "myoHandObjHoldFixed-v0",
    "myo-obj-hold-hard": "myoHandObjHoldRandom-v0",
    "myo-key-turn": "myoHandKeyTurnFixed-v0",
    "myo-key-turn-hard": "myoHandKeyTurnRandom-v0",
    "myo-pen-twirl": "myoHandPenTwirlFixed-v0",
    "myo-pen-twirl-hard": "myoHandPenTwirlRandom-v0",
}


class MyosuiteGymnasiumVersionWrapper(gym.Wrapper[F32NDArray, F32NDArray, F32NDArray, F32NDArray]):
    """
    myosuite originally requires gymnasium==0.15
    however, we are currently using  gymnasium==1.0.0a2,
    hence requiring some minor fix to the
      - fix a.
      - fix b.
    """

    def __init__(self, env: gym.Env[F32NDArray, F32NDArray]):
        super().__init__(env)
        self.unwrapped_env = env.unwrapped

    def step(self, action: F32NDArray) -> tuple[F32NDArray, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["success"] = info["solved"]
        return obs, reward, terminated, truncated, info

    def render(
        self, width: int = 192, height: int = 192, camera_id: str = "hand_side_inter"
    ) -> Union[RenderFrame, list[RenderFrame], None]:
        return self.unwrapped_env.sim.renderer.render_offscreen(  # type: ignore
            width=width,
            height=height,
            camera_id=camera_id,
        )


def make_myosuite_env(
    env_name: str,
    seed: int,
    **kwargs: Any,
) -> gym.Env[F32NDArray, F32NDArray]:
    from myosuite.utils import gym as myo_gym

    env = myo_gym.make(MYOSUITE_TASKS_DICT[env_name])
    env = MyosuiteGymnasiumVersionWrapper(env)

    return env
