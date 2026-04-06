from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym

Obs_T = TypeVar("Obs_T")
Action_T = TypeVar("Action_T")


class RepeatAction(gym.Wrapper[Obs_T, Action_T, Obs_T, Action_T]):
    def __init__(self, env: gym.Env[Obs_T, Action_T], action_repeat: int = 4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: Action_T) -> tuple[Obs_T, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, combined_info = self.env.step(action)
        total_reward = float(reward)
        for _ in range(self._action_repeat - 1):
            if terminated or truncated:
                break
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            combined_info.update(info)

        return obs, total_reward, terminated, truncated, combined_info
