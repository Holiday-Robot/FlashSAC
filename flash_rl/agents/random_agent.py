from typing import Any, Mapping, TypeVar

import gymnasium as gym
import numpy as np

from flash_rl.agents.base_agent import BaseAgent
from flash_rl.types import NDArray, Tensor

Config = TypeVar("Config")


class RandomAgent(BaseAgent[Config]):
    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: Config,
    ):
        """
        An agent that randomly selects actions without training.
        Useful for collecting baseline results and for debugging purposes.
        """
        super(RandomAgent, self).__init__(
            observation_space,
            action_space,
            env_info,
            cfg,
        )

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: Mapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        num_envs = prev_transition["next_observation"].shape[0]
        actions = []
        for _ in range(num_envs):
            actions.append(self._action_space.sample())

        return np.stack(actions).reshape(-1)

    def process_transition(
        self,
        transition: Mapping[str, Tensor],
    ) -> None:
        pass

    def can_start_training(self) -> bool:
        return True

    def update(self) -> dict[str, Any]:
        return {}

    def get_metrics(self) -> dict[str, Any]:
        return {}

    def save(self, path: str) -> None:
        pass

    def save_replay_buffer(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass

    def load_replay_buffer(self, path: str) -> None:
        pass
