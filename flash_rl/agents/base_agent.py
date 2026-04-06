from abc import ABC, abstractmethod
from typing import Any, Generic, MutableMapping, TypeVar

import gymnasium as gym

from flash_rl.types import NDArray, Tensor

Config = TypeVar("Config")


class BaseAgent(Generic[Config], ABC):
    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        env_info: dict[str, Any],
        cfg: Config,
    ):
        """
        A generic agent class.
        """
        self._observation_space = observation_space
        self._action_space = action_space
        self._cfg = cfg

    @abstractmethod
    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        pass

    @abstractmethod
    def process_transition(
        self,
        transition: MutableMapping[str, Tensor],
    ) -> None:
        """Handle interaction samples (e.g., add to replay buffer)"""
        pass

    @abstractmethod
    def can_start_training(self) -> bool:
        """Whether the agent is ready to update (e.g., enough samples in buffer)"""
        pass

    @abstractmethod
    def update(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def save_replay_buffer(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @abstractmethod
    def load_replay_buffer(self, path: str) -> None:
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        pass

    @property
    def observation_space(self) -> gym.spaces.Space[NDArray]:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space[NDArray]:
        return self._action_space

    @property
    def cfg(self) -> Config:
        return self._cfg
