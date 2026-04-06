from abc import ABC, abstractmethod
from typing import MutableMapping, Optional

import gymnasium as gym

from flash_rl.types import NDArray, Tensor

Batch = MutableMapping[str, Tensor]


class BaseBuffer(ABC):
    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
    ):
        """
        A generic buffer class.

        args:
            observation_shape
            action_shapce
            max_length: maximum length of buffer (max number of experiences stored within the state).
            min_length: minimum number of experiences saved in the buffer state before we can sample.
            add_sequences: indiciator of whether we will be adding data in sequences to the buffer?
            sample_batch_size: batch size of data that is sampled from a single sampling call.
        """

        self._observation_space = observation_space
        self._action_space = action_space
        self._max_length = max_length
        self._min_length = min_length
        self._n_step = n_step
        self._gamma = gamma
        self._sample_batch_size = sample_batch_size

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(self, transition: MutableMapping[str, Tensor]) -> None:
        pass

    @abstractmethod
    def can_sample(self) -> bool:
        pass

    @abstractmethod
    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def get_observations(self) -> Tensor:
        pass
