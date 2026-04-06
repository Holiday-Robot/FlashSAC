from typing import Any

import gymnasium as gym

from ..types import NDArray
from .base_buffer import BaseBuffer, Batch  # noqa
from .numpy_buffer import NpyUniformBuffer

__all__ = ["BaseBuffer", "Batch", "NpyUniformBuffer", "create_buffer"]


def create_buffer(
    buffer_class_type: str,
    buffer_type: str,
    observation_space: gym.spaces.Space[NDArray],
    action_space: gym.spaces.Space[NDArray],
    n_step: int,
    gamma: float,
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    **kwargs: Any,
) -> BaseBuffer:
    if buffer_class_type == "numpy":
        if buffer_type == "uniform":
            return NpyUniformBuffer(
                observation_space=observation_space,
                action_space=action_space,
                n_step=n_step,
                gamma=gamma,
                max_length=max_length,
                min_length=min_length,
                sample_batch_size=sample_batch_size,
            )
        else:
            raise NotImplementedError
    elif buffer_class_type == "jax":
        raise NotImplementedError
    elif buffer_class_type == "torch":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid buffer class type: {buffer_class_type}")
