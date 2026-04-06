import os
from collections import deque
from typing import Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch

from flash_rl.buffers.base_buffer import BaseBuffer, Batch
from flash_rl.types import NDArray

# Mapping from numpy dtypes to torch dtypes
_NP_TO_TORCH_DTYPE: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.float64): torch.float32,  # enforce float32
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
}


def _numpy_dtype_to_torch(dtype: Any) -> torch.dtype:
    """Convert a numpy dtype to a torch dtype, enforcing float32 for float64."""
    dtype = np.dtype(dtype)
    if dtype in _NP_TO_TORCH_DTYPE:
        return _NP_TO_TORCH_DTYPE[dtype]
    return torch.float32


class TorchUniformBuffer(BaseBuffer):
    """
    A uniform experience replay buffer using PyTorch tensors.
    Mirrors NpyUniformBuffer behavior exactly; data is stored on the given device.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device_type: str,
    ):
        super(TorchUniformBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
        )
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def reset(self) -> None:
        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()

        observation_shape = (self._observation_space.shape[-1],) if self._observation_space.shape is not None else (0,)
        observation_dtype = _numpy_dtype_to_torch(
            self._observation_space.dtype if self._observation_space.dtype is not None else np.float32
        )

        action_shape = (self._action_space.shape[-1],) if self._action_space.shape is not None else (0,)
        action_dtype = _numpy_dtype_to_torch(
            self._action_space.dtype if self._action_space.dtype is not None else np.float32
        )

        self._observations = torch.empty(
            (m,) + observation_shape, dtype=observation_dtype, device=self._device, pin_memory=pin
        )
        self._next_observations = torch.empty(
            (m,) + observation_shape, dtype=observation_dtype, device=self._device, pin_memory=pin
        )
        self._actions = torch.empty((m,) + action_shape, dtype=action_dtype, device=self._device, pin_memory=pin)
        self._rewards = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._terminateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._truncateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0

    def _to_tensor(self, value: Any) -> torch.Tensor:
        """Convert a value to a tensor on the buffer device (cloned if already a tensor)."""
        if isinstance(value, torch.Tensor):
            return value.detach().to(self._device, copy=True)
        return torch.tensor(value, device=self._device)

    def _get_n_step_prev_transition(self) -> Batch:
        """
        Processes n_step_transitions to compute the n-step return, done status,
        and next observation. Mirrors NpyUniformBuffer._get_n_step_prev_transition exactly.
        """
        n_step_prev_transition = self._n_step_transitions[0]
        curr_transition = self._n_step_transitions[-1]

        # clone last transition
        n_step_reward = curr_transition["reward"].clone()
        n_step_terminated = curr_transition["terminated"].clone()
        n_step_truncated = curr_transition["truncated"].clone()
        n_step_next_observation = curr_transition["next_observation"].clone()

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            reward = transition["reward"]  # (n,)
            terminated = transition["terminated"]  # (n,)
            truncated = transition["truncated"]  # (n,)
            next_observation = transition["next_observation"]  # (n, *obs_shape)

            # compute n-step return
            done = (terminated.bool() | truncated.bool()).float()
            n_step_reward = reward + self._gamma * n_step_reward * (1 - done)

            # assign next observation starting from done
            done_mask = done.bool()
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            n_step_next_observation[done_mask] = next_observation[done_mask]

        n_step_prev_transition["reward"] = n_step_reward
        n_step_prev_transition["terminated"] = n_step_terminated
        n_step_prev_transition["truncated"] = n_step_truncated
        n_step_prev_transition["next_observation"] = n_step_next_observation

        return cast(Batch, n_step_prev_transition)

    def add(self, transition: Batch) -> None:
        self._n_step_transitions.append({key: self._to_tensor(value) for key, value in transition.items()})

        if len(self._n_step_transitions) >= self._n_step:
            n_step_prev_transition = cast(dict[str, torch.Tensor], self._get_n_step_prev_transition())

            add_batch_size = len(n_step_prev_transition["observation"])
            end_idx = self._current_idx + add_batch_size

            if end_idx <= self._max_length:
                # Contiguous slice — avoids scatter and tensor allocation
                idxs: Any = slice(self._current_idx, end_idx)
            else:
                idxs = (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length

            self._observations[idxs] = n_step_prev_transition["observation"].to(self._observations.dtype)
            self._next_observations[idxs] = n_step_prev_transition["next_observation"].to(self._next_observations.dtype)
            self._actions[idxs] = n_step_prev_transition["action"].to(self._actions.dtype)
            self._rewards[idxs] = n_step_prev_transition["reward"].to(self._rewards.dtype)
            self._terminateds[idxs] = n_step_prev_transition["terminated"].to(self._terminateds.dtype)
            self._truncateds[idxs] = n_step_prev_transition["truncated"].to(self._truncateds.dtype)

            self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
            self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def can_sample(self) -> bool:
        return self._num_in_buffer >= self._min_length

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        if sample_idxs is None:
            idxs = torch.randint(0, self._num_in_buffer, (self._sample_batch_size,), device=self._device)
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device)

        batch: Batch = {}
        batch["observation"] = self._observations[idxs]
        batch["action"] = self._actions[idxs]
        batch["reward"] = self._rewards[idxs]
        batch["terminated"] = self._terminateds[idxs]
        batch["truncated"] = self._truncateds[idxs]
        batch["next_observation"] = self._next_observations[idxs]

        return batch

    def save(self, path: str) -> None:
        """
        Save buffer contents and metadata.
        args:
            path (str): The full file path (e.g. "checkpoints/replay_buffer.pt").
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n = self._num_in_buffer
        dataset: dict[str, Any] = {
            "observation": self._observations[:n],
            "action": self._actions[:n],
            "reward": self._rewards[:n],
            "terminated": self._terminateds[:n],
            "truncated": self._truncateds[:n],
            "next_observation": self._next_observations[:n],
            "num_in_buffer": self._num_in_buffer,
            "current_idx": self._current_idx,
        }
        torch.save(dataset, path)

    def load(self, path: str) -> None:
        """
        Load buffer contents and metadata.
        args:
            path (str): The full file path (e.g. "checkpoints/replay_buffer.pt").
        """
        dataset = torch.load(path, map_location=self._device)
        n = dataset["num_in_buffer"]

        self._observations[:n] = dataset["observation"]
        self._next_observations[:n] = dataset["next_observation"]
        self._actions[:n] = dataset["action"]
        self._rewards[:n] = dataset["reward"]
        self._terminateds[:n] = dataset["terminated"]
        self._truncateds[:n] = dataset["truncated"]

        self._num_in_buffer = n
        self._current_idx = dataset["current_idx"]
        # Note: _n_step_transitions is intentionally not saved/loaded.
        # At most (n_step - 1) in-flight transitions are lost, which is negligible.
        self._n_step_transitions.clear()

    def get_observations(self) -> torch.Tensor:
        return self._observations[: self._num_in_buffer]
