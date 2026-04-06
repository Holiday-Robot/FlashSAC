import os
import pickle
from collections import deque
from typing import Optional, cast

import gymnasium as gym
import numpy as np

from flash_rl.buffers.base_buffer import BaseBuffer, Batch
from flash_rl.types import NDArray, Tensor


class NpyUniformBuffer(BaseBuffer):
    """
    A uniform experience replay buffer using NumPy arrays.
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
    ):
        super(NpyUniformBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
        )
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def reset(self) -> None:
        m = self._max_length

        # for pixel-based environments, we would prefer uint8 dtype.
        observation_shape = (self._observation_space.shape[-1],) if self._observation_space.shape is not None else (0,)
        observation_dtype = self._observation_space.dtype if self._observation_space.dtype is not None else np.float32

        action_shape = (self._action_space.shape[-1],) if self._action_space.shape is not None else (0,)
        action_dtype = self._action_space.dtype if self._action_space.dtype is not None else np.float32

        # for float64, we enforce it to be float32
        if observation_dtype == np.float64:
            observation_dtype = np.float32

        if action_dtype == np.float64:
            action_dtype = np.float32

        self._observations = np.empty((m,) + observation_shape, dtype=observation_dtype)
        self._actions = np.empty((m,) + action_shape, dtype=action_dtype)
        self._rewards = np.empty((m,), dtype=np.float32)
        self._terminateds = np.empty((m,), dtype=np.float32)
        self._truncateds = np.empty((m,), dtype=np.float32)
        self._next_observations = np.empty((m,) + observation_shape, dtype=observation_dtype)

        self._n_step_transitions: deque[dict[str, NDArray]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0

    def _get_n_step_prev_transition(self) -> Batch:
        """
        This method processes a n_step_transitions to compute and update the
            n-step return, the done status, and the next observation.
        """
        # pop n-step previous transitions
        n_step_prev_transition = self._n_step_transitions[0]
        curr_transition = self._n_step_transitions[-1]

        # copy (np.array(,) generates copy version of array) last transition.
        n_step_reward = np.array(curr_transition["reward"])
        n_step_terminated = np.array(curr_transition["terminated"])
        n_step_truncated = np.array(curr_transition["truncated"])
        n_step_next_observation = np.array(curr_transition["next_observation"])

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            reward = transition["reward"]  # (n, )
            terminated = transition["terminated"]  # (n, )
            truncated = transition["truncated"]  # (n, )
            next_observation = transition["next_observation"]  # (n, *obs_shape)

            # compute n-step return
            done = (terminated.astype(bool) | truncated.astype(bool)).astype(np.float32)
            n_step_reward = reward + self._gamma * n_step_reward * (1 - done)

            # assign next observation starting from done
            done_mask = done.astype(bool)
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            n_step_next_observation[done_mask] = next_observation[done_mask]

        n_step_prev_transition["reward"] = n_step_reward
        n_step_prev_transition["terminated"] = n_step_terminated
        n_step_prev_transition["truncated"] = n_step_truncated
        n_step_prev_transition["next_observation"] = n_step_next_observation

        return cast(Batch, n_step_prev_transition)

    def add(self, transition: Batch) -> None:
        # temporarily hold current transition to the buffer
        self._n_step_transitions.append({key: np.array(value) for key, value in transition.items()})

        if len(self._n_step_transitions) >= self._n_step:
            n_step_prev_transition = self._get_n_step_prev_transition()

            add_batch_size = len(n_step_prev_transition["observation"])

            # add samples to the buffer
            add_idxs = np.arange(add_batch_size) + self._current_idx
            add_idxs = add_idxs % self._max_length

            self._observations[add_idxs] = n_step_prev_transition["observation"]
            self._actions[add_idxs] = n_step_prev_transition["action"]
            self._rewards[add_idxs] = n_step_prev_transition["reward"]
            self._terminateds[add_idxs] = n_step_prev_transition["terminated"]
            self._truncateds[add_idxs] = n_step_prev_transition["truncated"]
            self._next_observations[add_idxs] = n_step_prev_transition["next_observation"]

            self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
            self._current_idx = (self._current_idx + add_batch_size) % self._max_length

    def can_sample(self) -> bool:
        if self._num_in_buffer < self._min_length:
            return False
        else:
            return True

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        if sample_idxs is None:
            sample_idxs = np.random.randint(0, self._num_in_buffer, size=self._sample_batch_size)

        # Fancy indexing already returns a copy; np.array(...) is redundant.
        batch = {}
        batch["observation"] = self._observations[sample_idxs]
        batch["action"] = self._actions[sample_idxs]
        batch["reward"] = self._rewards[sample_idxs]
        batch["terminated"] = self._terminateds[sample_idxs]
        batch["truncated"] = self._truncateds[sample_idxs]
        batch["next_observation"] = self._next_observations[sample_idxs]

        return batch

    def save(self, path: str) -> None:
        dataset = {}
        dataset["observation"] = self._observations[: self._num_in_buffer]
        dataset["action"] = self._actions[: self._num_in_buffer]
        dataset["reward"] = self._rewards[: self._num_in_buffer]
        dataset["terminated"] = self._terminateds[: self._num_in_buffer]
        dataset["truncated"] = self._truncateds[: self._num_in_buffer]
        dataset["next_observation"] = self._next_observations[: self._num_in_buffer]
        with open(os.path.join(path, "dataset.pickle"), "wb") as f:
            pickle.dump(dataset, f)

    def get_observations(self) -> Tensor:
        return self._observations[: self._num_in_buffer]


"""
class NpyPrioritizedBuffer(NpyUniformBuffer):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        add_batch_size: int,
        sample_batch_size: int,
    ):
        super(NpyPrioritizedBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            add_batch_size,
            sample_batch_size,
        )

    def reset(self) -> None:
        super().reset()
        self._priority_tree = SegmentTree(self._max_length)

    def add(self, transition: Batch) -> None:
        super().add(transition)

        # add samples to the priority tree
        # SegmentTree class is not vectorized so just added instance one-by-one.
        if len(self._n_step_transitions) == self._n_step:
            for _ in range(self._add_batch_size):
                self._priority_tree.add(value=self._priority_tree.max)

    def _sample_idx_from_priority_tree(self):
        p_total = self._priority_tree.total  # sum of the priorities
        segment_length = p_total / self._sample_batch_size
        segment_starts = np.arange(self._sample_batch_size) * segment_length
        valid = False

        while not valid:
            # Uniformly sample from within all segments
            samples = (
                np.random.uniform(0.0, segment_length, [self._sample_batch_size])
                + segment_starts
            )
            # Retrieve samples from tree with un-normalised probability
            buffer_idxs, tree_idxs, sample_probs = self._priority_tree.find(samples)
            if np.all(sample_probs != 0):
                valid = True  # Note that conditions are valid but extra conservative around buffer index 0

        return buffer_idxs, tree_idxs, sample_probs

    def sample(self) -> Batch:
        sample_idxs, tree_idxs, sample_probs = self._sample_idx_from_priority_tree()

        batch = {}
        batch["observation"] = self._observations[sample_idxs]
        batch["action"] = self._actions[sample_idxs]
        batch["reward"] = self._rewards[sample_idxs]
        batch["terminated"] = self._terminateds[sample_idxs]
        batch["truncated"] = self._truncateds[sample_idxs]
        batch["next_observation"] = self._next_observations[sample_idxs]

        batch["tree_idxs"] = tree_idxs
        batch["sample_probs"] = sample_probs

        return batch

    def save(self, path: str) -> None:
        pass
"""
