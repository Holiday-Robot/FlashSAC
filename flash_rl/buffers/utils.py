import numpy as np

from ..types import NDArray


def fast_uniform_sample(max_size: int, num_samples: int) -> NDArray:
    interval = max_size // num_samples
    base = np.arange(num_samples) * interval
    offsets = np.random.randint(0, interval, size=num_samples)
    return base + offsets


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree:
    def __init__(self, size: int):
        self._index = 0
        self._size = size
        self._full = False  # Used to track actual capacity
        self._tree_start_idx = 2 ** (size - 1).bit_length() - 1  # Put all used node leaves on last tree level
        self._sum_tree = np.zeros((self._tree_start_idx + self._size,), dtype=np.float32)
        self._max = 1.0  # Initial max value to return (1 = 1^ω), default priority is set to max

    # Updates nodes values from current tree
    def _update_nodes(self, indices: NDArray) -> None:
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self._sum_tree[indices] = np.sum(self._sum_tree[children_indices], axis=0)

    # Propagates changes up tree given tree indices
    def _propagate(self, indices: NDArray) -> None:
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagates single value up tree given a tree index for efficiency
    def _propagate_index(self, index: int) -> None:
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self._sum_tree[parent] = self._sum_tree[left] + self._sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    # Updates values given tree indices
    def update(self, indices: NDArray, values: NDArray) -> None:
        self._sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self._max = max(current_max_value, self._max)

    # Updates single value given a tree index for efficiency
    def _update_index(self, index: int, value: float) -> None:
        self._sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self._max = max(value, self._max)

    def add(self, value: float) -> None:
        self._update_index(self._index + self._tree_start_idx, value)  # Update tree
        self._index = (self._index + 1) % self._size  # Update index
        self._full = self._full or self._index == 0  # Save when capacity reached
        self._max = max(value, self._max)

    # Searches for the location of values in sum tree
    def _retrieve(self, indices: NDArray, values: NDArray) -> NDArray:
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)  # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self._sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self._tree_start_idx:
            children_indices = np.minimum(children_indices, self._sum_tree.shape[0] - 1)
        left_children_values = self._sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(
            np.int32
        )  # Classify which values are in left or right branches
        successor_indices = children_indices[
            successor_choices, np.arange(indices.size)
        ]  # Use classification to index into the indices matrix
        successor_values = (
            values - successor_choices * left_children_values
        )  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)

    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values: NDArray) -> tuple[NDArray, NDArray, NDArray]:
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self._tree_start_idx
        return (
            data_index,
            indices,
            self._sum_tree[indices],
        )  # Return values, data indices, tree indices

    @property
    def total(self) -> float:
        return float(self._sum_tree[0])

    @property
    def max(self) -> float:
        return self._max
