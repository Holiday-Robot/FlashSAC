import re
from typing import Any, Callable, Union

import torch


def tree_map(func: Any, tree: Any) -> Any:
    """Equivalent of jax.tree_util.tree_map."""
    if isinstance(tree, dict):
        return {k: tree_map(func, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(func, item) for item in tree)
    else:
        return func(tree)


def tree_leaves(tree: Any) -> list[Any]:
    """Equivalent of jax.tree_util.tree_leaves."""
    leaves: list[Any] = []
    if isinstance(tree, dict):
        for v in tree.values():
            leaves.extend(tree_leaves(v))
    elif isinstance(tree, (list, tuple)):
        for item in tree:
            leaves.extend(tree_leaves(item))
    else:
        leaves.append(tree)
    return leaves


def tree_filter(
    f: Callable[..., Any], tree: Union[torch.Tensor, dict[str, Any]], target_re: str = "scaler"
) -> Union[torch.Tensor, dict[str, Any]]:
    if isinstance(tree, dict):
        # Keep only "target_re" keys in the dictionary
        filtered_tree = {}
        for k, v in tree.items():
            if re.fullmatch(target_re, k):
                filtered_tree[k] = tree_filter(f, v, target_re="scaler")
            elif isinstance(v, dict):  # Recursively check nested dictionaries
                filtered_value = tree_filter(f, v, target_re="scaler")
                if filtered_value:  # Only keep non-empty dictionaries
                    filtered_tree[k] = filtered_value
        return filtered_tree
    else:
        # If not a dictionary, return the tree as is (typically a leaf node)
        return f(tree)  # type: ignore
