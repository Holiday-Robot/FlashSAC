from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]

try:
    import jax.numpy as jnp

    Tensor = Union[NDArray, jnp.ndarray, torch.Tensor]
except ImportError:
    Tensor = Union[NDArray, torch.Tensor]
