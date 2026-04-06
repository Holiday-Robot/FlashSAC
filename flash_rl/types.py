from typing import Any, Union

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import torch

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, jnp.ndarray, torch.Tensor]
