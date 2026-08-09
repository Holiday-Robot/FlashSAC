"""Optional Triton accelerators for FlashSAC."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Optional

import torch

_NUM_BINS = 101
_MEASURED_TORCH_VERSION = (2, 9)
_MEASURED_TRITON_VERSION = (3, 5)
_kernel: Optional[Callable[..., torch.Tensor]] = None
_disabled = False


def _major_minor(version: str) -> Optional[tuple[int, int]]:
    try:
        major, minor = version.split(".")[:2]
        return int(major), int(minor)
    except (ValueError, IndexError):
        return None


@cache
def _has_measured_dependency_versions() -> bool:
    """Use only the Torch/Triton line where the kernel measured faster."""
    try:
        triton_version = package_version("triton")
    except PackageNotFoundError:
        return False
    return (
        _major_minor(str(torch.__version__)) == _MEASURED_TORCH_VERSION
        and _major_minor(triton_version) == _MEASURED_TRITON_VERSION
    )


def _can_use_kernel(
    next_qs: torch.Tensor,
    next_q_log_probs: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    actor_entropy: torch.Tensor,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> bool:
    if _disabled or not _has_measured_dependency_versions():
        return False
    if num_bins != _NUM_BINS or max_v <= min_v:
        return False
    if next_qs.ndim != 2 or next_qs.shape[0] != 2:
        return False
    if next_q_log_probs.ndim != 3 or next_q_log_probs.shape != (*next_qs.shape, num_bins):
        return False

    batch_size = next_qs.shape[1]
    if batch_size == 0 or any(tensor.shape != (batch_size,) for tensor in (reward, done, actor_entropy)):
        return False

    tensors = (next_qs, next_q_log_probs, reward, done, actor_entropy)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        return False
    if not all(
        tensor.is_cuda and tensor.dtype == torch.float32 and tensor.device == next_qs.device for tensor in tensors
    ):
        return False
    return bool(torch.cuda.get_device_capability(next_qs.device) >= (8, 0))


def try_fused_categorical_td_target(
    next_qs: torch.Tensor,
    next_q_log_probs: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    actor_entropy: torch.Tensor,
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
) -> Optional[torch.Tensor]:
    """Return a fused target, or ``None`` so the caller uses PyTorch."""
    global _disabled, _kernel

    try:
        if not _can_use_kernel(
            next_qs,
            next_q_log_probs,
            reward,
            done,
            actor_entropy,
            num_bins,
            min_v,
            max_v,
        ):
            return None

        if _kernel is None:
            from .categorical_target import fused_categorical_td_target

            _kernel = fused_categorical_td_target
        return _kernel(
            next_qs,
            next_q_log_probs,
            reward,
            done,
            actor_entropy,
            gamma,
            min_v,
            max_v,
        )
    except Exception as error:
        _disabled = True
        try:
            warnings.warn(
                "Disabling the optional Triton categorical target; "
                f"using the original PyTorch path instead ({type(error).__name__}: {error}).",
                RuntimeWarning,
                stacklevel=2,
            )
        except RuntimeWarning:
            pass
        return None
