"""Triton fast path for FlashSAC's categorical TD target."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_FUSED_CATEGORICAL_TARGET_BLOCK_SIZE = 128


@triton.jit  # type: ignore[misc]
def _fused_categorical_td_target_kernel(  # type: ignore[no-untyped-def]
    next_qs_ptr,
    next_q_log_probs_ptr,
    reward_ptr,
    done_ptr,
    actor_entropy_ptr,
    target_probs_ptr,
    num_bins,
    gamma,
    min_v,
    max_v,
    next_qs_stride_critic,
    next_qs_stride_batch,
    log_probs_stride_critic,
    log_probs_stride_batch,
    log_probs_stride_bin,
    reward_stride,
    done_stride,
    actor_entropy_stride,
    target_probs_stride_batch,
    target_probs_stride_bin,
    BLOCK: tl.constexpr,
) -> None:
    """Select the minimum-Q critic and project one batch row without atomics."""
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    valid_source = offsets < num_bins

    q0 = tl.load(next_qs_ptr + batch_idx * next_qs_stride_batch)
    q1 = tl.load(next_qs_ptr + next_qs_stride_critic + batch_idx * next_qs_stride_batch)

    # Match torch.argmin: select the first value on ties and the first NaN.
    selected_critic = tl.where((q0 <= q1) | (q0 != q0), 0, 1)
    selected_log_probs_ptr = (
        next_q_log_probs_ptr
        + selected_critic * log_probs_stride_critic
        + batch_idx * log_probs_stride_batch
        + offsets * log_probs_stride_bin
    )
    log_probs = tl.load(selected_log_probs_ptr, mask=valid_source, other=-float("inf")).to(tl.float32)

    reward = tl.load(reward_ptr + batch_idx * reward_stride).to(tl.float32)
    done = tl.load(done_ptr + batch_idx * done_stride).to(tl.float32)
    actor_entropy = tl.load(actor_entropy_ptr + batch_idx * actor_entropy_stride).to(tl.float32)

    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = min_v + offsets.to(tl.float32) * bin_width
    target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = tl.minimum(
        tl.maximum(target_bin_values, min_v, propagate_nan=tl.PropagateNan.ALL),
        max_v,
        propagate_nan=tl.PropagateNan.ALL,
    )

    projected = (target_bin_values - min_v) / bin_width
    projected = tl.minimum(
        tl.maximum(projected, 0.0, propagate_nan=tl.PropagateNan.ALL),
        num_bins - 1.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    projection_is_nan = projected != projected
    # Triton's NaN-to-integer conversion is undefined. Use an in-range index
    # for the reduction, then explicitly surface the numerical failure below.
    projected_for_index = tl.where(projection_is_nan, 0.0, projected)
    lower = tl.floor(projected_for_index).to(tl.int32)
    upper = tl.minimum(lower + 1, num_bins - 1)
    fraction = projected_for_index - lower.to(tl.float32)

    probabilities = tl.where(valid_source, tl.exp(log_probs), 0.0)
    lower_mass = probabilities * (1.0 - fraction)
    upper_mass = probabilities * fraction

    # Dense one-hot reduction avoids nondeterministic atomic accumulation. Rows
    # are destination bins and columns are source bins.
    destination = offsets[:, None]
    valid_columns = valid_source[None, :]
    lower_contribution = tl.where(
        valid_columns & (lower[None, :] == destination),
        lower_mass[None, :],
        0.0,
    )
    upper_contribution = tl.where(
        valid_columns & (upper[None, :] == destination),
        upper_mass[None, :],
        0.0,
    )
    accumulated_mass = tl.sum(lower_contribution + upper_contribution, axis=1)
    accumulated_mass = tl.where(projection_is_nan, float("nan"), accumulated_mass)

    target_offsets = target_probs_ptr + batch_idx * target_probs_stride_batch + offsets * target_probs_stride_bin
    tl.store(target_offsets, accumulated_mass, mask=offsets < num_bins)


def fused_categorical_td_target(
    next_qs: torch.Tensor,
    next_q_log_probs: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    actor_entropy: torch.Tensor,
    gamma: float,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    """Launch the kernel after the optional dispatcher validates its inputs."""
    batch_size = next_qs.shape[1]
    num_bins = next_q_log_probs.shape[2]
    target_probs = torch.empty(
        (batch_size, num_bins),
        dtype=torch.float32,
        device=next_qs.device,
    )
    _fused_categorical_td_target_kernel[(batch_size,)](
        next_qs,
        next_q_log_probs,
        reward,
        done,
        actor_entropy,
        target_probs,
        num_bins,
        gamma,
        min_v,
        max_v,
        next_qs.stride(0),
        next_qs.stride(1),
        next_q_log_probs.stride(0),
        next_q_log_probs.stride(1),
        next_q_log_probs.stride(2),
        reward.stride(0),
        done.stride(0),
        actor_entropy.stride(0),
        target_probs.stride(0),
        target_probs.stride(1),
        BLOCK=_FUSED_CATEGORICAL_TARGET_BLOCK_SIZE,
        num_warps=1,
    )
    return target_probs
