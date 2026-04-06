import math
from typing import Callable

import numpy as np

EPS = 1e-8

Scheduler = Callable[[int], float]


def cyclic_exponential_decay_scheduler(
    decay_period: int, initial_value: float, final_value: float, reverse: bool = False
) -> Scheduler:
    if reverse:
        initial_value = 1 - initial_value
        final_value = 1 - final_value

    start = np.log(initial_value + EPS)
    end = np.log(final_value + EPS)

    def scheduler(step: int) -> float:
        cycle_length = decay_period
        cycle_step = step % cycle_length

        steps_left = decay_period - cycle_step
        bonus_frac = steps_left / decay_period
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (start - end) + end

        new_value = np.exp(new_value) - EPS
        if reverse:
            new_value = 1 - new_value
        return float(new_value)

    return scheduler


def warmup_cosine_decay_scheduler(
    init_value: float,
    peak_value: float,
    end_value: float,
    warmup_steps: int,
    decay_steps: int,
) -> Callable[[int], float]:
    def scheduler(step: int) -> float:
        if step < warmup_steps:
            # Warmup phase: linear interpolation from init to peak
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        # NOTE: uses optax style (`decay_steps` = total schedule length)
        # https://github.com/google-deepmind/optax/blob/main/optax/schedules/_schedule.py#L652
        elif step < decay_steps:
            # Cosine decay phase
            decay_step = step - warmup_steps
            progress = decay_step / (decay_steps - warmup_steps)
            # Cosine decay from peak to end
            return end_value + (peak_value - end_value) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return end_value

    return scheduler


def linear_decay_scheduler(decay_period: int, initial_value: float, final_value: float) -> Scheduler:
    def scheduler(step: int) -> float:
        # Ensure step does not exceed decay_period
        step = min(step, decay_period)

        # Calculate the linear interpolation factor
        fraction = step / decay_period
        new_value = (1 - fraction) * initial_value + fraction * final_value

        return new_value

    return scheduler


def constant_value_scheduler(value: float) -> Scheduler:
    """
    Returns a scheduler function that always returns the same value.

    Args:
        value (float): The constant value to return.

    Returns:
        function: A scheduler function that always returns `value`.
    """

    def scheduler(step: int) -> float:
        return value

    return scheduler
