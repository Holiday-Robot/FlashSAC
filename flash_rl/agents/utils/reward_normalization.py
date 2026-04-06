import os
from typing import TypeVar

import torch

Config = TypeVar("Config")


@torch.compile
def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    G_r_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    done = torch.logical_or(terminated, truncated).float()
    new_G_r = gamma * (1.0 - done) * G_r + reward
    new_G_r_max = torch.maximum(G_r_max, torch.max(torch.abs(new_G_r)))
    return new_G_r, new_G_r_max


@torch.compile
def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    var_denominator = torch.sqrt(G_var + eps)
    min_required_denominator = G_r_max / G_max
    denominator = torch.maximum(var_denominator, min_required_denominator)
    return rewards / denominator


@torch.compile
def _update_mean_var_count_from_moments(
    samples: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    sample_mean = torch.mean(samples, dim=0)
    sample_var = torch.var(samples, dim=0, unbiased=False)
    sample_count = float(samples.shape[0])

    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    M2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = M2 / total_count

    return (
        new_mean,
        new_var,
        total_count,
    )


class RewardNormalizer:
    """
    Normalize rewards using the variance of a running estimate of the discounted returns.
    In policy gradient methods, the update rule often involves the term ∇log π(a|s)⋅G_t, where G_t is the return
    from time t. Scaling G_t to have unit variance can be an effective variance reduction technique.

    Note: Return statistics is updated only on sample_actions with training == True
    """

    def __init__(
        self,
        gamma: float,
        G_max: float,
        load_rms: bool,
        device: torch.device,
        epsilon: float = 1e-8,
    ):
        self.gamma = gamma
        # running estimates of the discounted return
        self.G_r = torch.zeros(1, dtype=torch.float32, device=device)  # Lazy init to num_envs
        self.G_r_max = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_rms = RunningMeanStd(
            shape=(1,),
            device=device,
            dtype=torch.float32,
        )
        self.G_max = G_max
        self.load_rms = load_rms
        self.epsilon = epsilon
        self.device = device

    def update_reward_stats(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        self.G_r, self.G_r_max = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            G_r=self.G_r,
            G_r_max=self.G_r_max,
            gamma=self.gamma,
        )
        self.G_rms.update(self.G_r)

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        normalized_rewards = _scale_reward(
            rewards=rewards,
            G_var=self.G_rms.var,
            G_r_max=self.G_r_max,
            G_max=self.G_max,
            eps=self.epsilon,
        )
        return normalized_rewards

    def save(self, path: str) -> None:
        """Save reward normalizer state.
        args:
            path (str): The full file path (e.g. "checkpoints/reward_normalizer.pt").
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "G_r": self.G_r,
            "G_r_max": self.G_r_max,
            "G_rms_mean": self.G_rms.mean,
            "G_rms_var": self.G_rms.var,
            "G_rms_count": self.G_rms.count,
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load reward normalizer state.
        args:
            path (str): The full file path (e.g. "checkpoints/reward_normalizer.pt").
        """
        state = torch.load(path, map_location=self.device)
        self.G_r = state["G_r"]
        self.G_r_max = state["G_r_max"]
        self.G_rms.mean = state["G_rms_mean"]
        self.G_rms.var = state["G_rms_var"]
        self.G_rms.count = state["G_rms_count"]


class RunningMeanStd:
    """
    Tracks the mean, variance and count of values.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        device: torch.device,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        """Updates the mean, var and count from a batch of samples."""
        self.mean, self.var, self.count = _update_mean_var_count_from_moments(
            samples=x,
            running_mean=self.mean,
            running_var=self.var,
            running_count=self.count,
            epsilon=self.epsilon,
        )
