import math

import torch
import torch.nn as nn

from flash_rl.agents.flashSAC.layer import (
    EnsembleCategoricalValue,
    EnsembleFlashSACBlock,
    EnsembleFlashSACEmbedder,
    EnsembleUnitRMSNorm,
    FlashSACBlock,
    FlashSACEmbedder,
    NormalTanhPolicy,
    UnitRMSNorm,
)


class FlashSACActor(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(hidden_dim=hidden_dim, action_dim=action_dim)

    def get_mean_and_std(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        mean, std = self.predictor.get_mean_and_std(x, training)
        return mean, std

    def forward(
        self,
        observations: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = observations
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        actions, info = self.predictor(x, training)
        return actions, info


class FlashSACDoubleCritic(nn.Module):
    """
    Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3

    Fuses N parallel critic networks into single batched operations.
    All internal computation uses (N, batch, dim) tensor layout.
    """

    def __init__(
        self,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ):
        super().__init__()
        self.num_qs = num_qs

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        training: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((observations, actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)
