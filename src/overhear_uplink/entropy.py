from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def quantize(
    values: Tensor,
    training: bool,
    means: Tensor | None = None,
) -> Tensor:
    """Uniform-noise relaxation in training and rounding during evaluation."""

    if means is None:
        means = torch.zeros((), device=values.device, dtype=values.dtype)
    centered = values - means
    if training:
        centered = centered + torch.empty_like(centered).uniform_(-0.5, 0.5)
    else:
        centered = torch.round(centered)
    return centered + means


def gaussian_likelihood(
    values: Tensor,
    means: Tensor,
    scales: Tensor,
    minimum: float = 1e-9,
) -> Tensor:
    """Probability mass of unit-width bins under a conditional Gaussian."""

    centered = values - means
    inv_std = 1.0 / scales
    sqrt_two = math.sqrt(2.0)
    upper = 0.5 * (1.0 + torch.erf((centered + 0.5) * inv_std / sqrt_two))
    lower = 0.5 * (1.0 + torch.erf((centered - 0.5) * inv_std / sqrt_two))
    return (upper - lower).clamp_min(minimum)


def logistic_likelihood(
    values: Tensor,
    scales: Tensor,
    minimum: float = 1e-9,
) -> Tensor:
    """Probability mass of unit-width bins under a zero-mean logistic prior."""

    upper = torch.sigmoid((values + 0.5) / scales)
    lower = torch.sigmoid((values - 0.5) / scales)
    return (upper - lower).clamp_min(minimum)


def bits_from_likelihood(likelihood: Tensor) -> Tensor:
    """Return an estimated number of bits for every batch item."""

    return -torch.log2(likelihood).flatten(1).sum(dim=1)


class FactorizedLogisticPrior(nn.Module):
    """A learned, channel-wise prior for hyper-latent symbols."""

    def __init__(self, channels: int, min_scale: float) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.log_scale = nn.Parameter(torch.zeros(channels))

    def scales(self) -> Tensor:
        return F.softplus(self.log_scale).view(1, -1, 1, 1) + self.min_scale

    def likelihood(self, values: Tensor) -> Tensor:
        return logistic_likelihood(values, self.scales())

