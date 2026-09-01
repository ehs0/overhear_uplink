from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    value_tensor = torch.tensor(float(value))
    return float(torch.log(torch.expm1(value_tensor)))


class GDN(nn.Module):
    """Lightweight channel-wise generalized divisive normalization."""

    def __init__(self, channels: int, inverse: bool = False) -> None:
        super().__init__()
        self.inverse = inverse
        self.beta_reparam = nn.Parameter(
            torch.full((channels,), _inverse_softplus(1.0))
        )
        self.gamma_reparam = nn.Parameter(
            torch.full((channels,), _inverse_softplus(0.1))
        )

    def forward(self, inputs: Tensor) -> Tensor:
        beta = F.softplus(self.beta_reparam).view(1, -1, 1, 1) + 1e-6
        gamma = F.softplus(self.gamma_reparam).view(1, -1, 1, 1)
        norm = torch.sqrt(beta + gamma * inputs.square())
        return inputs * norm if self.inverse else inputs / norm


def conv_down(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)


def conv_up(in_channels: int, out_channels: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=2,
        padding=2,
        output_padding=1,
    )


class AnalysisTransform(nn.Module):
    def __init__(self, channels: int, latent_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_down(3, channels),
            GDN(channels),
            conv_down(channels, channels),
            GDN(channels),
            conv_down(channels, channels),
            GDN(channels),
            conv_down(channels, latent_channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class SynthesisTransform(nn.Module):
    def __init__(self, channels: int, latent_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_up(latent_channels, channels),
            GDN(channels, inverse=True),
            conv_up(channels, channels),
            GDN(channels, inverse=True),
            conv_up(channels, channels),
            GDN(channels, inverse=True),
            conv_up(channels, 3),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class HyperAnalysis(nn.Module):
    def __init__(self, latent_channels: int, hyper_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            conv_down(hyper_channels, hyper_channels),
            nn.ReLU(inplace=True),
            conv_down(hyper_channels, hyper_channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class HyperSynthesis(nn.Module):
    def __init__(self, latent_channels: int, hyper_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_up(hyper_channels, hyper_channels),
            nn.ReLU(inplace=True),
            conv_up(hyper_channels, hyper_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyper_channels, 2 * latent_channels, 3, padding=1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class ReferenceEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.net = nn.Sequential(
            conv_down(3, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            conv_down(hidden, channels),
            nn.LeakyReLU(0.1, inplace=True),
            conv_down(channels, channels),
            nn.LeakyReLU(0.1, inplace=True),
            conv_down(channels, channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


class ReferenceDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.net = nn.Sequential(
            conv_up(channels, channels),
            nn.LeakyReLU(0.1, inplace=True),
            conv_up(channels, channels),
            nn.LeakyReLU(0.1, inplace=True),
            conv_up(channels, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            conv_up(hidden, 3),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(self.net(inputs))


class OverhearContext(nn.Module):
    """Fuse reconstructed prior views using global cross-attention.

    The latest reconstruction supplies the query. All available reconstructed
    views supply keys and values. Attention is performed on a bounded grid so
    memory does not grow quadratically with the input image resolution.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        attention_grid: int,
        max_history: int,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.attention_grid = attention_grid
        self.max_history = max_history
        self.encoder = ReferenceEncoder(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )
        self.norm2 = nn.LayerNorm(channels)
        self.decoder = ReferenceDecoder(channels)

    def forward(self, history: Tensor) -> tuple[Tensor, Tensor]:
        if history.ndim != 5 or history.shape[1] < 1:
            raise ValueError("history must have shape [B, T, 3, H, W] with T >= 1")

        history = history[:, -self.max_history :]
        batch, views, channels, height, width = history.shape
        encoded = self.encoder(history.reshape(batch * views, channels, height, width))
        _, feature_channels, feature_height, feature_width = encoded.shape
        encoded = encoded.reshape(
            batch, views, feature_channels, feature_height, feature_width
        )

        grid_height = min(feature_height, self.attention_grid)
        grid_width = min(feature_width, self.attention_grid)
        pooled = F.adaptive_avg_pool2d(
            encoded.reshape(batch * views, feature_channels, feature_height, feature_width),
            (grid_height, grid_width),
        ).reshape(batch, views, feature_channels, grid_height, grid_width)

        all_tokens = pooled.permute(0, 1, 3, 4, 2).reshape(
            batch, views * grid_height * grid_width, feature_channels
        )
        latest_tokens = pooled[:, -1].permute(0, 2, 3, 1).reshape(
            batch, grid_height * grid_width, feature_channels
        )
        attended, _ = self.attention(
            query=latest_tokens,
            key=all_tokens,
            value=all_tokens,
            need_weights=False,
        )
        fused_tokens = self.norm1(latest_tokens + attended)
        fused_tokens = self.norm2(fused_tokens + self.feed_forward(fused_tokens))
        fused = fused_tokens.reshape(
            batch, grid_height, grid_width, feature_channels
        ).permute(0, 3, 1, 2)
        if (grid_height, grid_width) != (feature_height, feature_width):
            fused = F.interpolate(
                fused,
                size=(feature_height, feature_width),
                mode="bilinear",
                align_corners=False,
            )
        prediction = self.decoder(fused)
        return prediction, fused

