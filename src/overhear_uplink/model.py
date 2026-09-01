from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .entropy import (
    FactorizedLogisticPrior,
    bits_from_likelihood,
    gaussian_likelihood,
    quantize,
)
from .layers import (
    AnalysisTransform,
    HyperAnalysis,
    HyperSynthesis,
    OverhearContext,
    SynthesisTransform,
)


@dataclass
class RobotPacket:
    """Ideal entropy-symbol packet transmitted by one robot.

    Symbols are intentionally kept as tensors. ``estimated_bits_*`` reports
    the entropy-model cost; this class is not an arithmetic-coded byte stream.
    """

    robot_index: int
    image_height: int
    image_width: int
    y_symbols: Tensor
    z_symbols: Tensor
    estimated_bits_y: Tensor
    estimated_bits_z: Tensor

    @property
    def estimated_bits(self) -> Tensor:
        return self.estimated_bits_y + self.estimated_bits_z

    def to(self, device: torch.device | str) -> "RobotPacket":
        return RobotPacket(
            robot_index=self.robot_index,
            image_height=self.image_height,
            image_width=self.image_width,
            y_symbols=self.y_symbols.to(device),
            z_symbols=self.z_symbols.to(device),
            estimated_bits_y=self.estimated_bits_y.to(device),
            estimated_bits_z=self.estimated_bits_z.to(device),
        )


class OverhearUplinkCodec(nn.Module):
    """Sequential learned codec for an ideal perfect-overhearing uplink."""

    downsampling_factor = 64

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        self.analysis = AnalysisTransform(cfg.channels, cfg.latent_channels)
        self.synthesis = SynthesisTransform(cfg.channels, cfg.latent_channels)
        self.hyper_analysis = HyperAnalysis(cfg.latent_channels, cfg.hyper_channels)
        self.hyper_synthesis = HyperSynthesis(cfg.latent_channels, cfg.hyper_channels)
        self.hyper_prior = FactorizedLogisticPrior(cfg.hyper_channels, cfg.min_scale)
        self.overhear_context = OverhearContext(
            channels=cfg.channels,
            heads=cfg.attention_heads,
            attention_grid=cfg.attention_grid,
            max_history=cfg.max_history,
        )
        self.context_to_prior = nn.Conv2d(
            cfg.channels, 2 * cfg.latent_channels, kernel_size=3, padding=1
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _validate_image(self, image: Tensor) -> None:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [B, 3, H, W]")
        height, width = image.shape[-2:]
        if height % self.downsampling_factor or width % self.downsampling_factor:
            raise ValueError(
                f"H and W must be divisible by {self.downsampling_factor}; "
                f"received {(height, width)}"
            )

    def _history_tensor(
        self,
        history: Sequence[Tensor],
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if not self.config.use_overhear or len(history) == 0:
            return None
        selected = list(history[-self.config.max_history :])
        for item in selected:
            if item.shape != (batch, 3, height, width):
                raise ValueError("all history images must match the current image shape")
        stacked = torch.stack([item.to(device=device, dtype=dtype) for item in selected], dim=1)
        if self.config.detach_history:
            stacked = stacked.detach()
        return stacked

    def _context(
        self,
        history: Sequence[Tensor],
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        history_tensor = self._history_tensor(
            history, batch, height, width, device, dtype
        )
        if history_tensor is None:
            prediction = torch.zeros(
                batch, 3, height, width, device=device, dtype=dtype
            )
            context = torch.zeros(
                batch,
                self.config.channels,
                height // 16,
                width // 16,
                device=device,
                dtype=dtype,
            )
            return prediction, context
        return self.overhear_context(history_tensor)

    def _prior_parameters(self, z_hat: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        parameters = self.hyper_synthesis(z_hat) + self.context_to_prior(context)
        means, log_scales = parameters.chunk(2, dim=1)
        scales = F.softplus(log_scales) + self.config.min_scale
        return means, scales

    def _codec_step(
        self,
        image: Tensor,
        history: Sequence[Tensor],
    ) -> dict[str, Tensor]:
        self._validate_image(image)
        batch, _, height, width = image.shape
        prediction, context = self._context(
            history, batch, height, width, image.device, image.dtype
        )

        residual = image - prediction
        y = self.analysis(residual)
        z = self.hyper_analysis(y)
        z_hat = quantize(z, self.training)
        means, scales = self._prior_parameters(z_hat, context)
        y_hat = quantize(y, self.training, means)

        residual_hat = self.synthesis(y_hat)
        reconstruction = (prediction + residual_hat).clamp(0.0, 1.0)
        y_likelihood = gaussian_likelihood(y_hat, means, scales)
        z_likelihood = self.hyper_prior.likelihood(z_hat)

        return {
            "x_hat": reconstruction,
            "prediction": prediction,
            "y": y,
            "z": z,
            "y_hat": y_hat,
            "z_hat": z_hat,
            "means": means,
            "scales": scales,
            "bits_y": bits_from_likelihood(y_likelihood),
            "bits_z": bits_from_likelihood(z_likelihood),
        }

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        """Process robot views in transmission order.

        Args:
            images: Tensor shaped ``[B, K, 3, H, W]`` in the range [0, 1].
        """

        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B, K, 3, H, W]")
        if images.shape[1] < 1:
            raise ValueError("at least one robot view is required")

        history: list[Tensor] = []
        reconstructions: list[Tensor] = []
        predictions: list[Tensor] = []
        bits_y: list[Tensor] = []
        bits_z: list[Tensor] = []
        for robot_index in range(images.shape[1]):
            result = self._codec_step(images[:, robot_index], history)
            reconstruction = result["x_hat"]
            reconstructions.append(reconstruction)
            predictions.append(result["prediction"])
            bits_y.append(result["bits_y"])
            bits_z.append(result["bits_z"])
            history.append(reconstruction)

        return {
            "x_hat": torch.stack(reconstructions, dim=1),
            "predictions": torch.stack(predictions, dim=1),
            "bits_y": torch.stack(bits_y, dim=1),
            "bits_z": torch.stack(bits_z, dim=1),
        }

    @torch.no_grad()
    def encode_robot(
        self,
        image: Tensor,
        history: Sequence[Tensor],
        robot_index: int,
    ) -> tuple[RobotPacket, Tensor]:
        """Create one ideal uplink packet and the encoder-side reconstruction."""

        if self.training:
            raise RuntimeError("call model.eval() before packet encoding")
        if robot_index != len(history):
            raise ValueError(
                "robot_index must equal the number of previously reconstructed packets"
            )
        result = self._codec_step(image, history)
        y_symbols = torch.round(result["y"] - result["means"]).to(torch.int32)
        z_symbols = torch.round(result["z"]).to(torch.int32)
        packet = RobotPacket(
            robot_index=robot_index,
            image_height=image.shape[-2],
            image_width=image.shape[-1],
            y_symbols=y_symbols,
            z_symbols=z_symbols,
            estimated_bits_y=result["bits_y"],
            estimated_bits_z=result["bits_z"],
        )
        return packet, result["x_hat"]

    @torch.no_grad()
    def decode_robot(
        self,
        packet: RobotPacket,
        history: Sequence[Tensor],
    ) -> Tensor:
        """Decode a packet using the same previously reconstructed history."""

        if self.training:
            raise RuntimeError("call model.eval() before packet decoding")
        if packet.robot_index != len(history):
            raise ValueError("packets must be decoded once and in robot order")
        y_symbols = packet.y_symbols.to(self.device)
        z_hat = packet.z_symbols.to(device=self.device, dtype=next(self.parameters()).dtype)
        batch = y_symbols.shape[0]
        prediction, context = self._context(
            history,
            batch=batch,
            height=packet.image_height,
            width=packet.image_width,
            device=self.device,
            dtype=z_hat.dtype,
        )
        means, _ = self._prior_parameters(z_hat, context)
        if means.shape != y_symbols.shape:
            raise ValueError("packet latent shape is incompatible with this model")
        y_hat = y_symbols.to(dtype=means.dtype) + means
        residual_hat = self.synthesis(y_hat)
        return (prediction + residual_hat).clamp(0.0, 1.0)
