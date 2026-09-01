from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RateDistortionLoss(nn.Module):
    def __init__(self, lambda_rd: float = 128.0, prediction_weight: float = 0.05) -> None:
        super().__init__()
        if lambda_rd <= 0 or prediction_weight < 0:
            raise ValueError("invalid rate-distortion loss weights")
        self.lambda_rd = lambda_rd
        self.prediction_weight = prediction_weight

    def forward(self, output: dict[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        if output["x_hat"].shape != target.shape:
            raise ValueError("reconstruction and target shapes must match")
        batch, robots, _, height, width = target.shape
        total_pixels = batch * robots * height * width
        rate_bpp = (output["bits_y"].sum() + output["bits_z"].sum()) / total_pixels
        mse = F.mse_loss(output["x_hat"], target)

        if robots > 1:
            prediction_mse = F.mse_loss(
                output["predictions"][:, 1:], target[:, 1:]
            )
        else:
            prediction_mse = target.new_zeros(())
        loss = (
            rate_bpp
            + self.lambda_rd * mse
            + self.prediction_weight * prediction_mse
        )
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-10))
        bpp_per_robot = (
            output["bits_y"] + output["bits_z"]
        ).mean(dim=0) / (height * width)
        mse_per_robot = (output["x_hat"] - target).square().mean(dim=(0, 2, 3, 4))
        psnr_per_robot = -10.0 * torch.log10(mse_per_robot.clamp_min(1e-10))
        return {
            "loss": loss,
            "bpp": rate_bpp,
            "mse": mse,
            "psnr": psnr,
            "prediction_mse": prediction_mse,
            "bpp_per_robot": bpp_per_robot,
            "psnr_per_robot": psnr_per_robot,
        }


def psnr_from_mse(mse: float) -> float:
    return -10.0 * math.log10(max(mse, 1e-10))

