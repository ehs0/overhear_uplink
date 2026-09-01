from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch
from torch import Tensor

from .losses import RateDistortionLoss
from .model import OverhearUplinkCodec


def train_one_epoch(
    model: OverhearUplinkCodec,
    loader: Iterable[Tensor],
    criterion: RateDistortionLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> dict[str, float | list[float]]:
    model.train()
    totals: dict[str, float] = defaultdict(float)
    per_robot_bpp: Tensor | None = None
    per_robot_psnr: Tensor | None = None
    samples = 0

    for images in loader:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        metrics = criterion(output, images)
        metrics["loss"].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch = images.shape[0]
        samples += batch
        for key in ("loss", "bpp", "mse", "psnr", "prediction_mse"):
            totals[key] += float(metrics[key].detach()) * batch
        bpp = metrics["bpp_per_robot"].detach().cpu() * batch
        psnr = metrics["psnr_per_robot"].detach().cpu() * batch
        per_robot_bpp = bpp if per_robot_bpp is None else per_robot_bpp + bpp
        per_robot_psnr = psnr if per_robot_psnr is None else per_robot_psnr + psnr

    if samples == 0 or per_robot_bpp is None or per_robot_psnr is None:
        raise ValueError("training loader produced no batches")
    result: dict[str, float | list[float]] = {
        key: value / samples for key, value in totals.items()
    }
    result["bpp_per_robot"] = (per_robot_bpp / samples).tolist()
    result["psnr_per_robot"] = (per_robot_psnr / samples).tolist()
    return result


@torch.no_grad()
def evaluate_loader(
    model: OverhearUplinkCodec,
    loader: Iterable[Tensor],
    criterion: RateDistortionLoss,
    device: torch.device,
) -> dict[str, float | list[float]]:
    model.eval()
    totals: dict[str, float] = defaultdict(float)
    bits_per_robot: Tensor | None = None
    squared_error_per_robot: Tensor | None = None
    pixel_count = 0
    samples = 0

    for images in loader:
        images = images.to(device)
        output = model(images)
        metrics = criterion(output, images)
        batch, _, _, height, width = images.shape
        samples += batch
        pixel_count += batch * height * width
        for key in ("loss", "bpp", "mse", "prediction_mse"):
            totals[key] += float(metrics[key]) * batch

        batch_bits = (output["bits_y"] + output["bits_z"]).sum(dim=0).cpu()
        squared_error = (
            (output["x_hat"] - images).square().sum(dim=(0, 2, 3, 4)).cpu()
        )
        bits_per_robot = (
            batch_bits if bits_per_robot is None else bits_per_robot + batch_bits
        )
        squared_error_per_robot = (
            squared_error
            if squared_error_per_robot is None
            else squared_error_per_robot + squared_error
        )

    if (
        samples == 0
        or bits_per_robot is None
        or squared_error_per_robot is None
        or pixel_count == 0
    ):
        raise ValueError("evaluation loader produced no batches")
    bpp_per_robot = bits_per_robot / pixel_count
    mse_per_robot = squared_error_per_robot / (3 * pixel_count)
    psnr_per_robot = -10.0 * torch.log10(mse_per_robot.clamp_min(1e-10))
    mean_mse = float(squared_error_per_robot.sum() / (3 * pixel_count * len(mse_per_robot)))
    result: dict[str, float | list[float]] = {
        key: value / samples for key, value in totals.items()
    }
    result["psnr"] = float(-10.0 * torch.log10(torch.tensor(max(mean_mse, 1e-10))))
    result["bpp_per_robot"] = bpp_per_robot.tolist()
    result["psnr_per_robot"] = psnr_per_robot.tolist()
    return result

