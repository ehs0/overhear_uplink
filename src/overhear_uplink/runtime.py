from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig
from .data import MultiRobotFolderDataset, SyntheticMultiRobotDataset


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_datasets(
    data_root: str | None,
    data_config: dict[str, Any],
    seed: int,
) -> tuple[Dataset[torch.Tensor], Dataset[torch.Tensor]]:
    image_size = int(data_config["image_size"])
    num_robots = int(data_config["num_robots"])
    if data_root is None:
        train_dataset = SyntheticMultiRobotDataset(
            samples=int(data_config["synthetic_train_samples"]),
            num_robots=num_robots,
            image_size=image_size,
            seed=seed,
        )
        val_dataset = SyntheticMultiRobotDataset(
            samples=int(data_config["synthetic_val_samples"]),
            num_robots=num_robots,
            image_size=image_size,
            seed=seed + 1_000_000,
        )
        return train_dataset, val_dataset

    root = Path(data_root)
    train_root = root / "train" if (root / "train").is_dir() else root
    val_root = root / "val" if (root / "val").is_dir() else root
    train_dataset = MultiRobotFolderDataset(
        train_root,
        image_size=image_size,
        num_robots=num_robots,
        training=True,
    )
    val_dataset = MultiRobotFolderDataset(
        val_root,
        image_size=image_size,
        num_robots=num_robots,
        training=False,
    )
    return train_dataset, val_dataset


def make_loader(
    dataset: Dataset[torch.Tensor],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[torch.Tensor]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    if "model_config" not in checkpoint:
        raise ValueError("checkpoint does not contain model_config")
    return ModelConfig.from_dict(checkpoint["model_config"])

