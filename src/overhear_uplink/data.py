from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _natural_key(path: Path) -> list[tuple[int, str | int]]:
    return [
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", path.name)
    ]


def _pil_to_tensor(image: Image.Image) -> Tensor:
    image = image.convert("RGB")
    width, height = image.size
    buffer = bytearray(image.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).clone().view(height, width, 3)
    return tensor.permute(2, 0, 1).float().div_(255.0)


def _discover_scenes(root: Path, num_robots: int | None) -> list[list[Path]]:
    manifest = root / "manifest.json"
    scenes: list[list[Path]] = []
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload["scenes"] if isinstance(payload, dict) else payload
        for record in records:
            views = [manifest.parent / item for item in record["views"]]
            scenes.append(views)
    else:
        for scene_dir in sorted((item for item in root.iterdir() if item.is_dir())):
            views = sorted(
                (
                    item
                    for item in scene_dir.iterdir()
                    if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
                ),
                key=_natural_key,
            )
            if views:
                scenes.append(views)

    if not scenes:
        raise ValueError(f"no multi-view scenes found under {root}")
    required = num_robots or len(scenes[0])
    if required < 1:
        raise ValueError("num_robots must be positive")
    normalized: list[list[Path]] = []
    for views in scenes:
        if len(views) < required:
            raise ValueError(
                f"scene has {len(views)} views but {required} robots were requested: {views}"
            )
        selected = views[:required]
        missing = [path for path in selected if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"manifest refers to missing images: {missing}")
        normalized.append(selected)
    return normalized


class MultiRobotFolderDataset(Dataset[Tensor]):
    """Load ordered robot views and apply the same crop to every view."""

    def __init__(
        self,
        root: str | Path,
        image_size: int,
        num_robots: int | None = None,
        training: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.training = training
        if image_size <= 0 or image_size % 64:
            raise ValueError("image_size must be a positive multiple of 64")
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.scenes = _discover_scenes(self.root, num_robots)

    def __len__(self) -> int:
        return len(self.scenes)

    def _resize_if_needed(self, images: Sequence[Image.Image]) -> list[Image.Image]:
        sizes = {image.size for image in images}
        if len(sizes) != 1:
            raise ValueError("all robot views in a scene must have the same resolution")
        width, height = images[0].size
        scale = max(self.image_size / width, self.image_size / height, 1.0)
        if scale == 1.0:
            return list(images)
        resized = (int(round(width * scale)), int(round(height * scale)))
        return [image.resize(resized, Image.Resampling.BICUBIC) for image in images]

    def __getitem__(self, index: int) -> Tensor:
        with_images = [Image.open(path).convert("RGB") for path in self.scenes[index]]
        try:
            images = self._resize_if_needed(with_images)
            width, height = images[0].size
            if self.training:
                left = random.randint(0, width - self.image_size)
                top = random.randint(0, height - self.image_size)
                flip = random.random() < 0.5
            else:
                left = (width - self.image_size) // 2
                top = (height - self.image_size) // 2
                flip = False
            box = (left, top, left + self.image_size, top + self.image_size)
            cropped = [image.crop(box) for image in images]
            if flip:
                cropped = [image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for image in cropped]
            return torch.stack([_pil_to_tensor(image) for image in cropped], dim=0)
        finally:
            for image in with_images:
                image.close()


class SyntheticMultiRobotDataset(Dataset[Tensor]):
    """Deterministic correlated views for smoke tests and pipeline debugging."""

    def __init__(
        self,
        samples: int = 256,
        num_robots: int = 3,
        image_size: int = 128,
        seed: int = 0,
    ) -> None:
        if samples <= 0 or num_robots <= 0:
            raise ValueError("samples and num_robots must be positive")
        if image_size <= 0 or image_size % 64:
            raise ValueError("image_size must be a positive multiple of 64")
        self.samples = samples
        self.num_robots = num_robots
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Tensor:
        generator = torch.Generator().manual_seed(self.seed + index)
        low_resolution = max(self.image_size // 8, 4)
        base = torch.rand(1, 3, low_resolution, low_resolution, generator=generator)
        base = F.interpolate(
            base,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )[0].clamp(0.0, 1.0)

        views: list[Tensor] = []
        center = (self.num_robots - 1) / 2.0
        for robot_index in range(self.num_robots):
            offset = robot_index - center
            shift_x = int(round(3.0 * offset))
            shift_y = int(round(1.5 * offset))
            view = torch.roll(base, shifts=(shift_y, shift_x), dims=(1, 2))
            gain = 1.0 + 0.025 * offset
            bias = 0.01 * offset
            view = (gain * view + bias).clamp(0.0, 1.0)

            # A small view-specific region represents occlusion/content change,
            # not communication-channel noise.
            patch = max(self.image_size // 12, 2)
            top = (index * 7 + robot_index * 13) % (self.image_size - patch + 1)
            left = (index * 11 + robot_index * 5) % (self.image_size - patch + 1)
            view = view.clone()
            color = torch.rand(3, 1, 1, generator=generator)
            view[:, top : top + patch, left : left + patch] = color
            views.append(view)
        return torch.stack(views, dim=0)
