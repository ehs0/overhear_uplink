from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings shared by training and packet decoding."""

    channels: int = 48
    latent_channels: int = 72
    hyper_channels: int = 48
    attention_heads: int = 4
    attention_grid: int = 12
    max_history: int = 4
    min_scale: float = 0.11
    use_overhear: bool = True
    detach_history: bool = False

    def __post_init__(self) -> None:
        if self.channels <= 0 or self.latent_channels <= 0:
            raise ValueError("channel counts must be positive")
        if self.channels % self.attention_heads != 0:
            raise ValueError("channels must be divisible by attention_heads")
        if self.attention_grid <= 0 or self.max_history <= 0:
            raise ValueError("attention_grid and max_history must be positive")
        if self.min_scale <= 0:
            raise ValueError("min_scale must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

