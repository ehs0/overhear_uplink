from __future__ import annotations

import argparse

import torch

from .config import ModelConfig
from .data import SyntheticMultiRobotDataset
from .model import OverhearUplinkCodec
from .runtime import choose_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-robots", type=int, default=3)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    if args.checkpoint is None:
        model = OverhearUplinkCodec(ModelConfig()).to(device)
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = OverhearUplinkCodec(
            ModelConfig.from_dict(checkpoint["model_config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"])
    model.eval()

    views = SyntheticMultiRobotDataset(
        samples=1,
        num_robots=args.num_robots,
        image_size=args.image_size,
        seed=123,
    )[0].unsqueeze(0).to(device)
    encoder_history: list[torch.Tensor] = []
    decoder_history: list[torch.Tensor] = []
    total_bits = 0.0
    for robot_index in range(args.num_robots):
        packet, encoder_reconstruction = model.encode_robot(
            views[:, robot_index], encoder_history, robot_index
        )
        decoder_reconstruction = model.decode_robot(packet, decoder_history)
        difference = float(
            (encoder_reconstruction - decoder_reconstruction).abs().max().cpu()
        )
        bits = float(packet.estimated_bits.sum().cpu())
        bpp = bits / (args.image_size * args.image_size)
        total_bits += bits
        print(
            f"robot={robot_index} estimated_bpp={bpp:.4f} "
            f"encoder_decoder_max_error={difference:.3e}"
        )
        if difference > 1e-6:
            raise RuntimeError("encoder and decoder lost their shared overhear state")
        encoder_history.append(encoder_reconstruction)
        decoder_history.append(decoder_reconstruction)
    average_bpp = total_bits / (
        args.num_robots * args.image_size * args.image_size
    )
    print(f"sequence_average_estimated_bpp={average_bpp:.4f}")


if __name__ == "__main__":
    main()

