from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import MultiRobotFolderDataset, SyntheticMultiRobotDataset
from .engine import evaluate_loader
from .losses import RateDistortionLoss
from .model import OverhearUplinkCodec
from .runtime import choose_device, make_loader, model_config_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--synthetic-samples", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-robots", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = OverhearUplinkCodec(model_config_from_checkpoint(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    loss_config = checkpoint.get(
        "loss_config", {"lambda_rd": 128.0, "prediction_weight": 0.05}
    )
    criterion = RateDistortionLoss(**loss_config)
    stored_data = checkpoint.get("data_config", {})
    image_size = args.image_size or int(stored_data.get("image_size", 128))
    num_robots = args.num_robots or int(stored_data.get("num_robots", 3))

    if args.data_root is None:
        dataset = SyntheticMultiRobotDataset(
            samples=args.synthetic_samples,
            num_robots=num_robots,
            image_size=image_size,
            seed=2_000_000,
        )
    else:
        data_root = Path(args.data_root)
        if (data_root / "val").is_dir():
            data_root = data_root / "val"
        dataset = MultiRobotFolderDataset(
            data_root,
            image_size=image_size,
            num_robots=num_robots,
            training=False,
        )
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        device=device,
    )
    metrics = evaluate_loader(model, loader, criterion, device)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "samples": len(dataset),
        "num_robots": num_robots,
        "image_size": image_size,
        "metrics": metrics,
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output_json is not None:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
