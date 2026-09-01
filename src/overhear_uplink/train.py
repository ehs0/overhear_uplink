from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .engine import evaluate_loader, train_one_epoch
from .losses import RateDistortionLoss
from .model import OverhearUplinkCodec
from .runtime import choose_device, load_json, make_datasets, make_loader, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def _format_metrics(metrics: dict[str, float | list[float]]) -> str:
    compact = {
        key: ([round(item, 4) for item in value] if isinstance(value, list) else round(value, 6))
        for key, value in metrics.items()
    }
    return json.dumps(compact, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    model_config = ModelConfig.from_dict(config["model"])
    data_config: dict[str, Any] = config["data"]
    train_config: dict[str, Any] = config["training"]
    if args.epochs is not None:
        train_config["epochs"] = args.epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_config["num_workers"] = args.num_workers

    seed = int(train_config["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    train_dataset, val_dataset = make_datasets(args.data_root, data_config, seed)
    train_loader = make_loader(
        train_dataset,
        batch_size=int(train_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        shuffle=True,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=int(train_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        shuffle=False,
        device=device,
    )

    model = OverhearUplinkCodec(model_config).to(device)
    criterion = RateDistortionLoss(
        lambda_rd=float(train_config["lambda_rd"]),
        prediction_weight=float(train_config["prediction_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    start_epoch = 0
    best_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["model_config"] != model_config.to_dict():
            raise ValueError("resume checkpoint model config does not match --config")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))

    print(f"device={device} train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            grad_clip=float(train_config["grad_clip"]),
        )
        val_metrics = evaluate_loader(model, val_loader, criterion, device)
        print(f"epoch={epoch:04d} train={_format_metrics(train_metrics)}")
        print(f"epoch={epoch:04d} val={_format_metrics(val_metrics)}")

        is_best = float(val_metrics["loss"]) < best_loss
        if is_best:
            best_loss = float(val_metrics["loss"])
        checkpoint = {
            "epoch": epoch,
            "best_loss": best_loss,
            "model_config": model_config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss_config": {
                "lambda_rd": criterion.lambda_rd,
                "prediction_weight": criterion.prediction_weight,
            },
            "data_config": data_config,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()

