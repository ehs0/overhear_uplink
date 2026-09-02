"""Compare overhear-ON and overhear-OFF codecs on a shared evaluation set.

Evaluates every requested checkpoint on the *same* synthetic multi-view split,
then reports per-robot rate/distortion, the overhearing rate saving G_R, and a
Bjontegaard delta-rate over the lambda sweep.
"""

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

EVAL_SEED = 2_000_000


def build_batches(
    device: torch.device,
    samples: int,
    data_root: str | None,
    batch_size: int,
    image_size: int,
    num_robots: int,
) -> tuple[list[torch.Tensor], int]:
    """Materialise the evaluation set once so every checkpoint sees identical data."""
    if data_root is None:
        dataset = SyntheticMultiRobotDataset(
            samples=samples,
            num_robots=num_robots,
            image_size=image_size,
            seed=EVAL_SEED,
        )
    else:
        root = Path(data_root)
        if (root / "val").is_dir():
            root = root / "val"
        dataset = MultiRobotFolderDataset(
            root, image_size=image_size, num_robots=num_robots, training=False
        )
    loader = make_loader(
        dataset, batch_size=batch_size, num_workers=0, shuffle=False, device=device
    )
    return [images.to(device) for images in loader], len(dataset)


def evaluate_checkpoint(
    path: Path,
    device: torch.device,
    batches: list[torch.Tensor],
    total_samples: int,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model_config = model_config_from_checkpoint(checkpoint)
    model = OverhearUplinkCodec(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    loss_config = checkpoint.get("loss_config", {"lambda_rd": 128.0, "prediction_weight": 0.05})
    criterion = RateDistortionLoss(**loss_config)
    stored = checkpoint.get("data_config", {})
    image_size = int(stored.get("image_size", 128))
    num_robots = int(stored.get("num_robots", 3))

    metrics = evaluate_loader(model, batches, criterion, device)
    return {
        "checkpoint": str(path),
        "use_overhear": bool(model_config.use_overhear),
        "lambda_rd": float(loss_config["lambda_rd"]),
        "samples": total_samples,
        "num_robots": num_robots,
        "image_size": image_size,
        "bpp": float(metrics["bpp"]),
        "psnr": float(metrics["psnr"]),
        "mse": float(metrics["mse"]),
        "bpp_per_robot": list(metrics["bpp_per_robot"]),
        "psnr_per_robot": list(metrics["psnr_per_robot"]),
    }


def _cubic_spline_integral(x: list[float], y: list[float], lo: float, hi: float) -> float:
    """Integrate a natural cubic spline y(x) over [lo, hi]; x must be increasing."""
    n = len(x)
    if n < 2:
        raise ValueError("need at least two points")
    if n == 2:  # linear fallback
        slope = (y[1] - y[0]) / (x[1] - x[0])

        def anti(t: float) -> float:
            return y[0] * (t - x[0]) + 0.5 * slope * (t - x[0]) ** 2

        return anti(hi) - anti(lo)

    h = [x[i + 1] - x[i] for i in range(n - 1)]
    # Natural cubic spline second derivatives via Thomas algorithm.
    alpha = [0.0] * n
    for i in range(1, n - 1):
        alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    l = [1.0] + [0.0] * (n - 1)
    mu = [0.0] * n
    z = [0.0] * n
    for i in range(1, n - 1):
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    c = [0.0] * n
    b = [0.0] * (n - 1)
    d = [0.0] * (n - 1)
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        d[j] = (c[j + 1] - c[j]) / (3.0 * h[j])

    def anti(t: float, j: int) -> float:
        s = t - x[j]
        return y[j] * s + b[j] * s**2 / 2.0 + c[j] * s**3 / 3.0 + d[j] * s**4 / 4.0

    total = 0.0
    for j in range(n - 1):
        a_lo, a_hi = max(lo, x[j]), min(hi, x[j + 1])
        if a_hi > a_lo:
            total += anti(a_hi, j) - anti(a_lo, j)
    return total


def bd_rate(anchor: list[tuple[float, float]], test: list[tuple[float, float]]) -> float | None:
    """Bjontegaard delta rate (%) of `test` vs `anchor`; points are (bpp, psnr)."""
    import math

    a = sorted(anchor, key=lambda p: p[1])
    t = sorted(test, key=lambda p: p[1])
    if len(a) < 2 or len(t) < 2:
        return None
    lo = max(a[0][1], t[0][1])
    hi = min(a[-1][1], t[-1][1])
    if hi <= lo:
        return None  # no overlapping PSNR range
    int_a = _cubic_spline_integral([p[1] for p in a], [math.log10(p[0]) for p in a], lo, hi)
    int_t = _cubic_spline_integral([p[1] for p in t], [math.log10(p[0]) for p in t], lo, hi)
    return (10.0 ** ((int_t - int_a) / (hi - lo)) - 1.0) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="path to a best.pt; repeatable",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--synthetic-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    probe = torch.load(args.checkpoint[0], map_location="cpu", weights_only=False)
    probe_data = probe.get("data_config", {})
    batches, total_samples = build_batches(
        device,
        args.synthetic_samples,
        args.data_root,
        args.batch_size,
        int(probe_data.get("image_size", 128)),
        int(probe_data.get("num_robots", 3)),
    )
    del probe
    rows = [
        evaluate_checkpoint(Path(item), device, batches, total_samples)
        for item in args.checkpoint
    ]
    rows.sort(key=lambda r: (r["use_overhear"], r["lambda_rd"]))

    overhear = [r for r in rows if r["use_overhear"]]
    baseline = [r for r in rows if not r["use_overhear"]]

    lines: list[str] = []
    lines.append(f"Evaluation set: {rows[0]['samples']} scenes, "
                 f"{rows[0]['num_robots']} robots, {rows[0]['image_size']}px, seed {EVAL_SEED}")
    lines.append("Rate is the entropy-model estimate, not an arithmetic-coded bitstream.\n")
    lines.append("| lambda_rd | overhear | avg BPP | avg PSNR (dB) | BPP per robot | PSNR per robot |")
    lines.append("|---:|:---:|---:|---:|:--|:--|")
    for r in rows:
        bpp_r = " / ".join(f"{v:.4f}" for v in r["bpp_per_robot"])
        psnr_r = " / ".join(f"{v:.2f}" for v in r["psnr_per_robot"])
        lines.append(
            f"| {r['lambda_rd']:.0f} | {'ON' if r['use_overhear'] else 'OFF'} | "
            f"{r['bpp']:.4f} | {r['psnr']:.2f} | {bpp_r} | {psnr_r} |"
        )

    pairs = []
    lines.append("\n| lambda_rd | BPP OFF | BPP ON | PSNR OFF | PSNR ON | G_R rate saving |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for b in baseline:
        match = next((o for o in overhear if o["lambda_rd"] == b["lambda_rd"]), None)
        if match is None:
            continue
        gain = (b["bpp"] - match["bpp"]) / b["bpp"] * 100.0
        pairs.append({"lambda_rd": b["lambda_rd"], "gain_percent": gain,
                      "delta_psnr": match["psnr"] - b["psnr"]})
        lines.append(
            f"| {b['lambda_rd']:.0f} | {b['bpp']:.4f} | {match['bpp']:.4f} | "
            f"{b['psnr']:.2f} | {match['psnr']:.2f} | {gain:+.1f}% |"
        )

    bd = bd_rate(
        [(r["bpp"], r["psnr"]) for r in baseline],
        [(r["bpp"], r["psnr"]) for r in overhear],
    )
    if bd is not None:
        lines.append(f"\nBD-rate (overhear vs no-overhear): {bd:+.2f}%  "
                     "(negative = same quality at lower rate)")

    report = {"points": rows, "same_lambda_pairs": pairs, "bd_rate_percent": bd}
    print("\n".join(lines))
    if args.output_json is not None:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
