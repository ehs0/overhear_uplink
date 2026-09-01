# Overhear Uplink

This project provides a learned image codec for a scenario in which multiple robots observe the same scene from different viewpoints and transmit their images sequentially over an uplink. It assumes perfect overhearing between robots and identical decoding states at each robot and the server.

The core procedure is as follows:

1. The first robot transforms its complete image into a latent representation, as in a standard nonlinear transform codec.
2. Each subsequent robot uses the previously overheard and reconstructed images as context.
3. A cross-attention predictor produces a shared prediction of the current image. The transmitting robot and server can compute the same prediction independently.
4. The current robot applies a nonlinear transform only to the residual, `current image - shared prediction`.
5. A hyperprior and the overheard context define a conditional Gaussian entropy model for the residual latent representation.
6. The server reconstructs the residual, adds it to the shared prediction, and uses the resulting image as the next shared state that other robots can overhear.

```text
robot k image x_k ── subtract ──> residual ── g_a, Q ──> uplink symbols
                         ▲                                  │
previous reconstructions ── JCT-style cross attention      ▼
                         │                         inverse transform
                         └──────── prediction ────── add ──> x_hat_k
```

## Relationship to the Reference Papers

- [LDMIC](https://arxiv.org/abs/2301.09799): The implementation uses cross-attention context transfer to model correlations among views without explicit disparity estimation. Because this project considers sequential overhearing, it does not directly reproduce LDMIC's symmetric independent-encoder and joint-decoder architecture. Instead, both the robot and server condition on previously reconstructed images that they share.
- [Nonlinear Transform Coding](https://arxiv.org/abs/2007.03034): The implementation uses analysis and synthesis transforms, differentiable quantization, learned entropy models, and a rate-distortion objective.
- [NTSCC](https://arxiv.org/abs/2112.10961): The hyperprior predicts the information content of latent variables and guides end-to-end rate-distortion training. Under the requested ideal-channel assumption, the AWGN channel and neural channel encoder and decoder are replaced with an identity channel.

This project is not an exact reproduction of the results reported in the papers. It is a research baseline that adapts ideas from all three papers to the specified overhearing scenario.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

The complete pipeline can be tested with synthetic multi-view scenes without downloading an external dataset.

```bash
python -m overhear_uplink.demo --image-size 64 --num-robots 3
python -m overhear_uplink.train \
  --config configs/base.json \
  --output-dir runs/synthetic
python -m overhear_uplink.evaluate \
  --checkpoint runs/synthetic/best.pt \
  --synthetic-samples 32
```

The `demo` command verifies that the transmitting robot and the server produce identical reconstructed images from every uplink symbol packet.

## Real Dataset Layout

File names within each scene directory determine the robot transmission order. All views in a scene must have the same resolution, and the training crop size must be a multiple of 64.

```text
data/
  train/
    scene_0001/
      robot_00.png
      robot_01.png
      robot_02.png
    scene_0002/
      robot_00.png
      robot_01.png
      robot_02.png
  val/
    scene_0101/
      robot_00.png
      robot_01.png
      robot_02.png
```

```bash
python -m overhear_uplink.train \
  --config configs/base.json \
  --data-root /path/to/data \
  --output-dir runs/real
```

A `manifest.json` file can be placed under each split directory as an alternative to scene subdirectories.

```json
{
  "scenes": [
    {"id": "scene_0001", "views": ["images/a.png", "images/b.png"]}
  ]
}
```

Paths are resolved relative to the manifest file.

## Training Objective

The default loss over the complete robot sequence is:

```text
L = estimated_bpp + lambda_rd * MSE(x, x_hat)
    + prediction_weight * MSE(x[1:], prediction[1:])
```

`estimated_bpp` includes both the residual latent `y` and the hyperprior `z`. Gradients from later robots flow through earlier reconstructed states, allowing the model to learn error propagation across the sequence. Set `detach_history` to `true` in the configuration if memory usage is too high.

Setting `use_overhear` to `false` in `configs/base.json` produces an ablation baseline in which every robot is compressed independently. Train both configurations with the same data and `lambda_rd` value to compare per-robot BPP and PSNR.

## Scope and Limitations

- `RobotPacket` currently stores quantized latent symbols and the ideal bit count estimated by the entropy model. It does not include an arithmetic or rANS layer that produces a byte stream. Reported BPP values are therefore `-log2(p)` estimates from the learned probability model.
- The robots and server are assumed to use identical model parameters, packet order, and previously reconstructed images.
- Packet loss, inter-robot channel noise, uplink noise, and retransmission are not modeled.
- When views have little or no overlap, residual coding may provide little benefit. Comparison against the independent-coding ablation is essential in this case.

## Tests

```bash
pytest
```

The tests verify tensor shapes, finite rate-distortion loss values, and identical sequential packet reconstructions at the encoder and decoder.
