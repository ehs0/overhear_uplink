import torch

from overhear_uplink.config import ModelConfig
from overhear_uplink.losses import RateDistortionLoss
from overhear_uplink.model import OverhearUplinkCodec


def small_model() -> OverhearUplinkCodec:
    return OverhearUplinkCodec(
        ModelConfig(
            channels=16,
            latent_channels=20,
            hyper_channels=12,
            attention_heads=4,
            attention_grid=4,
            max_history=2,
        )
    )


def test_forward_and_rate_distortion_loss() -> None:
    torch.manual_seed(0)
    model = small_model().train()
    images = torch.rand(2, 3, 3, 64, 64)
    output = model(images)
    metrics = RateDistortionLoss(lambda_rd=10.0)(output, images)

    assert output["x_hat"].shape == images.shape
    assert output["predictions"].shape == images.shape
    assert output["bits_y"].shape == (2, 3)
    assert output["bits_z"].shape == (2, 3)
    assert torch.isfinite(metrics["loss"])
    assert float(metrics["bpp"]) > 0.0

    metrics["loss"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_packet_round_trip_keeps_shared_state_identical() -> None:
    torch.manual_seed(1)
    model = small_model().eval()
    images = torch.rand(1, 3, 3, 64, 64)
    encoder_history: list[torch.Tensor] = []
    decoder_history: list[torch.Tensor] = []

    for robot_index in range(images.shape[1]):
        packet, encoder_reconstruction = model.encode_robot(
            images[:, robot_index], encoder_history, robot_index
        )
        decoder_reconstruction = model.decode_robot(packet, decoder_history)
        torch.testing.assert_close(
            encoder_reconstruction,
            decoder_reconstruction,
            rtol=0.0,
            atol=1e-6,
        )
        assert torch.all(packet.estimated_bits > 0)
        encoder_history.append(encoder_reconstruction)
        decoder_history.append(decoder_reconstruction)


def test_invalid_spatial_size_is_rejected() -> None:
    model = small_model()
    images = torch.rand(1, 2, 3, 96, 64)
    try:
        model(images)
    except ValueError as error:
        assert "divisible by 64" in str(error)
    else:
        raise AssertionError("invalid image size should fail")

