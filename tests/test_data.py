import torch

from overhear_uplink.data import SyntheticMultiRobotDataset


def test_synthetic_dataset_is_deterministic_and_correlated() -> None:
    dataset = SyntheticMultiRobotDataset(
        samples=2, num_robots=3, image_size=64, seed=10
    )
    first = dataset[0]
    repeated = dataset[0]

    assert first.shape == (3, 3, 64, 64)
    assert torch.equal(first, repeated)
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
    assert not torch.equal(first[0], first[1])

