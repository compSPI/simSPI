"""Tests for tem_distribution."""
import pytest
import torch
import numpy as np
from simSPI import tem_distribution


def test_constructor():
    """Test if distribution object is made."""
    generator = tem_distribution.DistributionGenerator("uniform", (0, 10))
    assert generator.distribution is not None

    generator = tem_distribution.DistributionGenerator("gaussian", (0, 1))
    assert generator.distribution is not None

    with pytest.raises(NotImplementedError):
        tem_distribution.DistributionGenerator("invalid_value", (0, 1))


def test_draw_samples_uniform():
    """Test if samples of expected length are returned."""

    upper, lower = (10, 0)
    size = 100
    generator = tem_distribution.DistributionGenerator("uniform", [lower, upper])

    samples = generator.draw_samples_1d(size)
    assert type(samples) is torch.Tensor
    assert samples.shape == torch.Size([size])
    assert samples[(samples < lower) & (samples > upper)].shape == torch.Size([0])


def test_draw_samples_gaussian():
    """Test if samples of expected length are returned."""
    loc, scale = (5,10)
    size = 10000
    generator = tem_distribution.DistributionGenerator("gaussian", [loc,scale])
    samples = generator.draw_samples_1d(size)

    mean_error = 3.69 * scale/np.sqrt(size)
    std_error = 0.1

    assert type(samples) is torch.Tensor
    assert samples.shape == torch.Size([size])
    assert np.abs(loc - torch.mean(samples).item()) < mean_error
    assert np.abs(scale - torch.std(samples).item()) < std_error
