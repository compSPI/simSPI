

import pytest
from simSPI import tem_distribution_utils
import torch



def test_constructor():
    """Test if distribution object is made."""

    generator = tem_distribution_utils.DistributionGenerator('uniform',(0,10))
    assert generator.distribution is not None

    generator = tem_distribution_utils.DistributionGenerator('gaussian', (0, 1))
    assert generator.distribution is not None

    with pytest.raises(NotImplementedError):
        tem_distribution_utils.DistributionGenerator('invalid_value', (0, 1))


def test_draw_samples_1d():
    """Test if samples of expected length are returned."""

    generator = tem_distribution_utils.DistributionGenerator('uniform', (0, 10))

    samples = generator.draw_samples_1d(10)
    assert type(samples) == torch.Tensor
    assert samples.shape == torch.Size([10])

