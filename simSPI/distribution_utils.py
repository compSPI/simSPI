"""Utility file for generating and sampling distributions."""
import torch.distributions as dist
from torch import zeros


def make_distribution(distribution_params, distribution_type):
    """Choose and generate distribution from passed type and parameters.

    Parameters
    ----------
    distribution_params: arr
      List of parameters to initialize distribution of interest.
    distribution_type: str
      String associated with a valid distribution.

    Returns
    -------
    distribution: Object
      Unsampled PyTorch distribution seeded by input parameters.
    """
    distribution_no_params = {
        "gaussian": dist.normal.Normal,
        "uniform": dist.uniform.Uniform,
    }

    distribution = None

    if distribution_type == "gaussian":
        loc, scale = distribution_params
        distribution = distribution_no_params["gaussian"](loc, scale)

    elif distribution_type == "uniform":
        low, high = distribution_params
        distribution = distribution_no_params["uniform"](low, high)

    return distribution


def draw_samples_distribution_1d(distribution, n_samples):
    """Draw n samples from a distribution.

    Parameters
    ----------
    distribution: Object
      Unsampled PyTorch distribution.
    n_samples: int
      Number of samples to be drawn from distribution

    Returns
    -------
    samples_1d: tensor
      PyTorch tensor (array) containing n_samples samples from distribution.
    """
    samples_1d = zeros(n_samples)

    for idx in range(n_samples):
        samples_1d[idx] = distribution.sample()

    return samples_1d
