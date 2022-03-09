"""Utility file for generating and sampling distributions."""
import torch.distributions as dist
from torch import zeros


class DistributionGenerator:
    """Class to generate parameters using the specified distribution.

    Parameters
    ----------
    distribution_type: str
      String associated with a valid distribution.
    distribution_params: arr
      List of parameters to initialize distribution of interest.
    """

    def __init__(self, distribution_type, distribution_params):
        distribution_no_params = {
            "gaussian": dist.normal.Normal,
            "uniform": dist.uniform.Uniform,
        }

        if distribution_type == "gaussian":
            loc, scale = distribution_params
            distribution = distribution_no_params["gaussian"](loc, scale)

        elif distribution_type == "uniform":
            low, high = distribution_params
            distribution = distribution_no_params["uniform"](low, high)
        else:
            raise NotImplementedError(
                f"Distribution type '{distribution_type}' " f"has not been implemented!"
            )

        self.distribution = distribution

    def draw_samples_1d(self, n_samples):
        """Draw n samples from a distribution.

        Parameters
        ----------
        n_samples: int
          Number of samples to be drawn from distribution
        Returns
        -------
        samples_1d: tensor
          PyTorch tensor (array) containing n_samples samples from distribution.
        """
        samples_1d = zeros(n_samples)
        for idx in range(n_samples):
            samples_1d[idx] = self.distribution.sample()
        return samples_1d
