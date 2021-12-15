"""Module to corrupt the projection with noise."""
import torch


class Noise(torch.nn.Module):
    """Class to corrupt the projection with noise.

    Written by J.N. Martel and H. Gupta.

    Parameters
    ----------
    config: class
        contains parameters of the noise distribution

    """

    def __init__(self, config):

        super(Noise, self).__init__()
        self.noise_sigma = config.noise_sigma

    def forward(self, proj):
        """Add noise to projections.

        Currently, only supports additive white gaussian noise.

        Parameters
        ----------
        proj: torch.Tensor
            input projection of shape (batch_size,1,side_len,side_len)

        Returns
        -------
        out: torch.Tensor
            noisy projection of shape (batch_size,1,side_len,side_len)
        """
        out = proj + self.noise_sigma * torch.randn_like(proj)
        return out
