"""Module containing a method to apply a spatial shift to an image in Fourier domain."""
import numpy as np
import torch


class Shift(torch.nn.Module):
    """class containing method to apply a spatial shift to an image in Fourier domain.

    Written by H. Gupta and Y.S.G. Nashed.

    Parameters
    ----------
    config: class
        Class containing parameters of the shift

    """

    def __init__(self, config):
        super(Shift, self).__init__()
        """Initialize image grid and frequency value."""

        self.config = config
        self.frequency = 1.0 / (self.config.side_len * self.config.pixel_size)

        n_half_len = float(self.config.side_len // 2)
        ax = torch.arange(-n_half_len, n_half_len + self.config.side_len % 2)
        ax = torch.flip(ax, dims=[0])
        mx, my = torch.meshgrid(ax, ax)

        self.register_buffer("mx", mx.clone())
        self.register_buffer("my", my.clone())

    def phase_shift(self, x_fourier, t_x, t_y):
        """Output modulated (fourier equivalent of shifting in primal domain).

        Input images given in batch wise format.
        The modulation depends on t_x, t_y.

        Parameters
        ----------
        x_fourier : torch.Tensor
            batch of input images (Bx1xSizexSize) in Fourier domain
        t_x: torch.Tensor
            batch of shifts (B,) along horizontal axis in Angstrom (A)
        t_y: torch.Tensor
            batch of shifts (B,) along vertical axis in Angstrom (A)

        Returns
        -------
        output: torch.Tensor
            batch of modulated fourier images (Bx1xSizexSize) given by
            output(f_1,f_2)=e^{-2*pi*j*[f_1,f_2]*[t_x, t_y] }*input(f_1,f_2)
        """
        t_y = t_y[:, None, None, None]
        t_x = t_x[:, None, None, None]

        modulation = torch.exp(
            -2 * np.pi * 1j * self.frequency * (self.mx * t_y + self.my * t_x)
        )

        return x_fourier * modulation

    def forward(self, x_fourier, shift_params={}):
        """Output modulated input images.

        Fourier equivalent of shifting in primal domain. The modulation
        depends on t_x, t_y.

        Parameters
        ----------
        x_fourier : torch.Tensor
            batch of input images (batch_sizex1xSizexSize) in Fourier domain
        shift_params: dict of type str to {torch.Tensor}
            dictionary containing
            'shift_x': torch.Tensor (batch_size,)
                batch of shifts along horizontal axis in Angstrom (A)
            'shift_y': torch.Tensor (batch_size,)
                batch of shifts along vertical axis in Angstrom (A)

        Returns
        -------
            output: torch.Tensor
                batch of modulated fourier images (batch_sizex1xSizexSize)
                if shift_params is not empty else input is outputted
        """
        if shift_params:
            x_fourier = self.phase_shift(
                x_fourier, shift_params["shift_x"], shift_params["shift_y"]
            )

        return x_fourier
