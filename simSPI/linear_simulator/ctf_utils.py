"""Module that multiplies ctf to fourier transform of the projection."""
import numpy as np
import torch
import torch.fft


class CTF(torch.nn.Module):
    """Class that multiplies ctf to fourier transform of the projection.

    Written by H. Gupta and J.N. Martel.

    Parameters
    ----------
    config: class
        Class containing parameters of the CTF

    """

    def __init__(self, config):
        super(CTF, self).__init__()

        self.config = config
        self.wavelength = self._get_ewavelength()
        self.frequency = 1.0 / (self.config.ctf_size * self.config.pixel_size)
        n2 = float(self.config.ctf_size // 2)
        ax = torch.arange(-n2, n2 + self.config.ctf_size % 2)
        mx, my = torch.meshgrid(ax, ax)
        self.register_buffer("r2", mx ** 2 + my ** 2)
        self.register_buffer("r", torch.sqrt(self.r2))
        self.register_buffer("angleFrequency", torch.atan2(my, mx))

    def _get_ewavelength(self):
        """Output the wavelength of the electron beam.

        Returns
        -------
        wavelength: float
            wavelength of the electron beam
        """
        wavelength = 12.2639 / np.sqrt(
            self.config.kv * 1e3 + 0.97845 * self.config.kv ** 2
        )
        return wavelength

    def get_ctf(self, ctf_params):
        """Output the CTF for a given params.

        Parameters
        ----------
        ctf_params: dict of type str to {tensor}
            contains "defocus_u", "defocus_v", "defocus_angle",
            all of shape (batch_size,1,1,1)

        Returns
        -------
        hFourier: torch.Tensor
            ctf of shape (batch_size,1,ctf_size,ctf_size)
        """
        defocus_u = ctf_params["defocus_u"]
        defocus_v = ctf_params["defocus_v"]
        angle_astigmatism = ctf_params["defocus_angle"]
        elliptical = (
            defocus_v * self.r2
            + (defocus_u - defocus_v)
            * self.r2
            * torch.cos(self.angleFrequency - angle_astigmatism) ** 2
        )
        defocusContribution = (
            np.pi * self.wavelength * 1e4 * elliptical * self.frequency ** 2
        )
        abberationContribution = (
            -np.pi
            / 2.0
            * self.config.cs
            * (self.wavelength ** 3)
            * 1e7
            * self.frequency ** 4
            * self.r2 ** 2
        )

        argument = abberationContribution + defocusContribution

        hFourier = (1 - self.config.amplitude_contrast ** 2) ** 0.5 * torch.sin(
            argument
        ) + self.config.amplitude_contrast * torch.cos(argument)

        if self.config.b_factor == 0:
            decay = (
                np.sqrt(-np.log(self.config.value_nyquist))
                * 2.0
                * self.config.pixel_size
            )
            envelope = torch.exp(-self.frequency ** 2 * decay ** 2 * self.r2)
        else:
            envelope = torch.exp(
                -self.frequency ** 2 * self.config.b_factor / 4.0 * self.r2
            )

        hFourier *= envelope
        return hFourier

    def forward(self, x_fourier, ctf_params={}):
        """Multiply the CTF and projection in fourier domain.

        If ctf_params is empty then skip multiplication.

        Parameters
        ----------
        x_fourier: torch.Tensor
            fourier transform of the projection (chunks,1,sidelen,sidelen)
        ctf_params: dict of type str to {torch.Tensor}
            contains parameters of the ctf

        Returns
        -------
        x_fourier: torch.Tensor
            modulated fourier transform of the projection (chunks,1,sidelen,sidelen)
        """
        if ctf_params:
            hFourier = self.get_ctf(ctf_params)

            x_fourier = x_fourier * hFourier

        return x_fourier
