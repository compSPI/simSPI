"""Module that multiplies ctf to fourier transform of the projection."""
import numpy as np
import torch


class CTF(torch.nn.Module):
    """Class that multiplies ctf to fourier transform of the projection.

    Written by H. Gupta and J.N. Martel.

    Parameters
    ----------
    config: class
        Class containing parameters of the CTF

    """

    def __init__(self, config):
        """Initialize image grid, wavelength, and frequency step per pixel."""
        super(CTF, self).__init__()

        self.config = config
        self.wavelength = self._get_ewavelength()
        self.frequency_step = 1.0 / (self.config.ctf_size * self.config.pixel_size)
        n_half_len = float(self.config.ctf_size // 2)
        ax = torch.arange(-n_half_len, n_half_len + self.config.ctf_size % 2)
        mx, my = torch.meshgrid(ax, ax)
        self.register_buffer("r2", mx ** 2 + my ** 2)
        self.register_buffer("frequency", torch.sqrt(self.r2) * self.frequency_step)
        self.register_buffer("angleFrequency", torch.atan2(my, mx))

    def _get_ewavelength(self):
        """Output the wavelength of the electron beam.

        Returns
        -------
        wavelength: float
            wavelength of the electron beam in A.
        """
        wavelength = 12.2639 / np.sqrt(
            self.config.kv * 1e3 + 0.97845 * self.config.kv ** 2
        )
        return wavelength

    def get_ctf(self, ctf_params):
        """Output the CTF for a given params.

        Continuous-domain CTF
        ---------------------
            Creates CTF using the formula (obtained from [1, Section 2.II.B]),

                CTF(f)= E(f).((1-ac^2)^0.5 sin(R(f))+ ac. cos(R(f))), where,
                R(f)= pi * wavelength*(D(f)* 1e4 *||f||^2 -
                      0.25 * cs * 1e7 * wavelength^3 * ||f||^4),
                D(f)=0.5*(defocus_u+ defocus_v) +
                     0.5*(defocus_u-defocus_v)*cos(2*(alpha_f-defocus_angle)), and,
                E(f)= exp^*(-0.25* b_factor*||f||^2).

            Expression D(f) can actually be simplified as
                D(f)=defocus_v+(defocus_u-defocus_v)*cos(alpha_f-defocus_angle)^2.

            Note that the above formulation has been slightly modified from [1] in
            order to make the CTF follow the sign and astigmatism convention of
            Relion [2]. The CTF from Relion is accessible via relion_project function.

        Constants and variables
        -----------------------
            Here,
                f:  frequency vector;
                alpha_f: phase of the vector f in polar coordinates;

                ac:  amplitude contrast (comes from self.config);
                wavelength: electron beam wavelength in A (comes from self.config);
                cs: spherical aberration constant in mm (comes from self.config);
                b_factor: decay for the envelope (comes from self.config)

                defocus_u: horizontal defocus in um (comes from ctf_params);
                defocus_v: vertical defocus in um (comes from ctf_params);
                defocus_angle: astigmatism angle in radians (comes from ctf_params).

            For a given dataset alpha_f, ac, wavelength, and cs are constants,
            whereas, other variables can vary per projection.

        Discretization
        --------------
            The code below simplifies and discretizes the above continuous-domain
            expression by sampling CTF value at config.ctf_size//2 number of
            equidistant frequency samples between 0 to Nyquist frequency.

            Specifically, in a grid of pixel location going from (0,0) to (N-1, N-1)
            for any pixel location (m,n):

            |f|= ((m-c)^2+(n-c)^2)**0.5 *frequency_step,and
            alpha_f=arctan(n-c,m-c)
            where frequency_step= Nyquist_frequency/(N/2)=1/(pixel_size*N)

        References
        ----------
            [1]: Frank, Joachim. Three-dimensional electron microscopy of
                macromolecular assemblies: visualization of biological molecules
                in their native state. Oxford university press, 2006.
            [2]: Scheres, Sjors HW. "RELION: implementation of a Bayesian approach to
                cryo-EM structure determination." Journal of structural biology,
                180, no. 3 (2012): 519-530.

        Remarks
        -------
            In the code below, the envelope function E(f), when b_factor is unknown
            (assumed to be 0 in that case) uses value_nyquist given by the user. This
            corresponds to the value of envelope function at nyquist frequency and
            hence, is directly related to CTF b_factor. Because of this simple
            definition of value_nyquist it is a more intuitive and easier input
            for the user to provide than b_factor.

        Parameters
        ----------
        ctf_params: dict of type str to torch.Tensor
            contains defocus parameters with keys
                defocus_u: str to torch.Tensor
                    horizontal defocus in um with shape: (batch_size,1,1,1)
                defocus_v: str to torch.Tensor
                    vertical defocus in um with shape: (batch_size,1,1,1)
                defocus_angle: str to torch.Tensor
                    astigmatism angle in radians with shape: (batch_size,1,1,1)

        Returns
        -------
        h_fourier: torch.Tensor
            ctf of shape (batch_size,1,ctf_size,ctf_size)
        """
        defocus_u = ctf_params["defocus_u"]
        defocus_v = ctf_params["defocus_v"]
        angle_astigmatism = ctf_params["defocus_angle"]
        elliptical = (
            defocus_v
            + (defocus_u - defocus_v)
            * torch.cos(self.angleFrequency - angle_astigmatism) ** 2
        )
        defocus_contribution = (
            np.pi * self.wavelength * 1e4 * elliptical * self.frequency ** 2
        )
        abberation_contribution = (
            -np.pi
            / 2.0
            * self.config.cs
            * (self.wavelength ** 3)
            * 1e7
            * self.frequency ** 4
        )

        argument = abberation_contribution + defocus_contribution

        h_fourier = (1 - self.config.amplitude_contrast ** 2) ** 0.5 * torch.sin(
            argument
        ) + self.config.amplitude_contrast * torch.cos(argument)

        if self.config.b_factor == 0:
            decay = (
                np.sqrt(-np.log(self.config.value_nyquist))
                * 2.0
                * self.config.pixel_size
            )
            envelope = torch.exp(-self.frequency ** 2 * decay ** 2)
        else:
            envelope = torch.exp(-self.frequency ** 2 * self.config.b_factor / 4.0)

        h_fourier *= envelope
        return h_fourier

    def forward(self, x_fourier, ctf_params=None):
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
        if ctf_params is not None:
            h_fourier = self.get_ctf(ctf_params)

            x_fourier = x_fourier * h_fourier

        return x_fourier
