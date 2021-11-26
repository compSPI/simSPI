import os

import mrcfile
import numpy as np
import torch
import torch.fft
from pytorch3d.transforms import Rotate

from .utils import fourier_to_primal_2D, init_cube, primal_to_fourier_2D


class simulator(torch.nn.Module):
    def __init__(self, config, initial_volume=None):
        super(simulator, self).__init__()
        """module that generates data using weak phase approximation (liner forward model) with a given config setting.
            projector->ctf->shift->noise
        Parameters
        __________
            config: class
                Class containing parameters of the simulator
        """
        self.config = config
        self.Projector = Projector(config)  # class for tomographic projector
        self.init_volume()  # changes the volume inside the projector
        self.CTF = CTF(config)  # class for ctf
        self.Shift = Shift(config)  # class for shifts
        self.Noise = Noise(config)  # class for noise

    def forward(self, rot_params, ctf_params, shift_params):
        """Creates cryoEM measurements using input parameters
        Parameters
        __________
            rot_params: dict of type str to {tensor}
                Dictionary of rotation parameters for a projection chunk
            ctf_params: dict of type str to {tensor}
                Dictionary of Contrast Transfer Function (CTF) parameters for a projection chunk
            shift_params: dict of type str to {tensor}
                Dictionary of shift parameters for a projection chunk

        Returns
        _______
            projection.real : tensor
                Tensor ([chunks,1,sidelen,sidelen]) contains cryoEM measurement

        """
        projection = self.Projector(rot_params)
        f_projection = primal_to_fourier_2D(projection)
        f_projection = self.CTF(f_projection, ctf_params)
        f_projection = self.Shift(f_projection, shift_params)
        projection = fourier_to_primal_2D(f_projection)
        projection = self.Noise(projection)

        return projection.real

    def init_volume(self):
        """Initializes the volume inside the projector with the
        mrcfile whose path is given in config.input_volume_path
        if the path is not given or doesn't exist then the volume is intialized with a cube
        """
        if (
            self.config.input_volume_path == ""
            or os.path.exists(os.path.join(os.getcwd(), self.config.input_volume_path))
            == False
        ):

            print(
                "No input volume specified or the path doesn't exist. Using cube as the default volume."
            )
            volume = init_cube(self.config.sidelen)
        else:
            with mrcfile.open(self.config.input_volume_path, "r") as m:
                volume = torch.from_numpy(m.data.copy()).to(self.Projector.vol.device)

        self.Projector.vol = volume


class Projector(torch.nn.Module):
    def __init__(self, config):
        super(Projector, self).__init__()
        """modules that generates tomographic projection
        Parameters
        ----------
        config: class
            Class containing parameters of the Projector
        """
        self.config = config
        self.vol = torch.rand([self.config.sidelen] * 3, dtype=torch.float32)
        lincoords = torch.linspace(
            -1.0, 1.0, self.config.sidelen
        )  # assume square volume
        [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords])
        coords = torch.stack([Y, X, Z], dim=-1)
        self.register_buffer("vol_coords", coords.reshape(-1, 3))

    def forward(self, rot_params, proj_axis=-1):
        """Outputs the tomographic projection of the volume by first
        rotating it using and then summing it along an axis
        Parameters
        ----------
            rot_params: dict of type str to {tensor}
                contains rotation matrix "rotmat" (chunks x 3 x 3) that is used to rotate the volume
            proj_axis: int
                index along which summation is done of the rotated volume
        Returns
        -------
            projection: tensor
                Tensor containing tomographic projection (chunks x 1 x sidelen x sidelen)
        """
        rotmat = rot_params["rotmat"]
        batch_sz = rotmat.shape[0]
        t = Rotate(rotmat, device=self.vol_coords.device)
        rot_vol_coords = t.transform_points(self.vol_coords.repeat(batch_sz, 1, 1))

        rot_vol = torch.nn.functional.grid_sample(
            self.vol.repeat(batch_sz, 1, 1, 1, 1),  # = (Batch, C,D,H,W)
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
        )
        projection = torch.sum(
            rot_vol.reshape(
                batch_sz, self.config.sidelen, self.config.sidelen, self.config.sidelen
            ),
            dim=proj_axis,
        )
        projection = projection[
            :, None, :, :
        ]  # add a dummy channel (for consistency w/ img fmt)
        # --> B,C,H,W
        return projection


class CTF(torch.nn.Module):
    def __init__(self, config):
        super(CTF, self).__init__()
        """module that multiplies ctf to fourier transform of the projection
        Parameters
        ----------
            config: class
                Class containing parameters of the CTF
        """

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
        """outputs the wavelength of the electron beam
        Returns
        ------
            wavelength: float
                wavelength of the electron beam
        """
        wavelength = 12.2639 / np.sqrt(
            self.config.kV * 1e3 + 0.97845 * self.config.kV ** 2
        )
        return wavelength

    def get_ctf(self, ctf_params):
        """outputs the CTF for a given params
        Parameters
        ----------
            ctf_params: dict of type str to {tensor}
                contains "defocusU", "defocusV", "defocusAngle", all of shape (chunks,1,1,1)
        Returns
        -------
            hFourier: torch.Tensor
                ctf of shape (chunks,1,ctf_size,ctf_size)
        """
        defocusU = ctf_params["defocusU"]
        defocusV = ctf_params["defocusV"]
        angleAstigmatism = ctf_params["defocusAngle"]
        elliptical = (
            defocusV * self.r2
            + (defocusU - defocusV)
            * self.r2
            * torch.cos(self.angleFrequency - angleAstigmatism) ** 2
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

        if self.config.bfactor == 0:
            decay = (
                np.sqrt(-np.log(self.config.valueNyquist))
                * 2.0
                * self.config.pixel_size
            )
            envelope = torch.exp(-self.frequency ** 2 * decay ** 2 * self.r2)
        else:
            envelope = torch.exp(
                -self.frequency ** 2 * self.config.bfactor / 4.0 * self.r2
            )

        hFourier *= envelope
        return hFourier

    def forward(self, x_fourier, ctf_params={}):
        """Multiplies the CTF and projection in fourier domain if ctf_params is not empty,

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


class Shift(torch.nn.Module):
    def __init__(self, config):
        super(Shift, self).__init__()
        """
           A class containing method to shift an image in Fourier domain.
           ...
           Parameters
           ----------
           config: class
                Class containing parameters of the shift
        """
        self.config = config
        self.frequency = 1.0 / (self.config.sidelen * self.config.pixel_size)

        n2 = float(self.config.sidelen // 2)
        ax = torch.arange(-n2, n2 + self.config.sidelen % 2)
        ax = torch.flip(ax, dims=[0])
        mx, my = torch.meshgrid(ax, ax)

        # shape SizexSize
        self.register_buffer("mx", mx.clone())
        self.register_buffer("my", my.clone())

    def modulate(self, x_fourier, t_x, t_y):
        """
        outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
        The modulation depends on t_x, t_y.
        Parameters
        ----------
            x_fourier : torch.Tensor
                batch of input images (Bx1xSizexSize) in Fourier domain
            t_x: torch.Tensor
                batch of shifts (B,) along horizontal axis
            t_y: torch.Tensor
                batch of shifts (B,) along vertical axis
        Returns
        -------
            output: torch.Tensor
                batch of modulated fourier images (Bx1xSizexSize) given by
                output(f_1,f_2)=e^{-2*pi*j*[f_1,f_2]*[t_x, t_y] }*input(f_1,f_2)
        """

        t_y = t_y[:, None, None, None]  # [B,1,1,1]
        t_x = t_x[:, None, None, None]  # [B,1,1,1]

        modulation = torch.exp(
            -2 * np.pi * 1j * self.frequency * (self.mx * t_y + self.my * t_x)
        )  # [B,1,Size,Size]

        return x_fourier * modulation  # [B,1,Size,Size]*[B,1,Size,Size]

    def forward(self, x_fourier, shift_params={}):
        """
        outputs modulated (fourier equivalent of shifting in primal domain) input images given in batch wise format.
        The modulation depends on t_x, t_y.
        Parameters
        ----------
            x_fourier : torch.Tensor
                batch of input images (Bx1xSizexSize) in Fourier domain
            shift_params: dict of type str to {torch.Tensor}
                dictionary containing
                'shiftX': torch.Tensor (B,)
                    batch of shifts along horizontal axis
                'shiftY': torch.Tensor (B,)
                    batch of shifts along vertical axis
        Returns
        -------
            output: torch.Tensor
                batch of modulated fourier images (Bx1xSizexSize) if shift_params is not empty else input is outputted
        """
        if shift_params:
            x_fourier = self.modulate(
                x_fourier, shift_params["shiftX"], shift_params["shiftY"]
            )  # [B,1,Size,Size]

        return x_fourier


class Noise(torch.nn.Module):
    def __init__(self, config):
        """corrupts the projection with noise
        Parameters
        ----------
            config: class
                contains parameters of the noise distribution
        """
        super(Noise, self).__init__()
        self.noise_sigma = config.noise_sigma

    def forward(self, proj):
        """adds noise to proj. Currently, only supports additive white gaussian noise.
        Parameters
        ---------
            proj: torch.Tensor
                input projection of shape B,1,sidelen,sidelen
        """
        return proj + self.noise_sigma * torch.randn_like(proj)
