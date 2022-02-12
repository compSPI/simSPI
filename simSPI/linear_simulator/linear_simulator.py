"""Contain simulator and its component classes."""

import os

import mrcfile
import torch
from compSPI.transforms import fourier_to_primal_2D, primal_to_fourier_2D

from simSPI.linear_simulator.ctf import CTF
from simSPI.linear_simulator.noise_utils import Noise
from simSPI.linear_simulator.projector import Projector
from simSPI.linear_simulator.shift_utils import Shift
from simSPI.linear_simulator.volume_utils import init_cube



class LinearSimulator(torch.nn.Module):
    """Class to generate data using linear forward model.

    Parameters
    ----------
    config: class
        Class containing parameters of the simulator

    """

    def __init__(self, config):
        super(LinearSimulator, self).__init__()

        self.config = config
        self.projector = Projector(config)
        self.init_volume()
        self.ctf = CTF(config)
        self.shift = Shift(config)
        self.noise = Noise(config)

    def forward(self, rot_params, ctf_params, shift_params):
        """Create cryoEM measurements using input parameters.

        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection chunk
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF) parameters
             for a projection chunk
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection chunk

        Returns
        -------
        projection.real : torch.Tensor
            Tensor ([chunks,1,sidelen,sidelen]) contains cryoEM measurement
        """
        projection = self.projector(rot_params)
        f_projection = primal_to_fourier_2D(projection)
        f_projection = self.ctf(f_projection, ctf_params)
        f_projection = self.shift(f_projection, shift_params)
        projection = fourier_to_primal_2D(f_projection)
        projection = self.noise(projection)

        return projection.real

    def init_volume(self):
        """Initialize the volume inside the projector.

        Initializes the mrcfile whose path is given in config.input_volume_path.
        If the path is not given or doesn't exist then the volume
        is initialized with a cube.
        """
        if (
            self.config.input_volume_path == ""
            or os.path.exists(os.path.join(os.getcwd(), self.config.input_volume_path))
            is False
        ):

            print(
                "No input volume specified or the path doesn't exist. "
                "Using cube as the default volume."
            )
            volume = init_cube(self.config.side_len)
        else:
            with mrcfile.open(self.config.input_volume_path, "r") as m:
                volume = torch.from_numpy(m.data.copy()).to(self.projector.vol.device)

        self.projector.vol = volume
