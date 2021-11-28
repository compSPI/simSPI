"""contain functions and classed for parameter generation."""
from abc import ABCMeta, abstractstaticmethod

import numpy as np
import starfile
import torch
import torch.fft
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    random_rotations,
)

from .starfile_utils import check_star_file, starfile_opticsparams
from .utils import U2T


def params_update(config):
    """Update attributes of config.

    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generator.

    Returns
    -------
    config: class
    """
    config.starfile_available = config.input_starfile_path != ""
    if config.starfile_available:
        config = starfile_opticsparams(config)
    config.ctf_size = config.sidelen  # TODO: make it adaptable
    print(
        f"Current CTF size has been configured to"
        f" be equal to the projection size = ({config.sidelen},{config.sidelen})"
    )
    return config


"""Module to instantiate param generator from a factory of choices."""


class ParamsFactory:
    """Class to instantiate param generator from a factory of choices."""

    def get_params_generator(config):
        """Return the param generator.

        If the starfile is available then chooses it otherwise
        chooses distributional param generator.

        Parameters
        ----------
        config: class

        Returns
        -------
        params_generator: class
        """
        if config.starfile_available:
            return starfile_params(config)
        else:
            return distributional_params(config)


class Iparams(metaclass=ABCMeta):
    """Abstract class for the params generator factory with get_params method."""

    @abstractstaticmethod
    def get_params(self):
        """Get the parameters."""


"""Module to generate parameters using the input star file."""


class starfile_params(Iparams):
    """Class to generate parameters using the input star file.

    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generator.

    """

    def __init__(self, config):

        self.counter = 0
        self.invert_hand = False
        self.df = starfile.read(config.input_starfile_path)
        check_star_file(config.input_starfile_path)
        self.config = config
        print("Reading parameters from the input starfile.")

    def get_params(self):
        """Get the parameters from the starfile.

        Returns
        -------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection chunk
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF) parameters
            for a projection chunk
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection chunk
        """
        self.particle = self.df["particles"].iloc[
            self.counter : self.counter + self.config.chunks
        ]
        self.counter += self.config.chunks
        return self.get_rotmat(), self.get_ctf_params(), self.get_shift_params()

    def get_rotmat(self):
        """Get the parameters for the rotation of the projections from starfile.

        Returns
        -------
        rotmat_params: dict of type str to {tensor}
            dictionary containing:
            "rotmat": torch.Tensor
                Tensor of size (chunks,3,3) that containing the rotation matrix for
                each projection
            "relion_AnglePsi": torch.Tensor
                Tensor of size (chunks,) that contains the Psi angle (in degrees)
                 for ZYZ rotation
            "relion_AngleTilt": torch.Tensor
                Tensor of size (chunks,) that contains the Tilt angle (in degrees)
                 for ZYZ rotation
            "relion_AngleRot": torch.Tensor
                Tensor of size (chunks,) that contains the Rot angle (in degrees)
                for ZYZ rotation
        """
        AnglePsi = np.array(self.particle["rlnAnglePsi"], ndmin=2).squeeze()
        AngleTilt = np.array(self.particle["rlnAngleTilt"], ndmin=2).squeeze()
        AngleRot = np.array(self.particle["rlnAngleRot"], ndmin=2).squeeze()
        relion_euler_np = np.radians(
            np.stack(
                [
                    -AnglePsi,  # convert Relion to our convention
                    AngleTilt * (-1 if self.config.relion_invert_hand else 1),
                    # convert Relion to our convention + invert hand
                    AngleRot,
                ]
            )
        )  # convert Relion to our convention
        relion_euler = torch.from_numpy(relion_euler_np).t()  # Bx3
        rotmat = euler_angles_to_matrix(relion_euler, convention="ZYZ")

        return {
            "rotmat": rotmat,
            "relion_AnglePsi": AnglePsi,
            "relion_AngleTilt": AngleTilt,
            "relion_AngleRot": AngleRot,
        }

    def get_ctf_params(self):
        """Get the parameters for the CTF of the particle from the starfile.

        If config.ctf is True else returns {}

        Returns
        -------
        ctf_params: dict of type str to {tensor}
            dictionary containing:
            "defocusU": torch.Tensor
                Tensor of size (chunks,1,1,1) that contains the major
                defocus value in microns
            "defocusV": torch.Tensor
                Tensor of size (chunks,1,1,1) that contains the minor
                defocus value in microns
            "defocusAngle": torch.Tensor
                Tensor of size (chunks,1,1,1) that contains the major
                defocus value in microns
        """
        params = {}
        if self.config.ctf:
            defocusU = (
                torch.from_numpy(np.array(self.particle["rlnDefocusU"] / 1e4))
                .float()
                .squeeze()[:, None, None, None]
            )
            defocusV = (
                torch.from_numpy(np.array(self.particle["rlnDefocusV"] / 1e4))
                .float()
                .squeeze()[:, None, None, None]
            )
            defocusAngle = (
                torch.from_numpy(np.radians(np.array(self.particle["rlnDefocusAngle"])))
                .float()
                .squeeze()[:, None, None, None]
            )
            params = {
                "defocusU": defocusU,
                "defocusV": defocusV,
                "defocusAngle": defocusAngle,
            }
        return params

    def get_shift_params(self):
        """Get the parameters for the shift of the particle from the starfile.

        If config.shift is True else returns {}

        Returns
        -------
            shift_params: dict of type str to {torch.Tensor}
            dictionary containing
            'shiftX': torch.Tensor (B,)
                batch of shifts along horizontal axis
            'shiftY': torch.Tensor (B,)
                batch of shifts along vertical axis
        """
        params = {}
        if self.config.shift:
            shiftX = torch.from_numpy(np.array(self.particle["rlnOriginXAngst"]))
            shiftY = torch.from_numpy(np.array(self.particle["rlnOriginYAngst"]))
            params = {"shiftX": shiftX, "shiftY": shiftY}
        return params


"""Module to generate parameters using the specified distribution."""


class distributional_params(Iparams):
    """Class to generate parameters using the specified distribution.

    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generator.

    """

    def __init__(self, config):
        self.config = config
        print("Parameters getting generated from specified distributions.")

    def get_params(self):
        """Get the rotation, ctf, and shift parameters."""
        return self.get_rotmat(), self.get_ctf_params(), self.get_shift_params()

    def get_rotmat(self):
        """Get the parameters for the rotation of the projection from a specified distribution.

        Returns
        -------
        rotmat_params: dict of type str to {tensor}
            dictionary containing:
            "rotmat": torch.Tensor
                Tensor of size (chunks,3,3) that containing the rotation matrix
                for each projection
            "relion_AnglePsi": torch.Tensor
                Tensor of size (chunks,) that contains the Psi angle (in degrees)
                for ZYZ rotation
            "relion_AngleTilt": torch.Tensor
                Tensor of size (chunks,) that contains the Tilt angle (in degrees)
                for ZYZ rotation
            "relion_AngleRot": torch.Tensor
                Tensor of size (chunks,) that contains the Rot angle (in degrees)
                for ZYZ rotation
        """
        if self.config.angle_distribution == "uniform":
            rotmat = random_rotations(self.config.chunks)
            euler = np.degrees(matrix_to_euler_angles(rotmat, convention="ZYZ").numpy())
            # the rot and psi below have been swapped and their sign have been changed
            # in order to follow the relion convention
            # (see starfile_params.get_rotmat())
            return {
                "rotmat": rotmat,
                "relion_AnglePsi": -euler[:, 0],
                "relion_AngleTilt": (-1 if self.config.relion_invert_hand else 1)
                * euler[:, 1],
                "relion_AngleRot": -euler[:, 2],
            }
        else:
            raise NotImplementedError(
                f"Angle distribution '{self.config.angle_distribution}' "
                f"has not been implemented!"
            )

    def get_ctf_params(self):
        """Get the parameters for the CTF of the particle from a distribution.

        If config.ctf is True else returns {}

        Returns
        -------
        ctf_params: dict of type str to {tensor}
            dictionary containing:
            "defocusU": torch.Tensor
                Tensor of size (chunks,1,1,1) that contains the major
                defocus value in microns
            "defocusV": torch.Tensor
                Tensor of size (chunks,1,1,1) that contains the minor
                defocus value in microns
            "defocusAngle": torch.Tensor
                Tensor of size (chunks,1,1,1) that contains the major
                defocus value in microns
        """
        if self.config.ctf:
            defocusU = (
                self.config.min_defocus
                + (self.config.max_defocus - self.config.min_defocus)
                * torch.zeros(self.config.chunks)[:, None, None, None].uniform_()
            )
            defocusV = defocusU.clone().detach()
            defocusAngle = torch.zeros_like(defocusU)
            return {
                "defocusU": defocusU,
                "defocusV": defocusV,
                "defocusAngle": defocusAngle,
            }
        else:
            return {}

    def get_shift_params(self):
        """Get the parameters for the shift of the particle from a distribution.

        If config.shift is True else returns {}.

        Returns
        -------
        shift_params: dict of type str to {torch.Tensor}
            dictionary containing
            'shiftX': torch.Tensor (B,)
                batch of shifts along horizontal axis
            'shiftY': torch.Tensor (B,)
                batch of shifts along vertical axis
        """
        if self.config.shift:
            if self.config.shift_distribution == "triangular":
                shiftNormalized = torch.Tensor(self.config.chunks, 2).uniform_()
                shifts = (
                    U2T(shiftNormalized)
                    * self.config.sidelen
                    * self.config.shift_variance
                    / 100.0
                ).float()
                params = {"shiftX": shifts[:, 0], "shiftY": shifts[:, 1]}
                return params
            else:
                print("here")
                raise NotImplementedError(
                    f"Shift distribution '{self.config.shift_distribution}' "
                    f"has not been implemented!"
                )

        else:
            return {}
