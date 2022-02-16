"""contains functions and classes for parameter generation."""
from abc import ABCMeta, abstractstaticmethod

import numpy as np
import starfile
import torch
from compSPI.distributions import uniform_to_triangular
from ioSPI.starfile import check_star_file, starfile_opticsparams
from pytorch3d.transforms import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    random_rotations,
)


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
    config.ctf_size = config.side_len
    print(
        f"Current CTF size has been configured to"
        f" be equal to the projection size = ({config.side_len},{config.side_len})"
    )
    return config


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
            params_generator = StarfileParams(config)
        else:
            params_generator = DistributionalParams(config)
        return params_generator


class Iparams(metaclass=ABCMeta):
    """Abstract class for the params generator factory with get_params method."""

    @abstractstaticmethod
    def get_params(self):
        """Get the parameters."""


class StarfileParams(Iparams):
    """Class to generate parameters using the input star file.

    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generator.

    """

    def __init__(self, config):

        self.counter = 0
        check_star_file(config.input_starfile_path)
        self.df = starfile.read(config.input_starfile_path)
        self.config = config
        print("Reading parameters from the input starfile.")

    def get_params(self):
        """Get the parameters from the starfile.

        Returns
        -------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection batch
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF) parameters
            for a projection batch
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection batch
        """
        self.particle = self.df["particles"].iloc[
            self.counter : self.counter + self.config.batch_size
        ]
        self.counter += self.config.batch_size
        return self.get_rotmat(), self.get_ctf_params(), self.get_shift_params()

    def get_rotmat(self):
        """Get the parameters for the rotation of the projections from starfile.

        Returns
        -------
        rotmat_params: dict of type str to {tensor}
            dictionary containing:
            "rotmat": torch.Tensor
                Tensor of size (batch_size,3,3) that containing the rotation matrix for
                each projection
            "relion_angle_psi": torch.Tensor
                Tensor of size (batch_size,) that contains the Psi angle (in degrees)
                 for ZYZ rotation
            "relion_angle_tilt": torch.Tensor
                Tensor of size (batch_size,) that contains the Tilt angle (in degrees)
                 for ZYZ rotation
            "relion_angle_rot": torch.Tensor
                Tensor of size (batch_size,) that contains the Rot angle (in degrees)
                for ZYZ rotation
        """
        angle_psi = np.array(self.particle["rlnAnglePsi"], ndmin=2).squeeze()
        angle_tilt = np.array(self.particle["rlnAngleTilt"], ndmin=2).squeeze()
        angle_rot = np.array(self.particle["rlnAngleRot"], ndmin=2).squeeze()
        relion_euler_np = np.radians(
            np.stack(
                [
                    -angle_psi,  # convert Relion to our convention
                    angle_tilt * (-1 if self.config.relion_invert_hand else 1),
                    # convert Relion to our convention + invert hand
                    angle_rot,
                ]
            )
        )  # convert Relion to our convention
        relion_euler = torch.from_numpy(relion_euler_np).t()  # Bx3
        rotmat = euler_angles_to_matrix(relion_euler, convention="ZYZ")

        return {
            "rotmat": rotmat,
            "relion_angle_psi": angle_psi,
            "relion_angle_tilt": angle_tilt,
            "relion_angle_rot": angle_rot,
        }

    def get_ctf_params(self):
        """Get the parameters for the CTF of the particle from the starfile.

        If config.ctf is True else returns None

        Returns
        -------
        ctf_params: dict of type str to {tensor}
            dictionary containing:
            "defocus_u": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the major
                defocus value in microns
            "defocus_v": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the minor
                defocus value in microns
            "defocus_angle": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the astigmatism
                angle in radians
        """
        params = None
        if self.config.ctf:
            defocus_u = (
                torch.from_numpy(np.array(self.particle["rlnDefocusU"] / 1e4))
                .float()
                .squeeze()[:, None, None, None]
            )
            defocus_v = (
                torch.from_numpy(np.array(self.particle["rlnDefocusV"] / 1e4))
                .float()
                .squeeze()[:, None, None, None]
            )
            defocus_angle = (
                torch.from_numpy(np.radians(np.array(self.particle["rlnDefocusAngle"])))
                .float()
                .squeeze()[:, None, None, None]
            )
            params = {
                "defocus_u": defocus_u,
                "defocus_v": defocus_v,
                "defocus_angle": defocus_angle,
            }
        return params

    def get_shift_params(self):
        """Get the parameters for the shift of the particle from the starfile.

        If config.shift is True else returns None

        Returns
        -------
            shift_params: dict of type str to {torch.Tensor}
            dictionary containing
            'shift_x': torch.Tensor (batch_size,)
                batch of shifts along horizontal axis in Angstrom (A)
            'shift_y': torch.Tensor (batch_size,)
                batch of shifts along vertical axis in Angstrom (A)
        """
        params = None
        if self.config.shift:
            shift_x = torch.from_numpy(np.array(self.particle["rlnOriginXAngst"]))
            shift_y = torch.from_numpy(np.array(self.particle["rlnOriginYAngst"]))
            params = {"shift_x": shift_x, "shift_y": shift_y}
        return params


class DistributionalParams(Iparams):
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
        """Get the rotation, ctf, and shift parameters.

        Returns
        -------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection batch
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF) parameters
            for a projection batch
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection batch
        """
        return self.get_rotmat(), self.get_ctf_params(), self.get_shift_params()

    def get_rotmat(self):
        """Get the rotation matrix from a specified distribution.

        Returns
        -------
        rotmat_params: dict of type str to {tensor}
            dictionary containing:
            "rotmat": torch.Tensor
                Tensor of size (batch_size,3,3) that containing the rotation matrix
                for each projection
            "relion_angle_psi": torch.Tensor
                Tensor of size (batch_size,) that contains the Psi angle (in degrees)
                for ZYZ rotation
            "relion_angle_tilt": torch.Tensor
                Tensor of size (batch_size,) that contains the Tilt angle (in degrees)
                for ZYZ rotation
            "relion_angle_rot": torch.Tensor
                Tensor of size (batch_size,) that contains the Rot angle (in degrees)
                for ZYZ rotation
        """
        if self.config.angle_distribution == "uniform":
            rotmat = random_rotations(self.config.batch_size)
            euler = np.degrees(matrix_to_euler_angles(rotmat, convention="ZYZ").numpy())
            # the rot and psi below have been swapped and their sign have been changed
            # in order to follow the relion convention
            # (see starfile_params.get_rotmat())
            return {
                "rotmat": rotmat,
                "relion_angle_psi": -euler[:, 0],
                "relion_angle_tilt": (-1 if self.config.relion_invert_hand else 1)
                * euler[:, 1],
                "relion_angle_rot": -euler[:, 2],
            }
        else:
            raise NotImplementedError(
                f"Angle distribution : '{self.config.angle_distribution}' "
                f"has not been implemented!"
            )

    def get_ctf_params(self):
        """Get the parameters for the CTF of the particle from a distribution.

        If config.ctf is True else returns None

        Returns
        -------
        ctf_params: dict of type str to {tensor}
            dictionary containing:
            "defocus_u": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the major
                defocus value in microns
            "defocus_v": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the minor
                defocus value in microns
            "defocus_angle": torch.Tensor
                Tensor of size (batch_size,1,1,1) that contains the astigatism
                angle in radians
        """
        if self.config.ctf:
            defocus_u = (
                self.config.min_defocus
                + (self.config.max_defocus - self.config.min_defocus)
                * torch.zeros(self.config.batch_size)[:, None, None, None].uniform_()
            )
            defocus_v = defocus_u.clone().detach()
            defocus_angle = torch.zeros_like(defocus_u)
            return {
                "defocus_u": defocus_u,
                "defocus_v": defocus_v,
                "defocus_angle": defocus_angle,
            }
        else:
            return None

    def get_shift_params(self):
        """Get the parameters for the shift of the particle from a distribution.

        If config.shift is True else returns None.

        Returns
        -------
        shift_params: dict of type str to {torch.Tensor}
            dictionary containing
            'shift_x': torch.Tensor (batch_size,)
                batch of shifts along horizontal axis in Angstrom
            'shift_y': torch.Tensor (batch_size,)
                batch of shifts along vertical axis in Angstrom
        """
        if self.config.shift:
            if self.config.shift_distribution == "triangular":
                shiftNormalized = torch.Tensor(self.config.batch_size, 2).uniform_()
                shifts = (
                    uniform_to_triangular(shiftNormalized)
                    * self.config.side_len
                    * self.config.shift_std_deviation
                    / 100.0
                ).float()
                params = {"shift_x": shifts[:, 0], "shift_y": shifts[:, 1]}
                return params
            else:
                print("here")
                raise NotImplementedError(
                    f"Shift distribution '{self.config.shift_distribution}' "
                    f"has not been implemented!"
                )

        else:
            return None
