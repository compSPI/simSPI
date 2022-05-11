"""Class to generate tomographic projection."""

import torch
from pytorch3d.transforms import Rotate


class Projector(torch.nn.Module):
    """Class to generate tomographic projection.

    Written by J.N. Martel, Y. S. G. Nashed, and Harshit Gupta.

    Parameters
    ----------
    config: class
        Class containing parameters of the Projector

    """

    def __init__(self, config):
        """Initialize volume grid."""
        super(Projector, self).__init__()

        self.config = config
        self.space = config.space

        if self.space == "real":
            self.vol = torch.rand([self.config.side_len] * 3, dtype=torch.float32)
            lin_coords = torch.linspace(-1.0, 1.0, self.config.side_len)
            [x, y, z] = torch.meshgrid(
                [
                    lin_coords,
                ]
                * 3
            )
            coords = torch.stack([y, x, z], dim=-1)

            self.register_buffer("vol_coords", coords.reshape(-1, 3))
        elif self.space == "fourier":
            # Assume DC coefficient is at self.vol[n//2+1,n//2+1]
            # this means that self.vol = fftshift(fft3(fftshift(real_vol)))
            self.vol = torch.rand([self.config.side_len] * 3, dtype=torch.complex64)
            freq_coords = torch.fft.fftfreq(self.config.side_len, dtype=torch.float32)
            [x, y] = torch.meshgrid(
                [
                    freq_coords,
                ]
                * 2
            )
            coords = torch.stack([y, x], dim=-1)
            # Rescale coordinates to [-1,1] to be compatible with
            # torch.nn.functional.grid_sample
            coords = 2 * coords
            self.register_buffer("vol_coords", coords.reshape(-1, 2))
        else:
            raise NotImplementedError(
                f"Space type '{self.space}' " f"has not been implemented!"
            )

    def forward(self, rot_params, proj_axis=-1):
        """Forward method for projection.

        Parameters
        ----------
        rot_params : tensor of rotation matrices
        """
        if self.space == "real":
            return self._forward_real(rot_params, proj_axis)
        elif self.space == "fourier":
            if proj_axis != -1:
                raise NotImplementedError("proj_axis must currently be -1 for Fourier space projection")
            return self._forward_fourier(rot_params)
        else:
            raise NotImplementedError(
                f"Space type '{self.space}' " f"has not been implemented!"
            )

    def _forward_fourier(self, rot_params):
        """Output the tomographic projection of the volume in Fourier space.

        Take a slide through the Fourier space volume whose normal is
        oriented according to rot_params.  The volume is assumed to be cube
        represented in the fourier space.  The output image follows
        (batch x channel x height x width) convention of pytorch.  Therefore,
        a dummy channel dimension is added at the end to projection.

        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary containing parameters for rotation, with keys
                rotmat: str map to tensor
                    rotation matrix (batch_size x 3 x 3) to rotate the volume

        Returns
        -------
        projection: tensor
            Tensor containing tomographic projection in the Fourier domain
            (batch_size x 1 x sidelen x sidelen)

        Comments
        --------
        Note that the Fourier volumes are arbitrary
        channel x height x width complex valued tensors,
        they are not assumed to be Fourier transforms of a real valued 3D functions.

        Note that the tomographic projection is interpolated on a rotated 2D grid.
        The rotated 2D grid extends outside the boundaries of the 3D grid.
        The values outside the boundaries are not defined in a useful way.
        Therefore, in most applications, it make sense to apply a radial filter
        to the sample.

        """
        rotmat = rot_params["rotmat"]
        batch_sz = rotmat.shape[0]

        # print(rotmat[0])
        rotmat = torch.transpose(rotmat, -1, -2)
        # print(rotmat[0])
        rot_vol_coords = self.vol_coords.repeat((batch_sz, 1, 1)).bmm(rotmat[:, :2, :])
        # print(rot_vol_coords[0])
        # print(rot_vol_coords[1])

        # rescale the coordinates to be compatible with the edge alignment of
        # torch.nn.functional.grid_sample
        if self.config.side_len % 2 == 0:  # even case
            rot_vol_coords = (
                (rot_vol_coords + 1)
                * (self.config.side_len)
                / (self.config.side_len - 1)
            ) - 1
        else:  # odd case
            rot_vol_coords = (
                (rot_vol_coords) * (self.config.side_len) / (self.config.side_len - 1)
            )

        projection = torch.empty(
            (batch_sz, self.config.side_len, self.config.side_len),
            dtype=torch.complex64,
        )
        # interpolation is decomposed to real and imaginary parts due to torch
        # grid_sample type rules. Requires data and coordinates of same type.
        # padding_mode="reflection" is required due to possible pathologies
        # right on the border.
        # however, padding_mode="zeros" is what users might expect in most
        # cases other than these axis aligned cases.
        padding_mode = "zeros"
        projection.real = torch.nn.functional.grid_sample(
            self.vol.real.repeat((batch_sz, 1, 1, 1, 1)),
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
            padding_mode=padding_mode,
        ).reshape(batch_sz, self.config.side_len, self.config.side_len)

        projection.imag = torch.nn.functional.grid_sample(
            self.vol.imag.repeat((batch_sz, 1, 1, 1, 1)),
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
            padding_mode=padding_mode,
        ).reshape(batch_sz, self.config.side_len, self.config.side_len)

        projection = projection[:, None, :, :]
        return projection

    def _forward_real(self, rot_params, proj_axis=-1):
        """Output the tomographic projection of the volume.

        First rotate the volume and then sum it along an axis.
        The volume is assumed to be cube. The output image
        follows (batch x channel x height x width) convention of pytorch.
        Therefore, a dummy channel dimension is added at the end to projection.

        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary containing parameters for rotation, with keys
                rotmat: str map to tensor
                    rotation matrix (batch_size x 3 x 3) to rotate the volume
        proj_axis: int
            index along which summation is done of the rotated volume

        Returns
        -------
        projection: tensor
            Tensor containing tomographic projection
            (batch_size x 1 x sidelen x sidelen)
        """
        rotmat = rot_params["rotmat"]
        batch_sz = rotmat.shape[0]
        t = Rotate(rotmat, device=self.vol_coords.device)
        rot_vol_coords = t.transform_points(self.vol_coords.repeat(batch_sz, 1, 1))

        rot_vol = torch.nn.functional.grid_sample(
            self.vol.repeat(batch_sz, 1, 1, 1, 1),
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
        )
        projection = torch.sum(
            rot_vol.reshape(
                batch_sz,
                self.config.side_len,
                self.config.side_len,
                self.config.side_len,
            ),
            dim=proj_axis,
        )
        projection = projection[:, None, :, :]
        return projection
