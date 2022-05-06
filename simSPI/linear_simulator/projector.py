"""Class to generate tomographic projection."""

import torch
import numpy as np
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
            freq_coords = torch.fft.fftfreq(self.config.side_len,dtype=torch.float32)
            [x, y] = torch.meshgrid(
                [
                    freq_coords,
                ]
                * 2
            )
            coords = torch.stack([y, x], dim=-1)
            self.register_buffer("vol_coords", coords.reshape(-1, 2))            

    def forward(self, rot_params, proj_axis=-1):
        if self.space == "real":
            return self._forward_real(rot_params,proj_axis)
        elif self.space == "fourier":
            return self._forward_fourier(rot_params)

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
            Tensor containing tomographic projection
            (batch_size x 1 x sidelen x sidelen)
        """
        
        rotmat = rot_params["rotmat"]
        batch_sz = rotmat.shape[0]
        rot_vol_coords = 2*self.vol_coords.repeat((batch_sz,1,1)).bmm(rotmat[:,:2,:])
        
        projection = torch.empty((batch_sz,
                                  self.config.side_len,
                                  self.config.side_len), dtype=torch.complex64)
        projection.real = torch.nn.functional.grid_sample(
            self.vol.real.repeat((batch_sz, 1, 1, 1, 1)),
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
        ).reshape(batch_sz,
                  self.config.side_len,
                  self.config.side_len)

        projection.imag = torch.nn.functional.grid_sample(
            self.vol.imag.repeat((batch_sz, 1, 1, 1, 1)),
            rot_vol_coords[:, None, None, :, :],
            align_corners=True,
        ).reshape(batch_sz,
                  self.config.side_len,
                  self.config.side_len)

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
