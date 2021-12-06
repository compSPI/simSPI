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
        super(Projector, self).__init__()

        self.config = config
        self.vol = torch.rand([self.config.side_len] * 3, dtype=torch.float32)
        lincoords = torch.linspace(
            -1.0, 1.0, self.config.side_len
        )  # assume square volume
        [X, Y, Z] = torch.meshgrid([lincoords, lincoords, lincoords])
        coords = torch.stack([Y, X, Z], dim=-1)
        self.register_buffer("vol_coords", coords.reshape(-1, 3))

    def forward(self, rot_params, proj_axis=-1):
        """Output the tomographic projection of the volume.

        First rotate the volume and then sum it along an axis.

        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            contains rotation matrix "rotmat" (batch_size x 3 x 3) that is
            used to rotate the volume
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
