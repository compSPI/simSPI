"""Test function for volume_utils."""
import torch

from simSPI.linear_simulator.volume_utils import init_cube


def test_init_cube():
    """Test if the volume from init_cube projects a square."""
    sidelen = 33
    volume = init_cube(sidelen)
    proj_actual = torch.zeros(sidelen, sidelen)
    half = sidelen // 2
    length = sidelen // 8
    proj_actual[half - length : half + length, half - length : half + length] = (
        length * 2
    )

    assert (volume.sum(0).squeeze() - proj_actual).pow(2).sum().item() < 1e-3
    assert (volume.sum(1).squeeze() - proj_actual).pow(2).sum().item() < 1e-3
    assert (volume.sum(2).squeeze() - proj_actual).pow(2).sum().item() < 1e-3
