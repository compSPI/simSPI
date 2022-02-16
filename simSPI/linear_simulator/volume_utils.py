"""contain function dealing with volumes."""
import torch


def init_cube(sidelen):
    """Create a volume with cube.

    Parameters
    ----------
    sidelen: int
        size of the volume
    Returns
    -------
    volume: torch.Tensor
        Tensor (sidelen,sidelen,sidelen) with a cube
        of size (sidelen//4,sidelen//4,sidelen//4)
    """
    half = sidelen // 2
    length = sidelen // 8
    volume = torch.zeros([sidelen] * 3)
    volume[
        half - length : half + length,
        half - length : half + length,
        half - length : half + length,
    ] = 1
    return volume
