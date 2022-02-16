"""contain function dealing with volumes."""
import torch


def init_cube(side_len):
    """Create a volume with cube.

    Parameters
    ----------
    side_len: int
        size of the volume
    Returns
    -------
    volume: torch.Tensor
        Tensor (side_len,side_len,side_len) with a cube
        of size (side_len//4,side_len//4,side_len//4)
    """
    half = side_len // 2
    length = side_len // 8
    volume = torch.zeros([side_len] * 3)
    volume[
        half - length : half + length,
        half - length : half + length,
        half - length : half + length,
    ] = 1
    return volume
