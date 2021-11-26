import numpy as np
import torch


def U2T(x):
    """converts samples from uniform distribution to triangular distribution
    Parameters
    ----------
        x: torch.Tensor
    Returns
    ---------
        out: torch.Tensor
    """
    return (x - 0.5).sign() * (1 - (1 - (2 * x - 1).abs()).sqrt())


def primal_to_fourier_2D(r):
    """returns fourier transform of a batch of image
    Parameters
    ----------
        r: torch.Tensor
            Tensor of size (batch,1, size,size)
    Returns
    ------
        out: torch.Tensor
            Tensor of size (batch,1, size,size)
    """
    r = torch.fft.fftshift(r, dim=(-2, -1))
    return torch.fft.ifftshift(
        torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1)
    )


def fourier_to_primal_2D(f):
    """returns inverse fourier transform of a batch of image
    Parameters
    ----------
    f: torch.Tensor
        Tensor of size (batch,1, size,size)
    Returns
    ------
    out: torch.Tensor
        Tensor of size (batch,1, size,size)
    """
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(
        torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1)
    )


def init_cube(sidelen):
    """Creates a volume with cube
    Parameters
    ----------
        sidelen: int
            size of the volume
    Returns
    -------
        volume: torch.Tensor
            Tensor (sidelen,sidelen,sidelen) with a cube of size (sidelen//2,sidelen//2,sidelen//2)
    """
    L = sidelen // 2
    l = sidelen // 8
    volume = torch.zeros([sidelen] * 3)
    volume[L - l : L + l, L - l : L + l, L - l : L + l] = 1
    return volume


class AttrDict(dict):
    """class to convert a dictionary to a class
    Parameters
    ----------
        dict: dictionary
    """

    def __init__(self, *args, **kwargs):
        """Method to return a class with attributes equal to the input dictionry"""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_data(path):
    """loads .npy file for the test functions from a path and converts its config dictionary into a class
    Parameters
    ----------
        path: str
    Returns
    -------
        saved_data: dict
            loaded dictionary
        config: object

    """
    saved_data = np.load(path, allow_pickle=True).item()
    if "config_dict" in saved_data:
        config_dict = saved_data["config_dict"]
    else:
        config_dict = {}
    config = AttrDict(config_dict)
    return saved_data, config


def normalized_mse(a, b):
    """calculates mean square error between two inputs normalized by the norm of the first input
    Parameters
    ----------
        a: torch.Tensor
        b: torch.Tensor
    Returns
    -------
        out: torch.Tensor
            normalized mse
    """
    return (a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()
