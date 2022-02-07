"""Test function for CTF module."""
import numpy as np
import torch

from simSPI.linear_simulator.ctf import CTF


class AttrDict(dict):
    """Class to convert a dictionary to a class.

    Parameters
    ----------
    dict: dictionary

    """

    def __init__(self, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_data(path):
    """Load data for the test functions.

    Loads .npy file from a path and converts its config dictionary into a class

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
    """Return mean square error.

    Calculate mean square error between two inputs normalized
    by the norm of the first input.

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


def primal_to_fourier_2D(r):
    """Return fourier transform of a batch of image.

    Parameters
    ----------
    r: torch.Tensor
        Tensor of size (batch,1, size,size)

    Returns
    -------
    out: torch.Tensor
        Tensor of size (batch,1, size,size)
    """
    r = torch.fft.fftshift(r, dim=(-2, -1))
    return torch.fft.ifftshift(
        torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1)
    )


def test_ctf_forward():
    """Test accuracy of the ctf."""
    path = "tests/data/ctf_data.npy"

    saved_data, config = init_data(path)
    ctf_params = saved_data["ctf_params"]
    im_input = saved_data["projector_output"]

    ctf = CTF(config)
    ctf_output = ctf(primal_to_fourier_2D(im_input), ctf_params)

    assert normalized_mse(saved_data["ctf_output"], ctf_output).abs() < 0.01


def test_ctf_forward_bfactor():
    """Test accuracy of the ctf with bfactor."""
    path = "tests/data/ctf_data.npy"

    saved_data, config = init_data(path)
    ctf_params = saved_data["ctf_params"]
    im_input = saved_data["projector_output"]
    decay = np.sqrt(-np.log(config.value_nyquist)) * 2.0 * config.pixel_size
    config.b_factor = 4 * decay ** 2

    ctf = CTF(config)

    ctf_output = ctf(primal_to_fourier_2D(im_input), ctf_params)

    assert normalized_mse(saved_data["ctf_output"], ctf_output).abs() < 0.01


def test_get_ctf():
    """Test accuracy of the get ctf."""
    path = "tests/data/ctf_data.npy"

    saved_data, config = init_data(path)
    ctf_params = saved_data["ctf_params"]

    ctf = CTF(config)
    ctf_out = ctf.get_ctf(ctf_params)
    assert normalized_mse(saved_data["ctf_fourier"], ctf_out).abs() < 0.01


def test_get_wavelength():
    """Test accuracy of the _get_wavelength."""
    path = "tests/data/ctf_data.npy"

    _, config = init_data(path)
    ctf = CTF(config)
    wavelength = ctf._get_ewavelength()
    true_wavelength = 0.1968
    error = (wavelength - true_wavelength) / true_wavelength
    assert error < 0.01
