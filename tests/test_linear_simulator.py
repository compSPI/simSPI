"""Test function for simulator module."""

import numpy as np

from simSPI.linear_simulator.linear_simulator import LinearSimulator


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

    Calculates mean square error between two inputs normalized
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


def test_simulator():
    """Test accuracy of linear forward model."""
    path = "simSPI/tests_simple_simulator/linear_simulator_data.npy"

    saved_data, config = init_data(path)
    rot_params = saved_data["rot_params"]
    ctf_params = saved_data["ctf_params"]
    shift_params = saved_data["shift_params"]

    sim = LinearSimulator(config)
    sim.Projector.vol = saved_data["volume"]
    out = sim(rot_params, ctf_params, shift_params)

    assert normalized_mse(saved_data["final_output"].real, out) < 0.01
