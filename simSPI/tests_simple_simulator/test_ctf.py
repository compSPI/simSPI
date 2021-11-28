"""Test function for CTF module."""
from ..linear_simulator.modules import CTF
from ..linear_simulator.utils import init_data, normalized_mse, primal_to_fourier_2D


def test_ctf():
    """Test accuracy of the ctf."""
    path = "simSPI/tests_simple_simulator/tests_data.npy"

    saved_data, config = init_data(path)
    ctf_params = saved_data["ctf_params"]
    im_input = saved_data["projector_output"]

    ctf = CTF(config)
    ctf_out = ctf.get_ctf(ctf_params)
    ctf_output = ctf(primal_to_fourier_2D(im_input), ctf_params)
    assert normalized_mse(saved_data["ctf_fourier"], ctf_out).abs() < 0.01
    assert normalized_mse(saved_data["ctf_output"], ctf_output).abs() < 0.01
