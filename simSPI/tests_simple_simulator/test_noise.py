"""Test function for noise module."""
from ..linear_simulator.modules import Noise
from ..linear_simulator.utils import init_data, normalized_mse


def test_shift():
    """Test accuracy of noise operation."""
    path = "simSPI/tests_simple_simulator/tests_data.npy"

    saved_data, config = init_data(path)
    im_input = saved_data["noise_input"]

    noise = Noise(config)
    noise_output = noise(im_input)
    assert normalized_mse(saved_data["final_output"], noise_output).abs() < 0.01
