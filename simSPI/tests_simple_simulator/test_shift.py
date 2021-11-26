from ..linear_simulator.modules import Shift
from ..linear_simulator.utils import init_data, normalized_mse


def test_shift():
    path = "simSPI/tests_simple_simulator/tests_data.npy"

    saved_data, config = init_data(path)
    shift_params = saved_data["shift_params"]
    f_im_input = saved_data["ctf_output"]

    shift = Shift(config)
    shift_output = shift(f_im_input, shift_params)
    assert normalized_mse(saved_data["shift_output"], shift_output).abs() < 0.01
