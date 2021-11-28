"""Test function for simulator module."""
from ..linear_simulator.modules import simulator
from ..linear_simulator.utils import init_data, normalized_mse


def test_simulator():
    """Test accuracy of linear forward model."""
    path = "simSPI/tests_simple_simulator/tests_data.npy"

    saved_data, config = init_data(path)
    rot_params = saved_data["rot_params"]
    ctf_params = saved_data["ctf_params"]
    shift_params = saved_data["shift_params"]

    sim = simulator(config)
    sim.Projector.vol = saved_data["volume"]
    out = sim(rot_params, ctf_params, shift_params)

    assert normalized_mse(saved_data["final_output"].real, out) < 0.01
