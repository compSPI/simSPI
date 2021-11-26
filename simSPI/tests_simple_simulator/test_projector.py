from ..linear_simulator.modules import Projector
from ..linear_simulator.utils import init_data, normalized_mse


def test_projector():
    path = "simSPI/tests_simple_simulator/tests_data.npy"

    saved_data, config = init_data(path)
    rot_params = saved_data["rot_params"]
    projector = Projector(config)
    projector.vol = saved_data["volume"]

    out = projector(rot_params)
    error = normalized_mse(saved_data["projector_output"], out).item()
    assert (error < 0.01) == 1
