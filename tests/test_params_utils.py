"""Contain test functions associated with params_utils."""
import pytest
import torch

from simSPI.linear_simulator.params_utils import (
    DistributionalParams,
    ParamsFactory,
    StarfileParams,
    params_update,
)


def test_params_factory():
    """Test if the factory outputs the right choice."""

    class config:
        input_starfile_path = "tests/data/test.star"
        starfile_available = True

    param_generator = ParamsFactory.get_params_generator(config)
    assert type(param_generator).__name__ == "StarfileParams"

    config.starfile_available = False
    param_generator = ParamsFactory.get_params_generator(config)
    assert type(param_generator).__name__ == "DistributionalParams"


class TestStarfileParams:
    """Class with test for starfile params."""

    def init(self):
        """Initialize the params generator."""

        class config:
            input_starfile_path = "tests/data/test.star"
            batch_size = 4
            relion_invert_hand = False

        self.config = config
        self.params_generator = StarfileParams(self.config)
        self.params_generator.particle = self.params_generator.df["particles"].iloc[
            0 : self.config.batch_size
        ]

    def test_get_rotmat(self):
        """Test if the rotmat shape is correct."""
        self.init()

        rot_params = self.params_generator.get_rotmat()
        assert rot_params["rotmat"].shape == torch.Size([self.config.batch_size, 3, 3])

    def test_get_ctf_params(self):
        """Test if the ctf params autput is correct."""
        self.init()
        self.config.ctf = True
        ctf_params = self.params_generator.get_ctf_params()
        assert ctf_params["defocus_u"].shape == torch.Size(
            [self.config.batch_size, 1, 1, 1]
        )
        assert ctf_params["defocus_v"].shape == torch.Size(
            [self.config.batch_size, 1, 1, 1]
        )
        assert ctf_params["defocus_angle"].shape == torch.Size(
            [self.config.batch_size, 1, 1, 1]
        )

        self.config.ctf = False
        ctf_params = self.params_generator.get_ctf_params()
        assert ctf_params is None

    def test_get_shift_params(self):
        """Test if the shift_params output is correct."""
        self.init()
        self.config.shift = True
        shift_params = self.params_generator.get_shift_params()
        assert shift_params["shift_x"].shape == torch.Size([self.config.batch_size])
        assert shift_params["shift_y"].shape == torch.Size([self.config.batch_size])

        self.config.shift = False
        shift_params = self.params_generator.get_shift_params()
        assert shift_params is None

    def test_get_params(self):
        """Test if the params output is correct."""
        self.init()
        self.config.ctf = True
        self.config.shift = True
        assert len(self.params_generator.get_params()) == 3


class TestDistributionalParams:
    """Class with test for distribution params."""

    def init(self):
        """Initialize the params generator."""

        class config:
            batch_size = 4
            relion_invert_hand = False
            min_defocus = 1
            max_defocus = 1.5
            shift_std_deviation = 3
            side_len = 32

        self.config = config
        self.params_generator = DistributionalParams(config)

    def test_get_rotmat(self):
        """Test if the rotmat shape is correct."""
        self.init()
        self.config.angle_distribution = "uniform"
        rot_params = self.params_generator.get_rotmat()
        assert rot_params["rotmat"].shape == torch.Size([self.config.batch_size, 3, 3])

        self.config.angle_distribution = "not_implemented"

        expected = (
            f"Angle distribution : '{self.config.angle_distribution}' "
            f"has not been implemented!"
        )
        with pytest.raises(NotImplementedError) as exception_context:
            rot_params = self.params_generator.get_rotmat()
        actual = str(exception_context.value)
        assert expected in actual

    def test_get_ctf_params(self):
        """Test if the ctf params is correct."""
        self.init()
        self.config.ctf = True
        ctf_params = self.params_generator.get_ctf_params()
        assert ctf_params["defocus_u"].shape == torch.Size(
            [self.config.batch_size, 1, 1, 1]
        )
        assert ctf_params["defocus_v"].shape == torch.Size(
            [self.config.batch_size, 1, 1, 1]
        )
        assert ctf_params["defocus_angle"].shape == torch.Size(
            [self.config.batch_size, 1, 1, 1]
        )

        self.config.ctf = False
        ctf_params = self.params_generator.get_ctf_params()
        assert ctf_params is None

    def test_get_shift_params(self):
        """Test if the shift shape is correct."""
        self.init()
        self.config.shift = True

        self.config.shift_distribution = "triangular"

        shift_params = self.params_generator.get_shift_params()
        assert shift_params["shift_x"].shape == torch.Size([self.config.batch_size])
        assert shift_params["shift_y"].shape == torch.Size([self.config.batch_size])

        self.config.shift_distribution = "not_implemented"
        expected = (
            f"Shift distribution '{self.config.shift_distribution}' "
            f"has not been implemented!"
        )
        with pytest.raises(NotImplementedError) as exception_context:
            shift_params = self.params_generator.get_shift_params()
        actual = str(exception_context.value)
        assert expected in actual

        self.config.shift = False
        shift_params = self.params_generator.get_shift_params()
        assert shift_params is None

    def test_get_params(self):
        """Test if the get params output is correct."""
        self.init()
        self.config.angle_distribution = "uniform"
        self.config.shift_distribution = "triangular"
        self.config.ctf = True
        self.config.shift = True
        assert len(self.params_generator.get_params()) == 3


def test_params_update():
    """Test if the parameters get updated."""

    class config:
        input_starfile_path = ""
        side_len = 32

    config = params_update(config)
    assert config is not None

    config.input_starfile_path = "tests/data/test.star"
    config = params_update(config)
    assert config is not None
