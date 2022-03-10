"""Unit test for TEM Simulator wrapper."""

import os
from pathlib import Path

import numpy as np
import pytest

from simSPI import fov, tem


@pytest.fixture
def sample_class():
    """Instantiate TEMSimulator for testing."""
    test_files_path = "./test_files"
    cwd = os.getcwd()

    tem_simulator = tem.TEMSimulator(
        str(Path(cwd, test_files_path, "path_config.yaml")),
        str(Path(cwd, test_files_path, "sim_config.yaml")),
    )

    return tem_simulator


@pytest.fixture
def sample_resources():
    """Return sample resources for testing."""
    test_files_path = "./test_files"
    cwd = os.getcwd()
    resources = {
        "files": {
            "path_yaml": str(Path(cwd, test_files_path, "path_config.yaml")),
            "sim_yaml": str(Path(cwd, test_files_path, "sim_config.yaml")),
            "pdb_file": str(Path(cwd, test_files_path, "4v6x.pdb")),
            "metadata_params_file": str(
                Path(cwd, test_files_path, "metadata_fields.yaml")
            ),
        }
    }

    micrograph = np.load(str(Path(cwd, test_files_path, "micrograph.npz")))

    resources["data"] = {"micrograph": np.array([micrograph.f.arr_0])}

    return resources


def test_temsimulator_constructor(sample_resources):
    """Test whether constructor populates attributes."""
    tem_sim = tem.TEMSimulator(
        sample_resources["files"]["path_yaml"], sample_resources["files"]["sim_yaml"]
    )
    parameters_dict_keys = [
        "simulation",
        "sample",
        "particle",
        "particleset",
        "beam",
        "optics",
        "detector",
        "geometry",
    ]
    assert tem_sim.output_path_dict is not None
    assert tem_sim.sim_dict is not None
    assert set(parameters_dict_keys).issubset(tem_sim.parameter_dict.keys())


def test_get_config_from_yaml(sample_resources, sample_class):
    """Test whether yaml is parsed."""
    expected_config_template = {
        "beam_parameters": 4,
        "optics_parameters": 10,
        "detector_parameters": 11,
        "specimen_grid_params": 3,
        "molecular_model": 3,
    }

    test_yaml = sample_resources["files"]["sim_yaml"]
    returned_config = sample_class.get_config_from_yaml(test_yaml)

    for config_group, config_list in returned_config.items():
        assert config_group in expected_config_template
        assert len(config_list) is expected_config_template[config_group]


def test_generate_simulator_inputs(sample_class): #TODO: update test
    """Test whether simulator required files are created.
    """
    sample_class.generate_simulator_inputs()

    assert os.path.isfile(sample_class.output_path_dict["inp_file"])
    assert os.path.isfile(sample_class.output_path_dict["defocus_file"])
    assert os.path.isfile(sample_class.output_path_dict["crd_file"])



def test_create_crd_file(sample_class):
    """Test creation of .crd file."""
    sample_class.create_crd_file(pad=5)
    assert os.path.isfile(sample_class.output_path_dict["crd_file"])


def test_create_inp_file(sample_class):
    """Test creation of .inp file."""
    sample_class.create_inp_file()
    assert os.path.isfile(sample_class.output_path_dict["inp_file"])


def test_create_defocus_file(sample_class):
    """Test creation of defocus file."""
    sample_class.create_defocus_file()
    assert os.path.isfile(sample_class.output_path_dict["defocus_file"])


def test_parse_simulator_data (sample_class, sample_resources): #TODO: update test
    """Test parse_simulator_data returns particles of expected shape from mrc."""
    particles = sample_class.parse_simulator_data(
        sample_resources["data"]["micrograph"], 5.0
    )

    assert particles.shape == (
        1,
        35,
        809,
        809,
    )


def test_export_particle_stack(sample_class, sample_resources):
    """Test if particle stack is exported as h5 file."""
    sample_class.sim_dict["molecular_model"] = [0.1, "toto", "None"]
    sample_class.sim_dict["optics_parameters"] = [
        81000,
        2.7,
        2.7,
        50,
        3.5,
        0.1,
        1.0,
        0.0,
        0.0,
        "None",
    ]
    sample_class.sim_dict["detector_parameters"] = [
        5760,
        4092,
        5,
        2,
        "no",
        0.5,
        0,
        0,
        1,
        0,
        0,
    ]

    particles = fov.micrograph2particles(
        sample_resources["data"]["micrograph"][0],
        sample_class.sim_dict["optics_parameters"],
        sample_class.sim_dict["detector_parameters"],
        pdb_file=sample_resources["files"]["pdb_file"],
        pad=5.0,
    )

    sample_class.export_simulated_data(particles)
    assert os.path.isfile(sample_class.output_path_dict["h5_file"])
    assert os.path.isfile(sample_class.output_path_dict["h5_file_noisy"])


def test_run_simulator(sample_class):
    """Test whether mrc data is generated from local tem installation.

    Notes
    -----
    This test requires a local TEM sim installation to run.
    """
    sample_class.create_crd_file(pad=5)
    sample_class.create_inp_file()
    data = sample_class.run_simulator()
    assert os.path.isfile(sample_class.output_path_dict["log_file"])
    assert os.path.isfile(sample_class.output_path_dict["mrc_file"])



def test_apply_gaussian_noise(sample_class, sample_resources):
    """Test if gaussian noise is applied properly to particle stack.

    Notes
    -----
    This test requires a local TEM sim installation to run.
    """
    sample_class.sim_dict["molecular_model"] = [0.1, "toto", "None"]
    sample_class.sim_dict["optics_parameters"] = [
        81000,
        2.7,
        2.7,
        50,
        3.5,
        0.1,
        1.0,
        0.0,
        0.0,
        "None",
    ]
    sample_class.sim_dict["detector_parameters"] = [
        5760,
        4092,
        5,
        2,
        "no",
        0.5,
        0,
        0,
        1,
        0,
        0,
    ]

    particles = fov.micrograph2particles(
        sample_resources["data"]["micrograph"][0],
        sample_class.sim_dict["optics_parameters"],
        sample_class.sim_dict["detector_parameters"],
        pdb_file=sample_resources["files"]["pdb_file"],
        pad=5.0,
    )

    noisy_particles = sample_class.apply_gaussian_noise(particles)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, particles, noisy_particles
    )

    sample_class.parameter_dict.pop("other", None)

    original_particles = sample_class.apply_gaussian_noise(particles)
    np.testing.assert_array_equal(particles, original_particles)
