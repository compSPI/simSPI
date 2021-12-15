"""Unit test for TEM Simulator wrapper."""

import os
from pathlib import Path

import numpy as np
import pytest

from simSPI import fov, tem


@pytest.fixture
def sample_class(tmp_path):
    """Instantiate TEMSimulator for testing."""
    test_files_path = "./tests/test_files"
    cwd = os.getcwd()

    tem_simulator = tem.TEMSimulator(
        str(Path(cwd, test_files_path, "path_config.yaml")),
        str(Path(cwd, test_files_path, "sim_config.yaml")),
    )

    # from test_files/path_config
    out_file_name = "_randomrot"

    tem_simulator.output_path_dict["crd_file"] = str(
        Path(cwd, tmp_path, out_file_name + ".txt")
    )
    tem_simulator.output_path_dict["mrc_file"] = str(
        Path(cwd, tmp_path, out_file_name + ".mrc")
    )
    tem_simulator.output_path_dict["log_file"] = str(
        Path(cwd, tmp_path, out_file_name + ".log")
    )
    tem_simulator.output_path_dict["inp_file"] = str(
        Path(cwd, tmp_path, out_file_name + ".inp")
    )
    tem_simulator.output_path_dict["h5_file"] = str(
        Path(cwd, tmp_path, out_file_name + ".h5")
    )
    tem_simulator.output_path_dict["h5_file_noisy"] = str(
        Path(cwd, tmp_path, out_file_name + "-noisy.h5")
    )
    tem_simulator.output_path_dict["pdb_file"] = str(
        Path(cwd, test_files_path, "4v6x.pdb")
    )

    return tem_simulator


@pytest.fixture
def sample_resources():
    """Return sample resources for testing."""
    test_files_path = "./tests/test_files"
    cwd = os.getcwd()
    resources = {}

    resources["files"] = {
        "path_yaml": str(Path(cwd, test_files_path, "path_config.yaml")),
        "sim_yaml": str(Path(cwd, test_files_path, "sim_config.yaml")),
        "pdb_file": str(Path(cwd, test_files_path, "4v6x.pdb")),
    }

    micrograph = np.load(str(Path(cwd, test_files_path, "micrograph.npz")))

    resources["data"] = {"micrograph": micrograph.f.arr_0}

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


def test_classify_input_config(sample_class):
    """Test classification of simulation parameters."""
    raw_params = {
        "molecular_model": {
            "voxel_size_nm": 0.1,
            "particle_name": "toto",
            "particle_mrcout": "None",
        },
        "specimen_grid_params": {
            "hole_diameter_nm": 1200,
            "hole_thickness_center_nm": 100,
            "hole_thickness_edge_nm": 100,
        },
        "beam_parameters": {
            "voltage_kv": 300,
            "energy_spread_v": 1.3,
            "electron_dose_e_nm2": 100,
            "electron_dose_std_e_per_nm2": 0,
        },
        "optics_parameters": {
            "magnification": 81000,
            "spherical_aberration_mm": 2.7,
            "chromatic_aberration_mm": 2.7,
            "aperture_diameter_um": 50,
            "focal_length_mm": 3.5,
            "aperture_angle_mrad": 0.1,
            "defocus_um": 1.0,
            "defocus_syst_error_um": 0.0,
            "defocus_nonsyst_error_um": 0.0,
            "optics_defocusout": "None",
        },
        "detector_parameters": {
            "detector_nx_px": 5760,
            "detector_ny_px": 4092,
            "detector_pixel_size_um": 5,
            "average_gain_count_per_electron": 2,
            "noise": "no",
            "detector_q_efficiency": 0.5,
            "mtf_params": [0, 0, 1, 0, 0],
        },
    }

    returned_params = sample_class.classify_input_config(raw_params)

    for param_group_name, param_list in returned_params.items():
        assert param_group_name in raw_params

        for param_value in raw_params[param_group_name].values():
            if type(param_value) is list:
                for items in param_value:
                    assert items in param_list
            else:
                assert param_value in param_list


def test_generate_path_dict(sample_class, sample_resources):
    """Test whether returned path dictionary has expected file paths."""
    expected_path_template = {
        "pdb_file": ".pdb",
        "crd_file": ".txt",
        "mrc_file": ".mrc",
        "log_file": ".log",
        "inp_file": ".inp",
        "h5_file": ".h5",
        "h5_file_noisy": "-noisy.h5",
    }
    returned_paths = sample_class.generate_path_dict(
        sample_resources["files"]["pdb_file"]
    )
    for file_type, file_path in returned_paths.items():
        assert file_type in expected_path_template
        directory, file = os.path.split(file_path)
        assert os.path.isdir(directory)
        assert expected_path_template[file_type] in file


def test_create_crd_file(sample_class):
    """Test creation of .crd file."""
    sample_class.create_crd_file(pad=5)
    assert os.path.isfile(sample_class.output_path_dict["crd_file"])


def test_create_inp_file(sample_class):
    """Test creation of .inp file."""
    sample_class.write_inp_file()
    assert os.path.isfile(sample_class.output_path_dict["inp_file"])


def test_extract_particles(sample_class, sample_resources):
    """Test extract_particles returns particles of expected shape from mrc."""
    particles = sample_class.extract_particles(
        sample_resources["data"]["micrograph"], 5.0
    )

    assert particles.shape == (
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
        sample_resources["data"]["micrograph"],
        sample_class.sim_dict["optics_parameters"],
        sample_class.sim_dict["detector_parameters"],
        pdb_file=sample_resources["files"]["pdb_file"],
        pad=5.0,
    )

    sample_class.export_particle_stack(particles)
    assert os.path.isfile(sample_class.output_path_dict["h5_file"])
    assert os.path.isfile(sample_class.output_path_dict["h5_file_noisy"])
