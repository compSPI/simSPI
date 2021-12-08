"""Unit test for TEM Simulator wrapper."""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import simutils
import tem


@pytest.fixture
def get_local_paths():
    """Return paths to run tests on local machine."""
    tem_sim = ""
    test_files_path = "/tests/test_files"
    return Path(tem_sim), Path(test_files_path)


@pytest.fixture
def sample_class(tmp_path, get_local_paths):
    """Instantiate TEMSimulator for testing."""
    tem_sim, test_files_path = get_local_paths

    t = tem.TEMSimulator(
        str(Path.joinpath(test_files_path, "path_config.yaml")),
        str(Path.joinpath(test_files_path, "sim_config.yaml")),
    )

    with open(Path(test_files_path, "sample_parameter_dict.pkl"), "wb") as f:
        pickle.dump(t.parameter_dict, f)

    out_file_name = t.path_dict["pdb_keyword"] + t.path_dict["micrograph_keyword"]

    t.output_path_dict["crd_file"] = str(Path(tmp_path, out_file_name + ".txt"))
    t.output_path_dict["mrc_file"] = str(Path(tmp_path, out_file_name + ".mrc"))
    t.output_path_dict["log_file"] = str(Path(tmp_path, out_file_name + ".log"))
    t.output_path_dict["inp_file"] = str(Path(tmp_path, out_file_name + ".inp"))
    t.output_path_dict["h5_file"] = str(Path(tmp_path, out_file_name + ".h5"))
    t.output_path_dict["pdb_file"] = str(Path(test_files_path, "4v6x.pdb"))
    t.path_dict["simulator_dir"] = str(tem_sim)

    return t


@pytest.fixture
def sample_resources(get_local_paths):
    """Return sample resources for testing."""
    _, test_files_path = get_local_paths
    resources = {}

    resources["files"] = {
        "path_yaml": str(Path.joinpath(test_files_path, "path_config.yaml")),
        "sim_yaml": str(Path.joinpath(test_files_path, "sim_config.yaml")),
        "pdb_file": str(Path.joinpath(test_files_path, "4v6x.pdb"))
    }

    micrograph = np.load("./test_files/micrograph.npz")

    resources["data"] = {"micrograph": micrograph.f.arr_0}

    return resources


def test_temsimulator_constructor(sample_resources):
    """Test whether constructor populates attributes."""
    tem_sim = tem.TEMSimulator(
        sample_resources["files"]["path_yaml"], sample_resources["files"]["sim_yaml"]
    )

    assert tem_sim.path_dict is not None
    assert tem_sim.sim_dict is not None
    assert tem_sim.parameter_dict is not None


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
            "voxel_size": 0.1,
            "particle_name": "toto",
            "particle_mrcout": "None",
        },
        "specimen_grid_params": {
            "hole_diameter": 1200,
            "hole_thickness_center": 100,
            "hole_thickness_edge": 100,
        },
        "beam_parameters": {
            "voltage": 300,
            "energy_spread": 1.3,
            "electron_dose": 100,
            "electron_dose_std": 0,
        },
        "optics_parameters": {
            "magnification": 81000,
            "spherical_aberration": 2.7,
            "chromatic_aberration": 2.7,
            "aperture_diameter": 50,
            "focal_length": 3.5,
            "aperture_angle": 0.1,
            "defocus": 1.0,
            "defocus_syst_error": 0.0,
            "defocus_nonsyst_error": 0.0,
            "optics_defocusout": "None",
        },
        "detector_parameters": {
            "detector_Nx": 5760,
            "detector_Ny": 4092,
            "detector_pixel_size": 5,
            "detector_gain": 2,
            "noise": "no",
            "detector_Q_efficiency": 0.5,
            "MTF_params": [0, 0, 1, 0, 0],
        },
    }

    returned_params = sample_class.classify_sim_params(raw_params)

    for param_group, param_dict in raw_params:
        assert param_group in returned_params

        for value in param_dict.values():
            assert value in returned_params[param_group]


def test_generate_path_dict(sample_class, sample_resources):
    """Test whether returned path dictionary has expected file paths."""
    expected_path_template = {
        "pdb_file": ".pdb",
        "crd_file": ".txt",
        "mrc_file": ".mrc",
        "log_file": ".log",
        "inp_file": ".inp",
        "h5_file": ".h5",
    }

    returned_paths = sample_class.generate_path_dict(sample_resources["pdb_file"])
    for file_type, file_path in returned_paths.items():
        assert file_type in expected_path_template.keys()
        directory, file = os.path.split(file_path)
        assert os.path.isdir(directory)
        assert expected_path_template[file_type] in file


def test_generate_parameter_dict(sample_class): #TODO update
    """Test whether parameter dictionary contains expected keys."""
    parameters_dict_keys = [
        "simulation",
        "sample",
        "particle",
        "particleset",
        "beam",
        "optics",
        "detector",
    ]
    sample_class.output_path_dict = {
        "pdb_file": "./test_files4v6x.pdb",
        "crd_file": "C:\\test_path\\4v6x_randomrot.txt",
        "mrc_file": "C:\\test_path\\4v6x_randomrot.mrc",
        "log_file": "C:\\test_path\\4v6x_randomrot.log",
        "inp_file": "C:\\test_path\\4v6x_randomrot.inp",
        "h5_file": "C:\\test_path\\04v6x_randomrot.h5",
    }
    sample_class.sim_dict = {
        "molecular_model": [0.1, "toto", "None"],
        "specimen_grid_params": [1200, 100, 100],
        "beam_parameters": [300, 1.3, 100, 0],
        "optics_parameters": [81000, 2.7, 2.7, 50, 3.5, 0.1, 1.0, 0.0, 0.0, "None"],
        "detector_parameters": [5760, 4092, 5, 2, "no", 0.5, 0, 0, 1, 0, 0],
    }
    sample_class.raw_sim_dict = {
        "molecular_model": {
            "voxel_size": 0.1,
            "particle_name": "toto",
            "particle_mrcout": "None",
        },
        "specimen_grid_params": {
            "hole_diameter": 1200,
            "hole_thickness_center": 100,
            "hole_thickness_edge": 100,
        },
        "beam_parameters": {
            "voltage": 300,
            "energy_spread": 1.3,
            "electron_dose": 100,
            "electron_dose_std": 0,
        },
        "optics_parameters": {
            "magnification": 81000,
            "spherical_aberration": 2.7,
            "chromatic_aberration": 2.7,
            "aperture_diameter": 50,
            "focal_length": 3.5,
            "aperture_angle": 0.1,
            "defocus": 1.0,
            "defocus_syst_error": 0.0,
            "defocus_nonsyst_error": 0.0,
            "optics_defocusout": "None",
        },
        "detector_parameters": {
            "detector_Nx": 5760,
            "detector_Ny": 4092,
            "detector_pixel_size": 5,
            "detector_gain": 2,
            "noise": "no",
            "detector_Q_efficiency": 0.5,
            "MTF_params": [0, 0, 1, 0, 0],
        },
    }

    generated_parameter_dict = sample_class.generate_parameter_dict(
        sample_class.output_path_dict,
        sample_class.sim_dict,
        sample_class.raw_sim_dict,
        seed = 1234,
    )
    assert set(parameters_dict_keys).issubset(generated_parameter_dict.keys())


def test_create_crd_file(sample_class):
    """Test creation of .crd file."""
    sample_class.create_crd_file(pad = 5)
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
    assert particles.shape == (48, 648, 648)

def test_export_particle_stack(sample_class,sample_resources):
    sample_class.sim_dict["molecular_model"] = [0.1, "toto", "None"]
    sample_class.sim_dict["optics_parameters"] = [81000, 2.7, 2.7, 50, 3.5, 0.1, 1.0, 0.0, 0.0, "None"]
    sample_class.sim_dict["detector_parameters"] = [5760, 4092, 5, 2, "no", 0.5, 0, 0, 1, 0, 0]

    particles = simutils.microgaph2particles(  #TODO: find where this function is now
        sample_resources["data"]["micrograph"],
        sample_class.sim_dict["molecular_model"],
        sample_class.sim_dict["optics_parameters"],
        sample_class.sim_dict["detector_parameters"],
        pdb_file = sample_resources.["files"]["pdb_file"],
        Dmax = 30,
        pad = 5.0,
    )

    sample_class.export_particle_stack(particles)
    assert os.path.isfile(sample_class.output_path_dict["h5_file"])
