"""Contain test functions for save_utils."""
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
import yaml
from ioSPI.particle_metadata import update_optics_config_from_starfile

from simSPI.tem_inputs import (
    populate_tem_input_parameter_dict,
    starfile_append_tem_simulator_data,
    write_tem_inputs_to_inp_file,
    write_tem_defocus_file_from_distribution
)


def normalized_mse(a, b):
    """Return normalized error between two numpy arrays."""
    return np.sum((a - b) ** 2) ** 0.5 / np.sum(a ** 2) ** 0.5


def test_fill_parameters_dictionary_max():
    """Test fill_parameters_dictionary with maximal garbage parameters."""
    tmp_yml = tempfile.NamedTemporaryFile(delete = False, suffix = ".yml")
    tmp_yml.close()

    mrc_file = "a.mrc"
    pdb_file = "a.pdb"
    defocus_file = "a.txt"
    voxel_size = 0.2
    particle_name = "africa"
    particle_mrcout = "b.mrc"
    crd_file = "a.crd"
    hole_diameter = 200
    hole_thickness_center = 10
    hole_thickness_edge = 5
    voltage = 100
    energy_spread = 2.1
    electron_dose = 50
    electron_dose_std = 1
    dose = 20
    magnification = 21000
    spherical_aberration = 2.1
    chromatic_aberration = 2.1
    aperture_diameter = 50
    focal_length = 3.1
    aperture_angle = 0.5
    defocus = 1.2
    defocus_syst_error = 0
    defocus_nonsyst_error = 0
    optics_defocusout = "optics.txt"
    detector_nx = 2120
    detector_ny = 1080
    detector_pixel_size = 2
    detector_gain = 31
    noise = "no"
    detector_q_efficiency = 0.1
    mtf_params = [0.1, 0.0, 0.7, 0, 0]
    noise_override = "yes"
    log_file = "itslog.log"
    seed = 210
    snr = 0.6
    snr_db = 10
    key = particle_mrcout.split(".mrc")[0]
    n_samples = 5
    dist_type = 'uniform'
    dist_parameters = [0, 1]
    try:
        with open(tmp_yml.name, "w") as f:
            data = {
                "molecular_model": {
                    "voxel_size_nm": voxel_size,
                    "particle_name": particle_name,
                    "particle_mrcout": particle_mrcout,
                },
                "specimen_grid_params": {
                    "hole_diameter_nm": hole_diameter,
                    "hole_thickness_center_nm": hole_thickness_center,
                    "hole_thickness_edge_nm": hole_thickness_edge,
                },
                "beam_parameters": {
                    "voltage_kv": voltage,
                    "energy_spread_v": energy_spread,
                    "electron_dose_e_per_nm2": electron_dose,
                    "electron_dose_std_e_per_nm2": electron_dose_std,
                },
                "optics_parameters": {
                    "magnification": magnification,
                    "spherical_aberration_mm": spherical_aberration,
                    "chromatic_aberration_mm": chromatic_aberration,
                    "aperture_diameter_um": aperture_diameter,
                    "focal_length_mm": focal_length,
                    "aperture_angle_mrad": aperture_angle,
                    "defocus_um": defocus,
                    "defocus_syst_error_um": defocus_syst_error,
                    "defocus_nonsyst_error_um": defocus_nonsyst_error,
                    "optics_defocusout": optics_defocusout,
                },
                "detector_parameters": {
                    "detector_nx_px": detector_nx,
                    "detector_ny_px": detector_ny,
                    "detector_pixel_size_um": detector_pixel_size,
                    "average_gain_count_per_electron": detector_gain,
                    "noise": noise,
                    "detector_q_efficiency": detector_q_efficiency,
                    "mtf_params": mtf_params,
                },
                "miscellaneous": {
                    "seed": seed,
                    "signal_to_noise": snr,
                    "signal_to_noise_db": snr_db,
                },
                "geometry_parameters": {
                    "n_samples": n_samples
                },
                "ctf_parameters":{
                    "distribution_type": dist_type,
                    "distribution_parameters": dist_parameters
                }

            }
            contents = yaml.dump(data)
            f.write(contents)
        out_dict = populate_tem_input_parameter_dict(
            tmp_yml.name,
            mrc_file,
            pdb_file,
            crd_file,
            log_file,
            defocus_file,
            dose = dose,
            noise = noise_override,
        )

        assert out_dict["simulation"]["seed"] == seed
        assert out_dict["simulation"]["log_file"] == log_file

        assert out_dict["sample"]["diameter"] == hole_diameter
        assert out_dict["sample"]["thickness_center"] == hole_thickness_center
        assert out_dict["sample"]["thickness_edge"] == hole_thickness_edge

        assert out_dict["particle"]["name"] == particle_name
        assert out_dict["particle"]["voxel_size"] == voxel_size
        assert out_dict["particle"]["pdb_file"] == pdb_file
        assert out_dict["particle"]["map_file_re_out"] == key + "_real.mrc"
        assert out_dict["particle"]["map_file_im_out"] == key + "_imag.mrc"

        assert out_dict["particleset"]["name"] == particle_name
        assert out_dict["particleset"]["crd_file"] == crd_file

        assert out_dict["beam"]["voltage"] == voltage
        assert out_dict["beam"]["spread"] == energy_spread
        assert out_dict["beam"]["dose_per_im"] == dose
        assert out_dict["beam"]["dose_sd"] == electron_dose_std

        assert out_dict["optics"]["magnification"] == magnification
        assert out_dict["optics"]["cs"] == spherical_aberration
        assert out_dict["optics"]["cc"] == chromatic_aberration
        assert out_dict["optics"]["aperture"] == aperture_diameter
        assert out_dict["optics"]["focal_length"] == focal_length
        assert out_dict["optics"]["cond_ap_angle"] == aperture_angle
        assert out_dict["optics"]["defocus_nominal"] == defocus
        assert out_dict["optics"]["defocus_syst_error"] == defocus_syst_error
        assert out_dict["optics"]["defocus_nonsyst_error"] == defocus_nonsyst_error
        assert out_dict["optics"]["defocus_file_out"] == optics_defocusout

        assert out_dict["detector"]["det_pix_x"] == detector_nx
        assert out_dict["detector"]["det_pix_y"] == detector_ny
        assert out_dict["detector"]["pixel_size"] == detector_pixel_size
        assert out_dict["detector"]["gain"] == detector_gain
        assert out_dict["detector"]["use_quantization"] == noise_override
        assert out_dict["detector"]["dqe"] == detector_q_efficiency
        assert out_dict["detector"]["mtf_a"] == defocus
        assert out_dict["detector"]["mtf_b"] == mtf_params[1]
        assert out_dict["detector"]["mtf_c"] == mtf_params[2]
        assert out_dict["detector"]["mtf_alpha"] == mtf_params[3]
        assert out_dict["detector"]["mtf_beta"] == mtf_params[4]
        assert out_dict["detector"]["image_file_out"] == mrc_file

        assert out_dict["other"]["signal_to_noise"] == snr
        assert out_dict["other"]["signal_to_noise_db"] == snr_db
    finally:
        os.unlink(tmp_yml.name)



def test_fill_parameters_dictionary_min():
    """Test fill_parameters_dictionary with minimal garbage parameters."""
    tmp_yml = tempfile.NamedTemporaryFile(delete = False, suffix = ".yml")
    tmp_yml.close()

    mrc_file = "a.mrc"
    pdb_file = "a.pdb"
    defocus_file = "a.txt"
    voxel_size = 0.2
    particle_name = "africa"
    crd_file = "a.crd"
    hole_diameter = 200
    hole_thickness_center = 10
    hole_thickness_edge = 5
    voltage = 100
    energy_spread = 2.1
    electron_dose = 50
    electron_dose_std = 1
    magnification = 21000
    spherical_aberration = 2.1
    chromatic_aberration = 2.1
    aperture_diameter = 50
    focal_length = 3.1
    aperture_angle = 0.5
    defocus_syst_error = 0
    defocus_nonsyst_error = 0
    detector_nx = 2120
    detector_ny = 1080
    detector_pixel_size = 2
    detector_gain = 31
    noise = "no"
    detector_q_efficiency = 0.1
    mtf_params = [0.1, 0.0, 0.7, 0, 0]
    log_file = "itslog.log"
    n_samples = 5

    try:
        with open(tmp_yml.name, "w") as f:
            data = {
                "molecular_model": {
                    "voxel_size_nm": voxel_size,
                    "particle_name": particle_name,
                },
                "specimen_grid_params": {
                    "hole_diameter_nm": hole_diameter,
                    "hole_thickness_center_nm": hole_thickness_center,
                    "hole_thickness_edge_nm": hole_thickness_edge,
                },
                "beam_parameters": {
                    "voltage_kv": voltage,
                    "energy_spread_v": energy_spread,
                    "electron_dose_e_per_nm2": electron_dose,
                    "electron_dose_std_e_per_nm2": electron_dose_std,
                },
                "optics_parameters": {
                    "magnification": magnification,
                    "spherical_aberration_mm": spherical_aberration,
                    "chromatic_aberration_mm": chromatic_aberration,
                    "aperture_diameter_um": aperture_diameter,
                    "focal_length_mm": focal_length,
                    "aperture_angle_mrad": aperture_angle,
                    "defocus_syst_error_um": defocus_syst_error,
                    "defocus_nonsyst_error_um": defocus_nonsyst_error,
                },
                "detector_parameters": {
                    "detector_nx_px": detector_nx,
                    "detector_ny_px": detector_ny,
                    "detector_pixel_size_um": detector_pixel_size,
                    "average_gain_count_per_electron": detector_gain,
                    "noise": noise,
                    "detector_q_efficiency": detector_q_efficiency,
                    "mtf_params": mtf_params,
                },
                "geometry_parameters":{
                    "n_samples": n_samples
                }
            }
            contents = yaml.dump(data)
            f.write(contents)
        out_dict = populate_tem_input_parameter_dict(
            tmp_yml.name, mrc_file, pdb_file, crd_file, log_file,defocus_file
        )

        assert out_dict["simulation"]["log_file"] == log_file

        assert out_dict["sample"]["diameter"] == hole_diameter
        assert out_dict["sample"]["thickness_center"] == hole_thickness_center
        assert out_dict["sample"]["thickness_edge"] == hole_thickness_edge

        assert out_dict["particle"]["name"] == particle_name
        assert out_dict["particle"]["voxel_size"] == voxel_size
        assert out_dict["particle"]["pdb_file"] == pdb_file

        assert out_dict["particleset"]["name"] == particle_name
        assert out_dict["particleset"]["crd_file"] == crd_file

        assert out_dict["beam"]["voltage"] == voltage
        assert out_dict["beam"]["spread"] == energy_spread
        assert out_dict["beam"]["dose_per_im"] == electron_dose
        assert out_dict["beam"]["dose_sd"] == electron_dose_std

        assert out_dict["optics"]["magnification"] == magnification
        assert out_dict["optics"]["cs"] == spherical_aberration
        assert out_dict["optics"]["cc"] == chromatic_aberration
        assert out_dict["optics"]["aperture"] == aperture_diameter
        assert out_dict["optics"]["focal_length"] == focal_length
        assert out_dict["optics"]["cond_ap_angle"] == aperture_angle
        assert out_dict["optics"]["defocus_nominal"] == mtf_params[0]
        assert out_dict["optics"]["defocus_syst_error"] == defocus_syst_error
        assert out_dict["optics"]["defocus_nonsyst_error"] == defocus_nonsyst_error

        assert out_dict["detector"]["det_pix_x"] == detector_nx
        assert out_dict["detector"]["det_pix_y"] == detector_ny
        assert out_dict["detector"]["pixel_size"] == detector_pixel_size
        assert out_dict["detector"]["gain"] == detector_gain
        assert out_dict["detector"]["use_quantization"] == noise
        assert out_dict["detector"]["dqe"] == detector_q_efficiency
        assert out_dict["detector"]["mtf_a"] == mtf_params[0]
        assert out_dict["detector"]["mtf_b"] == mtf_params[1]
        assert out_dict["detector"]["mtf_c"] == mtf_params[2]
        assert out_dict["detector"]["mtf_alpha"] == mtf_params[3]
        assert out_dict["detector"]["mtf_beta"] == mtf_params[4]
        assert out_dict["detector"]["image_file_out"] == mrc_file
    finally:
        os.unlink(tmp_yml.name)




def test_starfile_data():
    """Check if the data_list returned is equal to the input params."""

    class Config:
        """Class to instantiate the config object."""

        batch_size = 12
        input_starfile_path = "tests/data/test.star"
        b_factor = 23

    iterations = 3
    config = update_optics_config_from_starfile(Config)
    rot_val = torch.randn(config.batch_size, 3)
    shift_val = torch.randn(config.batch_size, 2)
    ctf_val = torch.randn(config.batch_size, 3)
    rot_params = {
        "relion_angle_rot": rot_val[:, 0],
        "relion_angle_tilt": rot_val[:, 1],
        "relion_angle_psi": rot_val[:, 2],
    }

    shift_params = {"shift_x": shift_val[:, 0], "shift_y": shift_val[:, 1]}
    ctf_params = {
        "defocus_u": ctf_val[:, 0],
        "defocus_v": ctf_val[:, 1],
        "defocus_angle": ctf_val[:, 2],
    }
    data_list = []
    data_list = starfile_append_tem_simulator_data(
        data_list, rot_params, ctf_params, shift_params, iterations, config
    )
    assert len(data_list) == config.batch_size
    for num, list_var in enumerate(data_list):
        assert isinstance(list_var[0], str)
        assert (
                normalized_mse(list_var[1], rot_params["relion_angle_rot"][num].numpy())
                < 0.01
        )
        assert (
                normalized_mse(list_var[2], rot_params["relion_angle_tilt"][num].numpy())
                < 0.01
        )
        assert (
                normalized_mse(list_var[3], rot_params["relion_angle_psi"][num].numpy())
                < 0.01
        )
        assert normalized_mse(list_var[4], shift_params["shift_x"][num].numpy()) < 0.01
        assert normalized_mse(list_var[5], shift_params["shift_y"][num].numpy()) < 0.01
        assert (
                normalized_mse(list_var[6], 1e4 * ctf_params["defocus_u"][num].numpy())
                < 0.01
        )
        assert (
                normalized_mse(list_var[7], 1e4 * ctf_params["defocus_v"][num].numpy())
                < 0.01
        )
        assert (
                normalized_mse(
                    list_var[8], np.radians(ctf_params["defocus_angle"][num].numpy())
                )
                < 0.01
        )
        assert normalized_mse(list_var[9], config.kv) < 0.01
        assert normalized_mse(list_var[10], config.pixel_size) < 0.01
        assert normalized_mse(list_var[11], config.cs) < 0.01
        assert normalized_mse(list_var[12], config.amplitude_contrast) < 0.01
        assert normalized_mse(list_var[13], config.b_factor) < 0.01


def test_write_inp_file():
    """Test write_inp_file helper with output from fill_parameters_dictionary."""
    tmp_inp = tempfile.NamedTemporaryFile(delete = False, suffix = ".imp")
    tmp_inp.close()
    tmp_yml = tempfile.NamedTemporaryFile(delete = False, suffix = ".yml")
    tmp_yml.close()

    defocus_file = "a.txt"
    mrc_file = "a.mrc"
    pdb_file = "a.pdb"
    voxel_size = 0.2
    particle_name = "africa"
    crd_file = "a.crd"
    hole_diameter = 200
    hole_thickness_center = 10
    hole_thickness_edge = 5
    voltage = 100
    energy_spread = 2.1
    electron_dose = 50
    electron_dose_std = 1
    magnification = 21000
    spherical_aberration = 2.1
    chromatic_aberration = 2.1
    aperture_diameter = 50
    focal_length = 3.1
    aperture_angle = 0.5
    defocus_syst_error = 0
    defocus_nonsyst_error = 0
    detector_nx = 2120
    detector_ny = 1080
    detector_pixel_size = 2
    detector_gain = 31
    noise = "no"
    detector_q_efficiency = 0.1
    mtf_params = [0.1, 0.0, 0.7, 0, 0]
    log_file = "itslog.log"
    n_samples = 10
    try:
        with open(tmp_yml.name, "w") as f:
            data = {
                "molecular_model": {
                    "voxel_size_nm": voxel_size,
                    "particle_name": particle_name,
                },
                "specimen_grid_params": {
                    "hole_diameter_nm": hole_diameter,
                    "hole_thickness_center_nm": hole_thickness_center,
                    "hole_thickness_edge_nm": hole_thickness_edge,
                },
                "beam_parameters": {
                    "voltage_kv": voltage,
                    "energy_spread_v": energy_spread,
                    "electron_dose_e_per_nm2": electron_dose,
                    "electron_dose_std_e_per_nm2": electron_dose_std,
                },
                "optics_parameters": {
                    "magnification": magnification,
                    "spherical_aberration_mm": spherical_aberration,
                    "chromatic_aberration_mm": chromatic_aberration,
                    "aperture_diameter_um": aperture_diameter,
                    "focal_length_mm": focal_length,
                    "aperture_angle_mrad": aperture_angle,
                    "defocus_syst_error_um": defocus_syst_error,
                    "defocus_nonsyst_error_um": defocus_nonsyst_error,
                },
                "detector_parameters": {
                    "detector_nx_px": detector_nx,
                    "detector_ny_px": detector_ny,
                    "detector_pixel_size_um": detector_pixel_size,
                    "average_gain_count_per_electron": detector_gain,
                    "noise": noise,
                    "detector_q_efficiency": detector_q_efficiency,
                    "mtf_params": mtf_params,
                },
                "geometry_parameters": {
                    "n_samples": n_samples
                }
            }
            contents = yaml.dump(data)
            f.write(contents)
        out_dict = populate_tem_input_parameter_dict(
            tmp_yml.name, mrc_file, pdb_file, crd_file, log_file,defocus_file
        )
        write_tem_inputs_to_inp_file(tmp_inp.name, out_dict)
    finally:
        os.unlink(tmp_inp.name)
        os.unlink(tmp_yml.name)


def test_write_tem_defocus_file_from_distribution(tmp_path):
    """Test if defocus file is generated with right format."""

    test_defocus_file = str(Path(tmp_path, "defocus_file_test.txt"))
    test_distribution_len = 10
    test_distribution = list(np.random.rand(test_distribution_len))

    print(test_distribution)

    write_tem_defocus_file_from_distribution(test_defocus_file, test_distribution)

    with open(test_defocus_file, "r") as generated_file:
        rows = generated_file.readlines()
        assert len(rows) == test_distribution_len + 2
        assert str(test_distribution_len) in rows[1]
