"""Unit test for tem wrapper I/O helper functions."""
import os
import tempfile

import h5py
import mrcfile
import numpy as np

from simSPI import cryoemio


def test_data_and_dic2hdf5():
    """Test data_and_dic2hdf5 helper with a simple hdf5 file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
    tmp.close()

    data = {"a": 1, "b": 2, "c": 3}

    try:
        cryoemio.data_and_dic2hdf5(data, tmp.name)
        with h5py.File(tmp.name, "r") as f:
            assert f["a"] == 1 and f["b"] == 2 and f["c"] == 3
    finally:
        os.unlink(tmp.name)


def test_fill_parameters_dictionary_max():
    """Test data_and_dic2hdf5 helper with a maximal selection of garbage parameters."""
    mrc_file = "a.mrc"
    pdb_file = "a.pdb"
    voxel_size = 0.2
    particle_name = "africa"
    particle_mrcout = "b.mrc"
    crd_file = "a.crd"
    sample_dimensions = [200, 10, 5]
    beam_params = [100, 2.1, 50, 1]
    dose = 20
    optics_params = [21000, 2.1, 2.1, 50, 3.1, 0.5, 1.2, 0, 0]
    defocus = 1.5
    optics_defocout = "optics.txt"
    detector_params = [2120, 1080, 2, 31, "no", 0.1, 0.1, 0.0, 0.7, 0, 0]
    noise = 0.1
    log_file = "itslog.log"
    seed = 210
    key = particle_mrcout.split(".mrc")[0]

    out_dict = cryoemio.fill_parameters_dictionary(
        mrc_file=mrc_file,
        pdb_file=pdb_file,
        voxel_size=voxel_size,
        particle_name=particle_name,
        particle_mrcout=particle_mrcout,
        crd_file=crd_file,
        sample_dimensions=sample_dimensions,
        beam_params=beam_params,
        dose=dose,
        optics_params=optics_params,
        defocus=defocus,
        optics_defocout=optics_defocout,
        detector_params=detector_params,
        noise=noise,
        log_file=log_file,
        seed=seed,
    )

    assert out_dict["simulation"] == {"seed": seed, "logfile": log_file}
    assert out_dict["sample"] == {
        "diameter": sample_dimensions[0],
        "thickness_center": sample_dimensions[1],
        "thickness_edge": sample_dimensions[2],
    }
    assert out_dict["particle"] == {
        "name": particle_name,
        "voxel_size": voxel_size,
        "pdb_file": pdb_file,
        "map_file_re_out": key + "_real.mrc",
        "map_file_im_out": key + "_imag.mrc",
    }
    assert out_dict["particleset"] == {"name": particle_name, "crd_file": crd_file}
    assert out_dict["beam"] == {
        "voltage": beam_params[0],
        "spread": beam_params[1],
        "dose_per_im": dose,
        "dose_sd": beam_params[3],
    }
    assert out_dict["optics"] == {
        "magnification": optics_params[0],
        "cs": optics_params[1],
        "cc": optics_params[2],
        "aperture": optics_params[3],
        "focal_length": optics_params[4],
        "cond_ap_angle": optics_params[5],
        "defocus_nominal": optics_params[6],
        "defocus_syst_error": optics_params[7],
        "defocus_nonsyst_error": optics_params[8],
        "defocus_file_out": optics_defocout,
    }
    assert out_dict["detector"] == {
        "det_pix_x": detector_params[0],
        "det_pix_y": detector_params[1],
        "pixel_size": detector_params[2],
        "gain": detector_params[3],
        "use_quantization": noise,
        "dqe": detector_params[5],
        "mtf_a": defocus,
        "mtf_b": detector_params[7],
        "mtf_c": detector_params[8],
        "mtf_alpha": detector_params[9],
        "mtf_beta": detector_params[10],
        "image_file_out": mrc_file,
    }


def test_fill_parameters_dictionary_min():
    """Test data_and_dic2hdf5 helper with a minimal selection of garbage parameters."""
    mrc_file = "a.mrc"
    pdb_file = "a.pdb"
    crd_file = "a.crd"

    out_dict = cryoemio.fill_parameters_dictionary(
        mrc_file=mrc_file,
        pdb_file=pdb_file,
        crd_file=crd_file,
    )

    assert out_dict["particle"]["pdb_file"] == pdb_file
    assert out_dict["detector"]["image_file_out"] == mrc_file
    assert out_dict["particleset"]["crd_file"] == crd_file


def test_mrc2data():
    """Test mrc2data helper function with a basic mrc file."""
    tmp_mrc = tempfile.NamedTemporaryFile(delete=False, suffix=".mrc")
    tmp_mrc.close()
    data = np.arange(12, dtype=np.int8).reshape(3, 4)

    try:
        with mrcfile.open(tmp_mrc.name) as mrc:
            mrc.set_data(data)
        assert cryoemio.mrc2data(tmp_mrc.name) == data
    finally:
        os.unlink(tmp_mrc.name)


def test_mrc2data_large():
    """Test mrc2data helper function with a 3D mrc file."""
    tmp_mrc = tempfile.NamedTemporaryFile(delete=False, suffix=".mrc")
    tmp_mrc.close()
    data = np.arange(12, dtype=np.int8).reshape(3, 4)
    data = np.tile(data[:, :, np.newaxis], 3, axis=2)

    try:
        with mrcfile.open(tmp_mrc.name) as mrc:
            mrc.set_data(data)
        assert cryoemio.mrc2data(tmp_mrc.name) == data
    finally:
        os.unlink(tmp_mrc.name)


def test_recursively_save_dict_contents_to_group():
    """Test recursively_save_dict_contents_to_group helper with a simple hdf5 file."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5")
    tmp.close()

    data = {"a": 1.0, "b": None, "c": {"d": 1}}

    try:
        with h5py.File(tmp.name, "w") as f:
            cryoemio.recursively_save_dict_contents_to_group(f, "", data)
        with h5py.File(tmp.name, "r") as f:
            assert f["a"] == 1.0 and f["b"] == "None" and f["c/d"] == 1
    finally:
        os.unlink(tmp.name)


def test_write_inp_file():
    """Test write_inp_file helper with output from fill_parameters_dictionary."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".imp")
    tmp.close()

    try:
        data = cryoemio.fill_parameters_dictionary(
            mrc_file="a.mrc",
            pdb_file="a.pdb",
            crd_file="a.crd",
            particle_mrcout="b.mrc",
            optics_defocout="optics.txt",
        )
        cryoemio.write_inp_file(data, tmp.name)
    finally:
        os.unlink(tmp.name)
