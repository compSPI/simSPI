"""Unit tests for fov.py."""

import numpy as np

from simSPI import fov

pdb_path = "../tests/test_files/fov_test.pdb"


def test_define_grid_in_fov():
    """Test calculation of grid for picking micrograph particles."""
    # detector params
    detector_nx = 5760
    detector_ny = 4092
    detector_pixel_size = 5

    # optics params
    magnification = 81000

    x, y, n = fov.define_grid_in_fov(
        [magnification], [detector_nx, detector_ny, detector_pixel_size], pdb_path
    )

    assert x.size == 16
    assert y.size == 11
    assert n == 176


def test_get_fov():
    """Test correct calculation of fov from input simulation parameters."""
    # detector params
    detector_nx = 5760
    detector_ny = 4092
    detector_pixel_size = 5

    # optics params
    magnification = 81000

    d_max = 40

    expected_fov_lx = 355.55555555555554
    expected_fov_ly = 252.59259259259258

    assert fov.get_fov(
        [magnification], [detector_nx, detector_ny, detector_pixel_size]
    ) == (expected_fov_lx, expected_fov_ly, 102.0)
    assert fov.get_fov(
        [magnification], [detector_nx, detector_ny, detector_pixel_size], pdb_path
    ) == (expected_fov_lx, expected_fov_ly, 21.367856950503086)
    assert (
        fov.get_fov(
            [magnification],
            [detector_nx, detector_ny, detector_pixel_size],
            pdb_path,
            d_max,
        )
        == (expected_fov_lx, expected_fov_ly, 42.0)
    )


def test_get_dmax():
    """Test correct retrieval of maximum particle dimension from example .pdb file."""
    assert fov.get_dmax(pdb_path) == 19.367856950503086


def test_get_xyz_from_pdb():
    """Test reading and retrieval of particle dimensions from example .pdb file."""
    assert fov.get_xyz_from_pdb(pdb_path).shape == (1, 3, 3)


def test_micrograph2particles():
    """Test correct picking of particles from micrograph."""
    # detector params
    detector_nx = 5760
    detector_ny = 4092
    detector_pixel_size = 5

    # optics params
    magnification = 81000

    micrograph = np.ones((32, 216))

    particles = fov.micrograph2particles(
        micrograph, [magnification], [detector_nx, detector_ny, detector_pixel_size]
    )

    assert particles.shape == (32, 8, 8)


def test_slice_and_stack():
    """Test correct slicing of micrograph into particle data."""
    box_size = 256
    overlap = 2
    arr = np.ones((10 * box_size, 10 * box_size))

    assert fov.slice_and_stack(arr, box_size, 0).shape == (100, 256, 256)
    assert fov.slice_and_stack(arr, box_size, overlap).shape == (81, 260, 260)


def test_blockshaped():
    """Test dimensions of array returned by blockshaped are correct."""
    n_rows = 256
    n_cols = 256
    arr = np.ones((10 * n_rows, 10 * n_cols))

    expanded_arr = fov.blockshaped(arr, n_rows, n_cols)
    expanded_arr_shape = expanded_arr.shape

    assert (
        arr.size
        == expanded_arr_shape[0] * expanded_arr_shape[1] * expanded_arr_shape[2]
    )
    assert expanded_arr_shape[0] == 100
    assert expanded_arr_shape[1] == 256
    assert expanded_arr_shape[2] == 256
