"""Helper functions for tem.py to define grid FOV for displaying particles."""
import logging
import math

import mdtraj as md
import numpy as np
from scipy.spatial.distance import pdist


def define_grid_in_fov(
    optics_params, detector_params, pdb_file=None, dmax=None, pad=1.0
):
    """Define particle grid for picking particles from micrograph.

    Parameters
    ----------
    optics_params : list
        List of sim parameters pertaining to microscope settings.
    detector_params : list
        List of sim parameters pertaining to detector settings.
    pdb_file : str
        Relative path to write .pdb file to.
    dmax : int
        Maximum dimension of molecule.
    pad : int
        Amount of padding.

    Return
    ------
    x_range : ndarray
        Coordinate range of each picked particle in micrograph in x-dimension.
    y_range : ndarray
        Coordinate range of each picked particle in micrograph in y-dimension.
    n_particles : int
        Number of particles to be picked from micrograph.
    """
    fov_Lx, fov_Ly, boxsize = get_fov(
        optics_params,
        detector_params,
        pdb_file=pdb_file,
        dmax=dmax,
        pad=pad,
    )

    fov_Nx = np.floor(fov_Lx / boxsize)
    fov_Ny = np.floor(fov_Ly / boxsize)

    x_origin = -fov_Lx / 2.0 + boxsize / 2.0
    x_frontier = x_origin + fov_Nx * boxsize
    y_origin = -fov_Ly / 2.0 + boxsize / 2.0
    y_frontier = y_origin + fov_Ny * boxsize

    x_range = np.arange(x_origin, x_frontier, boxsize)
    y_range = np.arange(y_origin, y_frontier, boxsize)
    n_particles = np.int(fov_Nx * fov_Ny)
    return x_range, y_range, n_particles


def get_fov(optics_params, detector_params, pdb_file=None, dmax=None, pad=1.0):
    """Define field of view dimensions for displaying particle.

    Parameters
    ----------
    optics_params : list
        List of sim parameters pertaining to microscope settings.
    detector_params : list
        List of sim parameters pertaining to detector settings.
    pdb_file : str
        Relative path to .pdb file output.
    dmax : int
        Maximum dimension of molecule.
    pad : int
        Amount of padding.

    Returns
    -------
    fov_Lx : float
        Length of fov in x-dimension.
    fov_Ly : float
        Length of fov in y-dimension.
    boxsize : float
        Boxsize of particle, equal to max particle dimension with pad.
    """
    detector_Nx = detector_params[0]
    detector_Ny = detector_params[1]
    detector_pixel_size = detector_params[2] * 1e3
    magnification = optics_params[0]

    detector_Lx = detector_Nx * detector_pixel_size
    detector_Ly = detector_Ny * detector_pixel_size
    fov_Lx = detector_Lx / magnification
    fov_Ly = detector_Ly / magnification

    if dmax is None:
        if pdb_file is not None:
            dmax = get_dmax(pdb_file)
        else:
            dmax = 100
    boxsize = dmax + 2 * pad
    return fov_Lx, fov_Ly, boxsize


def get_dmax(filename):
    """Get maximmum dimension of particle.

    Parameters
    ----------
    filename : str
        Relative path to file containing topological information of particle

    Returns
    -------
    np.amax(distance) : float
        Max dimension of given particle in .pdb source file.
    """
    xyz = get_xyz_from_pdb(filename)
    distance = pdist(xyz[0, ...])
    return np.amax(distance)


def get_xyz_from_pdb(filename=None):
    """Get particle coordinates from .pdb file.

    Parameters
    ----------
    filename : str
        Relative path to file containing topological information of particle

    Returns
    -------
    traj_small.xyz : ndarray
        Particle coordinates from .pdb file.
    """
    traj = md.load(filename)
    atom_indices = traj.topology.select("name CA or name P")
    traj_small = traj.atom_slice(atom_indices)
    return traj_small.xyz


def micrograph2particles(
    micrograph, optics_params, detector_params, pdb_file=None, dmax=30, pad=5.0
):
    """Extract particles from given micrograph.

    Parameters
    ----------
    micrograph : ndarray
        Particle micrograph to extract from.
    optics_params : list
        List of sim parameters pertaining to microscope settings.
    detector_params : list
        List of sim parameters pertaining to detector settings.
    pdb_file : str
        Relative path to .pdb file.
    dmax : int
        Predefined maximum dimension of molecule.
    pad : int
        Amount of padding for each particle box.

    Returns
    -------
    particles : ndarray
        Picked and sliced particle data from micrograph.
    """
    fov_Lx, fov_Ly, boxsize = get_fov(
        optics_params, detector_params, pdb_file=pdb_file, dmax=dmax, pad=pad
    )
    fov_Nx = np.floor(fov_Lx / boxsize)
    fov_Ny = np.floor(fov_Ly / boxsize)
    pixel_size = (fov_Lx / micrograph.shape[1] + fov_Ly / micrograph.shape[0]) / 2.0
    n_boxsize = np.int(boxsize / pixel_size)
    x_pixels = np.int(fov_Nx * n_boxsize)
    y_pixels = np.int(fov_Ny * n_boxsize)
    data = micrograph[0:y_pixels, 0:x_pixels]
    particles = slice_and_stack(data, n_boxsize=n_boxsize)
    return particles


def slice_and_stack(data, n_boxsize=256, n_ovl=0):
    """Convert a 2D numpy array into a 3D numpy array.

    Parameters
    ----------
    data : ndarray
        2D array to stack.
    n_boxsize : int
        Boxsize.
    n_ovl : int
        Overlap.

    Returns
    -------
    data_stack : ndarray
        Array containing sliced particle data.
    """
    log = logging.getLogger()

    if n_ovl == 0:
        data_stack = blockshaped(data, n_boxsize, n_boxsize)
    else:
        n_split = math.floor((data.shape[0] - 2 * n_ovl) / n_boxsize)
        n_dilat = n_boxsize + 2 * n_ovl
        data_stack = np.zeros((n_split * n_split, n_dilat, n_dilat))
        log.info("Array dimensions: {}".format(data_stack.shape))
        i_stack = 0
        for i in np.arange(n_split):
            for j in np.arange(n_split):
                istart = i * n_boxsize
                istop = istart + n_dilat
                jstart = j * n_boxsize
                jstop = jstart + n_dilat
                rows = np.arange(istart, istop)
                columns = np.arange(jstart, jstop)
                data_tmp = data[np.ix_(rows, columns)]
                data_stack[i_stack, ...] = data_tmp[np.newaxis, ...]
                i_stack += 1

    return data_stack


def blockshaped(arr, nrows, ncols):
    """Return an array of shape (n, nrows, ncols) where n * nrows * ncols = arr.size.

    Parameters
    ----------
    arr : ndarray
        Array to reshape.
    nrows : int
        Number of rows.
    ncols : int
        Number of cols.

    Returns
    -------
    reshaped_arr : ndarray
        Reshaped input array with dimension (n, nrows, ncols)
    """
    h, _ = arr.shape
    reshaped_arr = (
        arr.reshape(h // nrows, nrows, -1, ncols)
        .swapaxes(1, 2)
        .reshape(-1, nrows, ncols)
    )
    return reshaped_arr
