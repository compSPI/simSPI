"""Helper functions for tem.py to define grid FOV for displaying particles."""
import math

import mdtraj as md
import numpy as np
from scipy.spatial.distance import pdist


def define_grid_in_fov(
    optics_params, detector_params, pdb_file=None, dmax=None, pad=1.0
):
    """Define field of view for graph.

    Parameters
    ----------
    optic_params : list
        List of sim parameters pertaining to microscope settings.
    detector_params : dict
        List of sim parameters pertaining to detector settings.
    pdb_file : str
        Relative path to write .pdb file to.
    dmax : int
        Maximum dimension of molecule.
    pad : int
        Amount of padding.
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
    optic_params : list
        List of sim parameters pertaining to microscope settings.
    detector_params : dict
        List of sim parameters pertaining to detector settings.
    pdb_file : str
        Relative path to .pdb file output.
    dmax : int
        Maximum dimension of molecule
    pad : int
        Amount of padding.
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
    """
    traj = md.load(filename)
    atom_indices = traj.topology.select("name CA or name P")
    traj_small = traj.atom_slice(atom_indices)
    return traj_small.xyz


def microgaph2particles(
    micrograph, optics_params, detector_params, pdb_file=None, dmax=30, pad=5.0
):
    """Extract particles from given micrograph.

    Parameters
    ----------
    micrograph : ndarray
        Particle micrograph to extract from.
    optic_params : list
        List of sim parameters pertaining to microscope settings.
    detector_params : dict
        List of sim parameters pertaining to detector settings.
    pdb_file : str
        Relative path to write .pdb file to.
    dmax : int
        Maximum dimension of molecule.
    pad : int
        Amount of padding.
    """
    fov_Lx, fov_Ly, boxsize = get_fov(
        optics_params, detector_params, pdb_file=pdb_file, dmax=dmax, pad=pad
    )
    fov_Nx = np.floor(fov_Lx / boxsize)
    fov_Ny = np.floor(fov_Ly / boxsize)
    pixel_size = (fov_Lx / micrograph.shape[1] + fov_Ly / micrograph.shape[0]) / 2.0
    n_boxsize = np.int(boxsize / pixel_size)
    Nx = np.int(fov_Nx * n_boxsize)
    Ny = np.int(fov_Ny * n_boxsize)
    data = micrograph[0:Ny, 0:Nx]
    particles = slicenstack(data, n_boxsize=n_boxsize)
    return particles


def slicenstack(data, n_boxsize=256, n_ovl=0):
    """Convert a 2D numpy array into a 3D numpy array.

    Parameters
    ----------
    data : ndarray
        2D array to stack.
    n_boxsize : int
        Boxsize.
    n_ovl : int
        Overlap.
    """
    if n_ovl == 0:
        data_stack = blockshaped(data, n_boxsize, n_boxsize)
    else:
        n_split = math.floor((data.shape[0] - 2 * n_ovl) / (n_boxsize))
        n_dilat = n_boxsize + 2 * n_ovl
        data_stack = np.zeros((n_split * n_split, n_dilat, n_dilat))
        print("Array dimensions: ", data_stack.shape)
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
    """
    h, _ = arr.shape
    return (
        arr.reshape(h // nrows, nrows, -1, ncols)
        .swapaxes(1, 2)
        .reshape(-1, nrows, ncols)
    )
