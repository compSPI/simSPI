"""Helper functions for tem.py to define grid FOV for displaying particles."""
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
