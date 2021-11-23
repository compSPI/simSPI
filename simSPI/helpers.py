"""Helper functions for tem.py."""
import math
import os

import h5py
import mdtraj as md
import mrcfile
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import special_ortho_group


def mrc2data(mrc_file=None):
    """Return micrograph from an input .mrc file.

    Parameters
    ----------
    mrc_file : str
        File name for .mrc file to turn into micrograph
    """
    if mrc_file is not None:
        with mrcfile.open(mrc_file, "r", permissive=True) as mrc:
            micrograph = mrc.data
        if micrograph is not None:
            if len(micrograph.shape) == 2:
                micrograph = micrograph[np.newaxis, ...]
        else:
            print("Warning! Data in {} is None...".format(mrc_file))
        return micrograph
    return None


def data_and_dic_2hdf5(data, h5_file, dic=None):
    """Save a dictionary which might contain different data types to a .hdf5 file.

    Parameters
    ----------
    data : Object
        Data to store.
    h5_file : str
        Relative path to write .h5 file to.
    dic : dict
        Dictionary containing fields to save.
    """
    if dic is None:
        dic = {}
    dic["data"] = data
    with h5py.File(h5_file, "w") as file:
        recursively_save_dict_contents_to_group(file, "/", dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """Recursively save dictionary contents to group.

    Parameters
    ----------
    h5file : File
        .hdf5 file to write to.
    path : str
        Relative path to save dictionary contents.
    dic : dict
        Dictionary containing data.
    """
    for k, v in dic.items():
        if isinstance(v, (np.ndarray, np.int64, np.float64, int, float, str, bytes)):
            h5file[path + k] = v
        elif isinstance(v, type(None)):
            h5file[path + k] = str("None")
        elif isinstance(v, dict):
            recursively_save_dict_contents_to_group(h5file, path + k + "/", v)
        else:
            raise ValueError("Cannot save %s type" % type(v))


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
    #
    return fov_Lx, fov_Ly, boxsize


def get_dmax(filename=None):
    """Get maximmum dimension of particle.

    Parameters
    ----------
    filename : str
        Relative path to file containing topological information of particle
    """
    if filename is not None:
        xyz = get_xyz_from_pdb(filename)
        distance = pdist(xyz[0, ...])
        return np.amax(distance)
    return None


def get_xyz_from_pdb(filename=None):
    """Get particle coordinates from .pdb file.

    Parameters
    ----------
    filename : str
        Relative path to file containing topological information of particle
    """
    if filename is not None:
        traj = md.load(filename)
        atom_indices = traj.topology.select("name CA or name P")
        traj_small = traj.atom_slice(atom_indices)
        return traj_small.xyz
    return None


def write_crd_file(
    numpart,
    xrange=np.arange(-100, 110, 10),
    yrange=np.arange(-100, 110, 10),
    crd_file="crd.txt",
):
    """Write particle data to .crd file.

    Parameters
    ----------
    numpart : int
        Number of particles.
    xrange : ndarray
        Valid horizontal range for display.
    yrange : ndarray
        Valid vertical range for display.
    crd_file : str
        Relative path to output .crd file.
    """
    if os.path.exists(crd_file):
        print(crd_file + " already exists.")
    else:
        rotlist = get_rotlist(numpart)
        with open(crd_file, "w") as crd:
            crd.write("# File created by TEM-simulator, version 1.3.\n")
            crd.write("{numpart}  6\n".format(numpart=numpart))
            crd.write(
                "#            \
                x             \
                y             \
                z             \
                phi           \
                theta         \
                psi  \n"
            )
            i = 0
            for y in yrange:
                for x in xrange:
                    if i == int(numpart):
                        break
                    crd_table = {
                        "x": x,
                        "y": y,
                        "z": 0,
                        "phi": rotlist[i][0],
                        "theta": rotlist[i][1],
                        "psi": rotlist[i][2],
                    }
                    crd.write(
                        "{0[x]:14.4f} \
                        {0[y]:14.4f} \
                        {0[z]:14.4f} \
                        {0[phi]:14.4f} \
                        {0[theta]:14.4f} \
                        {0[psi]:14.4f}\n".format(
                            crd_table
                        )
                    )
                    i += 1


def get_rotlist(numpart):
    """Return a rotation list containing Euler angles.

    Parameters
    ----------
    numpart : int
        Number of particles.
    """
    rotlist = []
    for x in range(0, numpart + 1):
        x = special_ortho_group.rvs(3)
        y = rotation_matrix_to_euler_angles(x)
        rotlist.append(y)
    return rotlist


def rotation_matrix_to_euler_angles(R):
    """Compute Euler angles given a rotation matrix.

    Parameters
    ----------
    R : ndarray
        Rotation matrix.
    """
    if not is_rotation_matrix(R):
        raise ValueError()

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    x = rad2deg(x)
    y = rad2deg(y)
    z = rad2deg(z)
    return np.array([x, y, z])


def rad2deg(x):
    """Take a value x in radians and convert to degrees.

    Parameters
    ----------
    x : float
        Degree value to convert to degrees.
    """
    return (x * 180) / np.pi


def is_rotation_matrix(matrix):
    """Determine whether a given matrix is a valid rotation matrix.

    Parameters
    ----------
    matrix : ndarray
        Matrix to check.
    """
    transposed_matrix = np.transpose(matrix)
    identity_matrix = np.identity(3, dtype=matrix.dtype)
    n = np.linalg.norm(identity_matrix - np.dot(transposed_matrix, matrix))
    return n < 1e-6
