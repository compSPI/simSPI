"""Helper functions for tem.py to write particle data to .crd file."""
import logging
import math
import os

import numpy as np
from scipy.stats import special_ortho_group


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
        Range of particle center x-coordinates to write.
    yrange : ndarray
        Range of particle center y-coordinates to write.
    crd_file : str
        Relative path to output .crd file.
    """
    log = logging.getLogger()

    if os.path.exists(crd_file):
        log.info(crd_file + " already exists.")
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


def rotation_matrix_to_euler_angles(mat):
    """Compute Euler angles given a rotation matrix.

    Parameters
    ----------
    mat : ndarray
        Matrix to compute Euler angles from.
    """
    if not is_rotation_matrix(mat):
        raise ValueError()

    sy = math.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(mat[2, 1], mat[2, 2])
        y = math.atan2(-mat[2, 0], sy)
        z = math.atan2(mat[1, 0], mat[0, 0])
    else:
        x = math.atan2(-mat[1, 2], mat[1, 1])
        y = math.atan2(-mat[2, 0], sy)
        z = 0

    x = np.rad2deg(x)
    y = np.rad2deg(y)
    z = np.rad2deg(z)

    return np.array([x, y, z])


def is_rotation_matrix(matrix):
    """Determine whether a given matrix is a valid rotation matrix.

    Parameters
    ----------
    matrix : ndarray
        Matrix to check if it is a valid rotation matrix.
    """
    is_orthogonal = np.allclose(np.dot(matrix, matrix.T), np.identity(3))
    is_det_one = np.isclose(np.linalg.det(matrix), 1)
    return is_orthogonal and is_det_one
