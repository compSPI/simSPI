""""""

import numpy as np
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SO3 = SpecialOrthogonal(n=3, point_type="vector")


def coords_n_by_d(coords_1d=None, N=None, d=3):
    """


    Parameters
    ----------
    coords_1d : TYPE, optional
        DESCRIPTION. The default is None.
    N : TYPE, optional
        DESCRIPTION. The default is None.
    d : TYPE, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    coords : TYPE
        DESCRIPTION.

    """
    if coords_1d is not None and N is None:
        pass
    elif coords_1d is None and N is not None:
        assert N % 2 == 0, "N must be even"
        coords_1d = np.arange(-N // 2, N // 2)
    else:
        assert False, "use either N or coords_1d, not both"

    if d == 2:
        X = np.meshgrid(coords_1d, coords_1d)
    elif d == 3:
        X = np.meshgrid(coords_1d, coords_1d, coords_1d)
    coords = np.zeros((X[0].size, d))
    for di in range(d):
        coords[:, di] = X[di].flatten()
    if d == 3:
        coords[:, [0, 1]] = coords[:, [1, 0]]

    return coords


def deg_to_rad(deg):
    """Convert degree into radian,

    Parameters
    ----------
    deg : float
        angle in degree.

    Returns
    -------
    flaot
        angle in radian.

    """
    return deg * np.pi / 180


def get_random_quat(num_pts):
    """Generate a list of quaternions by using uniform distribution.

    Parameters
    ----------
    num_pts : int
        number of quaternions to return.

    Returns
    -------
    array
        list of simulated quaternions of shape [num_pts,4].

    """
    u = np.random.rand(3, num_pts)
    u1, u2, u3 = [u[x] for x in range(3)]

    quat = np.zeros((4, num_pts))
    quat[0] = np.sqrt(1 - u1) * np.sin(np.pi * u2 * 2)
    quat[1] = np.sqrt(1 - u1) * np.cos(np.pi * u2 * 2)
    quat[2] = np.sqrt(u1) * np.sin(np.pi * u3 * 2)
    quat[3] = np.sqrt(u1) * np.cos(np.pi * u3 * 2)

    return np.transpose(quat)


def uniform_rotations(num_pts):
    """Generate ratation as quaternion and convert them as rotation matrix.

    Parameters
    ----------
    num : int
        number of quaternion to return.

    Returns
    -------
    rots : ndarray
        matrix of rotation of the simulated quaternions of shape (num_pts,3,3)
    qs : array
        list of simulated quaternions of shape [num_pts,4].

    """
    qs = get_random_quat(num_pts)
    rots = SO3.matrix_from_quaternion(qs)  # num,3,3
    return (rots, qs)
