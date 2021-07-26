"""Generate 3D map of molecules."""

import numpy as np
import os
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SO3 = SpecialOrthogonal(n=3, point_type="vector")

PARTICULES = np.asarray([[1, 0, 0, 3], [0, 8, 0, 5], [0, 0, 7, 9]])
N_VOLUMES = 2000
VOL_SIZE = 64
NAME = "4_points_3d"
CENTER = 2


def uniform_quaternion(num_pts):
    """Generate a list of quaternions by using uniform distribution.

    Parameters
    ----------
    num_pts : int
        Number of quaternions to return.

    Returns
    -------
    array
        List of simulated quaternions of shape [num_pts,4].

    Source
    -------
    Graphics Gems III (IBM Version)
    III.6 - UNIFORM RANDOM ROTATIONS
    David Kirk
         https://www.sciencedirect.com/science/article/pii/B9780080507552500361
    http://planning.cs.uiuc.edu/node198.html


    """
    u1, u2, u3 = np.random.rand(3, num_pts)

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
        Number of quaternion to return.

    Returns
    -------
    rots : ndarray
        Matrix of rotation of the simulated quaternions of shape (num_pts,3,3).
    qs : array
        List of simulated quaternions of shape [num_pts,4].

    """
    qs = uniform_quaternion(num_pts)
    rots = SO3.matrix_from_quaternion(qs)
    return (rots, qs)


def modify_weight(points, volume, vol_size, center):
    """
    Fill an empty volume with particles. The volume is represented as an empty
    voxel. each atome of the molecule is localized around a point with a
    gaussian probability of presence.

    Parameters
    ----------
    points : array
        List particules position.
    volume : ndarray
        Volume of the molecule.
    size : int
        Size of the volume.
    center : int
        Center the molecule around zero.

    Returns
    -------
    volume : ndarray
        Volume of the molecule.

    """
    for point in points.T:
        for i in range(vol_size):
            for j in range(vol_size):
                for k in range(vol_size):
                    volume[i][j][k] += np.exp(-np.linalg.norm(
                        [i/center-vol_size/center/2, j/center -
                         vol_size/center/2, k/center-vol_size/center/2] -
                        point)**2/2)
    return volume


def simulate_volumes(particules, n_volumes, vol_size, center=2):
    """
    Update a volume with new particles.

    Parameters
    ----------
    particules : array
        List particules position.
    n_volumes : int
        Number of data.
    img_size : int
        Size of the molecule.
    center : int
        Optional center the molecule around zero.

    Returns
    -------
    volumes : ndarray
        3D map of the molecule.
    qs : list
        Describes rotations in quaternions.

    """
    rots, qs = uniform_rotations(n_volumes)
    volumes = np.zeros((n_volumes,) + (vol_size,) * 3)
    for idx in range(n_volumes):
        if idx % (n_volumes/10) == 0:
            print(idx)
        points = rots[idx].dot(particules)
        volumes[idx] = modify_weight(points, volumes[idx], vol_size, center)
    return volumes, qs


def save_volume(particules, n_volumes, vol_size, main_dir, name, center=2):
    """
    Save the simulate volumes and meta_data.

    Parameters
    ----------
    particules : array
        Position of particules.
    n_volumes : int
        Number of data.
    vol_size : int
        Size of the volume.
    main_dir : string
        Main directory.
    name : string
        Name.
    center : int
        Center the molecule around 0.

    Returns
    -------
    volumes : ndarray
        3D map of the molecule
    labels : ndarray
        Describes rotations in quaternions

    Example
    -------
    >>>import numpy as np
    >>>import coords
    >>> _ = save_volume(PARTICULES, N_VOLUMES, VOL_SIZE, dir, NAME, CENTER)
    """
    path_molecules = os.path.join(main_dir, name + "_molecules.npy")
    path_labels = os.path.join(main_dir, name + "_labels.npy")
    volumes, labels = simulate_volumes(
        particules, n_volumes, vol_size, center)
    np.save(path_molecules, volumes)
    np.save(path_labels, volumes)

    return volumes, labels
