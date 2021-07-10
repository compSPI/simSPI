"""Generate 3D map of molecules."""

import numpy as np
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

SO3 = SpecialOrthogonal(n=3, point_type="vector")

PARTICULES = np.asarray([[1, 0, 0, 3], [0, 8, 0, 5], [0, 0, 7, 9]])
N_VOLUMES = 2000
VOL_SIZE = 64
NAME = "4_points_3d"
CENTER = 2


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


def modify_weight(points, volume, vol_size, center):
    """
    Fill an empty volume with particles.

    Parameters
    ----------
    points : array
        list particules position.
    volume : ndarray
        volume of the molecule.
    size : int
        size of the volume.
    center : int
        center the molecule around zero.

    Returns
    -------
    volume : ndarray
        volume of the molecule.

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
        list particules position.
    n_volumes : int
        number of data.
    img_size : int
        size of the molecule.
    center : int
        optional center the molecule around zero.

    Returns
    -------
    volumes : ndarray
        3D map of the molecule.
    qs : list
        describes rotations in quaternions.

    """
    rots, qs = uniform_rotations(n_volumes)
    volumes = np.zeros((n_volumes,) + (vol_size,) * 3)
    for idx in range(n_volumes):
        if idx % (n_volumes/10) == 0:
            print(idx)
        points = rots[idx].dot(particules)
        volumes[idx] = modify_weight(points, volumes[idx], vol_size, center)
    qs = [np.array2string(q) for q in qs]
    return volumes, qs


def save_volume(particules, n_volumes, vol_size, main_dir, name, center=2):
    """
    Save the simulate volumes and meta_data.

    Parameters
    ----------
    particules : array
        position of particules.
    n_volumes : int
        number of data.
    vol_size : int
        size of the volume.
    main_dir : string
        main directory.
    name : string
        name.
    center : int
        center the molecule around 0.

    Returns
    -------
    volumes : ndarray
        3D map of the molecule
    labels : ndarray
        describes rotations in quaternions

    Example
    -------
    >>>import numpy as np
    >>>import coords
    _ = save_volume(PARTICULES,N_VOLUMES,VOL_SIZE, dir, NAME, CENTER)
    """
    volumes, labels = simulate_volumes(
        particules, n_volumes, vol_size, center)
    np.save(main_dir + name + '_molecules.npy', volumes)
    np.save(main_dir + name + '_labels.npy', volumes)

    return volumes, labels
