"""Generate 3D map of molecules."""
import os
import numpy as np
import torch
import coords


CUDA = torch.cuda.is_available()

if CUDA:
    Main_dir = "/scratch/ex-kdd-1/NicolasLegendre/Cryo/Data"
else:
    Main_dir = os.getcwd() + "\\Data\\"

Particules = np.asarray([[1, 0, 0, 3], [0, 8, 0, 5], [0, 0, 7, 9]])
N_particules = 2000
Vol_size = 64
Name = '4_points_3D'
Long = 2


def modify_weight(points, volume, size, long):
    """
    Fill an empty volume with particles.

    Parameters
    ----------
    points : array, list particules position.
    volume : ndarray, volume of the molecule.
    size : int, size of the volume.
    long : int, center the molecule around zero.

    Returns
    -------
    volume : ndarray, volume of the molecule.

    """
    for point in points.T:
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    volume[i][j][k] += np.exp(-np.linalg.norm(
                        [i/long-size/long/2, j/long - size/long/2,
                         k/long-size/long/2]-point)**2/2)
    return volume


def simulate_volumes(particules, n_volumes, img_size, long=2):
    """
    Update a volume with new particles.

    Parameters
    ----------
    particules : ndarray, volume of the molecule.
    n_particles : int, number of data.
    img_size : int, size of the molecule.
    long : float, optional center the molecule around zero.

    Returns
    -------
    volumes : ndarray, 3D map of the molecule.
    qs : dataframe, describes rotations in quaternions.

    """
    rots, qs = coords.uniform_rotations(n_volumes)
    volumes = np.zeros((n_volumes,) + (img_size,) * 3)
    for idx in range(n_volumes):
        if idx % (n_volumes/10) == 0:
            print(idx)
        points = rots[idx].dot(particules)
        volumes[idx] = modify_weight(points, volumes[idx], img_size, long)
    qs = [np.array2string(q) for q in qs]
    return volumes, qs


def save_volume(particules, n_volumes, vol_size, main_dir, name, long=2):
    """
    Save the simulate volumes and meta_data.

    Parameters
    ----------
    particules : array, position of particules.
    n_particules : int, number of data.
    vol_size : int, size of the volume.
    main_dir : string, main directory.
    name : string, name.
    long : int, center the molecule around 0.

    Returns
    -------
    volumes : ndarray 3D map of the molecule
    labels : dataframe describes rotations in quaternions
    """
    volumes, labels = simulate_volumes(
        particules, n_volumes, vol_size, long)
    np.save(main_dir + name + '_molecules.npy', volumes)
    np.save(main_dir + name + '_labels.npy', volumes)

    return volumes, labels


def main(particules=Particules, n_particules=N_particules, vol_size=Vol_size,
         main_dir=Main_dir, name=Name, long=Long):
    """
    Create and save a volume.

    Parameters
    ----------
    particules : array, position of particules.
    n_particules : int, optional number of rotations.
    vol_size : int, size of the volume.
    main_dir : string, dir where to save the volumes
    name : string, name of the volumes
    long : int, center the molecule around 0.

    Returns
    -------
    None.

    """
    _ = save_volume(particules, n_particules, vol_size,
                    main_dir, name, long)


if __name__ == "__main__":
    '''execute only if run as a script'''
    main()
