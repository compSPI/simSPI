# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 02:07:02 2021

@author: NLEGENDN
"""
import coords
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

CUDA = torch.cuda.is_available()

if CUDA:
    main_dir = "/scratch/ex-kdd-1/NicolasLegendre/Cryo/Data"
else:
    main_dir = os.getcwd()+"\\Data\\"

particules = np.asarray([[1, 0, 0, 3], [0, 8, 0, 5], [0, 0, 7, 9]])
n_particules = 2000
vol_size = 64
name = '4_points_3D'
long = 2


def long(point1, point2):
    """

    Parameters
    ----------
    point1 : tuple float.
    point2 : tuple float.

    Returns
    -------
    distance between point1 and point2.

    """
    a, b, c = point1
    d, e, f = point2
    return (a-d)**2+(b-e)**2+(c-f)**2


def modif_weight(points, volume, SIZE, LONG):
    """

    Parameters
    ----------
    points : list of particules position.
    volume : ndarray volume of the molecule.
    SIZE : int size of the volume.
    LONG : int center the molecule around zero.

    Returns
    -------
    volume : ndarray volume of the molecule.

    """
    n = len(points)
    for i in range(n):
        point = points.T[i]
        for i in range(SIZE):
            for j in range(SIZE):
                for k in range(SIZE):
                    volume[i][j][k] += np.exp(-long([i/LONG-SIZE/LONG/2,
                                              j/LONG-SIZE/LONG/2, k/LONG-SIZE/LONG/2], point)/2)
    return volume


def simulate_3D_volumes(particules, n_particles, img_size, long=2):
    """

    Parameters
    ----------
    particules : ndarray volume of the molecule.
    n_particles : int number of data.
    img_size : int size of the molecule.
    long : float, optional center the molecule around zero.

    Returns
    -------
    volumes : ndarray 3D map of the molecule
    qs : dataframe describes rotations in quaternions

    """
    rots, qs = coords.uniform_rotations(n_particles)
    volumes = np.zeros((n_particles, img_size, img_size,
                       img_size))
    for idx in range(n_particles):
        if idx % (n_particles/10) == 0:
            print(idx)
        points = rots[idx].dot(particules)
        volumes[idx] = modif_weight(points, volumes[idx], img_size, long)
    qs = [np.array2string(q) for q in qs]
    qs = pd.DataFrame(qs)
    qs.columns = ['rotation_quaternion']
    return volumes, qs


def save_volume(particules, n_particules, vol_size, main_dir, name, long=2):
    """

    Parameters
    ----------
    particules : array position of particules.
    n_particules : int number of data.
    vol_size : int size of data.
    main_dir : string main directory
    name : string name 
    long : int center the molecule around 0

    Returns
    -------
    volumes : ndarray 3D map of the molecule
    labels : dataframe describes rotations in quaternions
    """
    volumes, labels = simulate_3D_volumes(
        particules, n_particules, vol_size, long)
    np.save(main_dir+name+'.npy', volumes)
    labels.to_csv(main_dir+name+'.csv')
    return volumes, labels


A = save_volume(particules, n_particules, vol_size, main_dir, name, long=2)
