import os, sys
import numpy as np
from matplotlib import pyplot as plt


# TEMSimulator, two options
def TEMSimulator(config):
    '''

    Parameters
    ----------
    config

    Returns
    -------
    simulation

    '''
    return simulation


def fill_parameters_dictionary():
    '''

    Returns
    -------

    '''
    return


def define_grid_in_fov(sample_dimensions, optics_params, detector_params, pdb_file=None, Dmax=None, pad=1.):
    '''

    Parameters
    ----------
    sample_dimensions
    optics_params
    detector_params
    pdb_file
    Dmax
    pad

    Returns
    -------
    x_range
    y_range
    n_particles
    '''

    return x_range, y_range, n_particles


def write_crd_file(umpart, xrange=np.arange(-100, 110, 10), yrange=np.arange(-100, 110, 10), crd_file='crd.txt',
                   pre_rotate=None):
    '''

    Parameters
    ----------
    umpart
    xrange
    yrange
    crd_file
    pre_rotate

    Returns
    -------

    '''
    return


def fill_parameters_dictionary(mrc_file=None,
                               pdb_file=None, voxel_size=0.1, particle_name='toto', particle_mrcout=None,
                               crd_file=None, sample_dimensions=[1200, 50, 150],
                               beam_params=[300, 1.3, 100, 0], dose=None,
                               optics_params=[81000, 2.7, 2.7, 50, 3.5, 0.1, 1.0, 0, 0], defocus=None,
                               optics_defocout=None,
                               detector_params=[5760, 4092, 5, 32, 'yes', 0.5, 0.0, 0.0, 1.0, 0, 0], noise=None,
                               log_file='simulator.log', seed=-1234):
    '''

    Parameters
    ----------
    mrc_file
    pdb_file
    voxel_size
    particle_name
    particle_mrcout
    crd_file
    sample_dimensions
    beam_params
    dose
    optics_params
    defocus
    optics_defocout
    detector_params
    noise
    log_file
    seed

    Returns
    -------

    '''
    return parameters_dict


def write_inp_file(inp_file='input.txt', dict_params=None):
    '''

    Parameters
    ----------
    inp_file
    dict_params

    Returns
    -------

    '''
    return


def mrc2data(mrc_file):
    '''

    Parameters
    ----------
    mrc_file

    Returns
    -------

    '''
    return data

