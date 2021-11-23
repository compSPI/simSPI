import mrcfile
import numpy as np
import mdtraj as md
import h5py

from scipy.spatial.distance import pdist


def mrc2data(mrc_file=None):
    """mrc2data
    """
    if mrc_file is not None:
        with mrcfile.open(mrc_file, 'r', permissive=True) as mrc:
            micrograph = mrc.data
        if micrograph is not None:
            if len(micrograph.shape) == 2:
                micrograph = micrograph[np.newaxis, ...]
        else:
            print('Warning! Data in {} is None...'.format(mrc_file))
        return micrograph


def data_and_dic_2hdf5(data, h5_file, dic=None):
    """data_and_dic_2hdf5
    """
    if dic is None:
        dic = {}
    dic['data'] = data
    with h5py.File(h5_file, 'w') as file:
        recursively_save_dict_contents_to_group(file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """recursively_save_dict_contents_to_group
    """
    for k, v in dic.items():
        if isinstance(v, (np.ndarray, np.int64, np.float64, int, float, str, bytes)):
            h5file[path + k] = v
        elif isinstance(v, type(None)):
            h5file[path + k] = str('None')
        elif isinstance(v, dict):
            recursively_save_dict_contents_to_group(h5file, path + k + '/', v)
        else:
            raise ValueError('Cannot save %s type' % type(v))
