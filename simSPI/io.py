"""Helper functions for tem.py processing of input and output files."""
import mrcfile
import numpy as np


def mrc2data(mrc_file):
    """Return micrograph from an input .mrc file.

    Parameters
    ----------
    mrc_file : str
        File name for .mrc file to turn into micrograph
    """
    with mrcfile.open(mrc_file, "r", permissive=True) as mrc:
        micrograph = mrc.data
    if micrograph is not None:
        if len(micrograph.shape) == 2:
            micrograph = micrograph[np.newaxis, ...]
    return micrograph


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
