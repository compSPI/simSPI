"""Test function for projector module."""

import numpy as np
import torch

from simSPI.linear_simulator.projector import Projector


class AttrDict(dict):
    """Class to convert a dictionary to a class.

    Parameters
    ----------
    dict: dictionary

    """

    def __init__(self, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def init_data(path):
    """Load data for the test functions.

    Loads .npy file from a path and converts its config dictionary into a class

    Parameters
    ----------
    path: str

    Returns
    -------
    saved_data: dict
        loaded dictionary
    config: object
    """
    saved_data = np.load(path, allow_pickle=True).item()
    if "config_dict" in saved_data:
        config_dict = saved_data["config_dict"]
    else:
        config_dict = {}
    config_dict["space"] = "real"
    config = AttrDict(config_dict)
    return saved_data, config


def normalized_mse(a, b):
    """Return mean square error.

    Calclulates mean square error between two inputs normalized
    by the norm of the first input.

    Parameters
    ----------
    a: torch.Tensor
    b: torch.Tensor

    Returns
    -------
    out: torch.Tensor
        normalized mse
    """
    return (a - b).pow(2).sum().sqrt() / a.pow(2).sum().sqrt()


def test_projector_real():
    """Test accuracy of projector function."""
    path = "tests/data/projector_data.npy"

    saved_data, config = init_data(path)
    config.space = "real"
    rot_params = saved_data["rot_params"]
    projector = Projector(config)
    projector.vol = saved_data["volume"]

    out = projector(rot_params)
    error = normalized_mse(saved_data["projector_output"], out).item()
    assert (error < 0.01) == 1


def test_projector_fourier_axis_aligned():
    """Test accuracy of projector function - simplified."""
    path = "tests/data/projector_data.npy"

    saved_data, config = init_data(path)
    config.space = "fourier"
    rot_params = saved_data["rot_params"]
    rot_params["rotmat"].data[0]=torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
    rot_params["rotmat"].data[1]=torch.tensor([[0,1,0],[-1,0,0],[0,0,1]])
    rot_params["rotmat"].data[2]=torch.tensor([[0,0,1],[0,1,0],[-1,0,0]])
    #print(rot_params["rotmat"])
    config["side_len"]=4
    projector = Projector(config)
    #print(saved_data["volume"])
    vol = torch.fft.fftshift(saved_data["volume"],dim=[-3,-2,-1])
    #vol_shift = vol
    #vol_shift = vol
    vol = torch.zeros((4,4,4),dtype=torch.float32)
    vol[0,1,0]=1
    vol_shift = vol
    
    #projector.vol = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(vol_shift)))
    projector.vol = torch.fft.ifftshift(torch.fft.fftn(vol_shift),dim=[-3,-2,-1])
    
    #print(projector.vol.shape)
    sz = projector.vol.shape[0]
    #print("vol_coords", projector.vol_coords)
    #print("Vol Center", projector.vol[sz//2,sz//2,sz//2])
    
    out = projector(rot_params)
    out_r = torch.fft.ifft2(out,dim=(2,3))
    #fft_proj_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(saved_data["projector_output"],dim=(2,3))),dim=(2,3))
    #fft_proj_out = (torch.fft.fft2(torch.fft.fftshift(saved_data["projector_output"],dim=(2,3))))
    print("FOURIER2",out.dtype)
    
    print(vol_shift.shape)
    
    sum1 = vol_shift.sum(axis=1)
    sum2 = vol_shift.sum(axis=2)
    sum0 = vol_shift.sum(axis=0)
    
    print("sum0",sum0)
    print("sum1",sum1)
    print("sum2",sum2)
    print(out_r[0])
    print(out_r[1])
    print(out_r[2])
    
    #print( sum1 )
    #print( (sum0 - out_r[0]).numpy() )
    #print( (sum1 - out_r[0]).numpy() 
    print( (sum0 - out_r[0]).abs().pow(2).sum().numpy()  )
    print( (sum1 - out_r[1]).abs().pow(2).sum().numpy() )
    print( (sum2 - out_r[2]).abs().pow(2).sum().numpy() )
    assert( (sum0 - out_r[0]).abs().pow(2).sum().numpy() < 1e-12 )
    assert( (sum1 - out_r[1]).abs().pow(2).sum().numpy() < 1e-12 )
    assert( (sum2 - out_r[2]).abs().pow(2).sum().numpy() < 1e-12 )
    
    assert( False )
    

def test_projector_fourier():
    """Test accuracy of projector function."""
    path = "tests/data/projector_data.npy"
    
    return 
    saved_data, config = init_data(path)
    config.space = "fourier"
    rot_params = saved_data["rot_params"]
    #rot_params["rotmat"].data[0]=torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
    #print(rot_params["rotmat"])
    projector = Projector(config)
    #print(saved_data["volume"])
    projector.vol = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(saved_data["volume"],dim=[-3,-2,-1])),dim=[-3,-2,-1])
    
    #print(projector.vol.shape)
    sz = projector.vol.shape[0]
    #print("vol_coords", projector.vol_coords)
    #print("Vol Center", projector.vol[sz//2,sz//2,sz//2])
    
    out = projector(rot_params)
    #fft_proj_out = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(saved_data["projector_output"],dim=(2,3))),dim=(2,3))
    fft_proj_out = (torch.fft.fft2(torch.fft.fftshift(saved_data["projector_output"],dim=(2,3))))
    print(out.dtype)
    print(fft_proj_out[0,0,0])
    print(out[0,0,0])
    print("ratio", sz, (fft_proj_out.real/out.real).median())
    print("ratio", sz, 1/(fft_proj_out.real[0,0,0,0]/out.real[0,0,0,0]))
    print("ratio", sz, 1/(fft_proj_out.real[:,0,0,0]/out.real[:,0,0,0]))
    print(out.shape[0],np.sqrt(out.shape[0]),1.0/np.sqrt(out.shape[0]))
    error_r = normalized_mse(fft_proj_out.real, out.real).item()
    error_i = normalized_mse(fft_proj_out.imag, out.imag).item()
    assert (error_r < 0.01) == 1
    assert (error_i < 0.01) == 1





    

