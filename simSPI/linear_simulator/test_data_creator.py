
import os,sys
sys.path.append(os.getcwd()+"/")
import torch
import numpy as np
from simSPI.linear_simulator.modules import Projector, CTF, Shift, Noise
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import mrcfile
from pytorch3d.transforms import random_rotations
from simSPI.linear_simulator.params_utils import  primal_to_fourier_2D, fourier_to_primal_2D

#=====================
class config:
    sidelen=32
    chunks=4
    kV = 300
    pixel_size = 0.8
    cs=2.7
    amplitude_contrast=0.1
    ctf_size=32
    bfactor=0
    valueNyquist=0.1
    noise=False
    noise_sigma=0
    noise_distribution="gaussian"
    input_volume_path=''
rotmat=random_rotations(config.chunks)

defocusU = torch.Tensor([1.1, 2.1, 1.5, 1.7])[:,None,None,None]
defocusV = torch.Tensor([1, 2.2, 1.6, 1.8])[:,None,None,None]
defocusAngle = np.pi * torch.Tensor([0.1, 0.5, 0.8, 1])[:,None,None,None]
shiftX=torch.Tensor([4, 5.5, -3.2, -6])
shiftY=torch.Tensor([6, -4.5, -4, 0])
rot_params={"rotmat":rotmat}
ctf_params = {"defocusU": defocusU, "defocusV": defocusV, "defocusAngle": defocusAngle}
shift_params={"shiftX":shiftX, "shiftY":shiftY}

L=config.sidelen//2
l=config.sidelen//8
volume=torch.zeros([config.sidelen]*3 )
volume[L-l:L+l, L-l:L+l, L-l:L+l]=1
#=====================


P=Projector(config)
CTF=CTF(config)
Shift=Shift(config)
Noise=Noise(config)
P.vol=volume


projection=P(rot_params)
f_projection=primal_to_fourier_2D(projection)
h_fourier=CTF.get_ctf(ctf_params)
ctf_output=CTF(f_projection,ctf_params )
shift_output= Shift(ctf_output, shift_params)
noise_input=fourier_to_primal_2D(shift_output)
final_output=Noise(noise_input)

config_dict= {"sidelen":config.sidelen, "chunks":config.chunks,
              "kV": config.kV, "pixel_size":config.pixel_size,
              "cs":config.cs,"amplitude_contrast": config.amplitude_contrast,
              "ctf_size":config.ctf_size,
              "bfactor": config.bfactor,
              "valueNyquist": config.valueNyquist,
              "noise": config.noise,
              "noise_sigma":config.noise_sigma,
              "noise_distribution":config.noise_distribution,
              "input_volume_path":config.input_volume_path
              }
saved_data={ "rot_params":rot_params,
             "volume":volume,
             "projector_output":projection,
             "config_dict": config_dict,
             "ctf_params": ctf_params,
             "ctf_fourier": h_fourier,
             "ctf_output": ctf_output,
             "shift_params": shift_params,
             "shift_output":shift_output,
             "noise_input": noise_input,
             "final_output":final_output
            }

np.save('simSPI/tests_simple_simulator/tests_data.npy', saved_data)

