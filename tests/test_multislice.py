import numpy as np
import raster_geometry
from ioSPI import transfer
from ioSPI import fourier
from simSPI import exit_wave_to_image
from simSPI import apply_poisson_shot_noise_sample
from simSPI import apply_complex_ctf_to_exit_wave
from simSPI import apply_dqe
from simSPI import apply_ntf

def test_exit_wave_to_image():
  """ high dose, no ctf/dqe/ntf
  """
  N = 64
  sphere = raster_geometry.sphere([N,N,N],radius=N//8,position=0.25).astype(np.float32)
  ones = np.ones((N,N))
  zeros = np.zeros((N,N))
  exit_wave = sphere.sum(-1)
  exit_wave_f = fourier.do_fft(exit_wave,d=2)
  dose = 1e9*exit_wave.max()

  i, shot_noise_sample, i0_dqe, i0 = exit_wave_to_image(
      exit_wave_f=exit_wave_f,
      complex_ctf=ones,
      dose=dose,
      noise_bg=0,
      dqe=ones,
      ntf=ones
      )

  assert np.allclose(i/dose,exit_wave,atol=1e-4)

test_exit_wave_to_image()

def test_apply_poisson_shot_noise_sample():
  """ poisson noise high vs low
  """
  # high noise
  N=64
  signal = np.ones((N,N))
  dose_highnoise = 0.1
  noise_bg_highnoise = 1
  shot_noise_sample_highnoise = apply_poisson_shot_noise_sample(signal=signal,
                                                            dose=dose_highnoise, 
                                                            noise_bg=noise_bg_highnoise)
  assert shot_noise_sample_highnoise.shape == (N,N)
  
  dose_highdose = 1
  noise_bg_highdose = 0.1
  shot_noise_sample_highdose = apply_poisson_shot_noise_sample(signal=signal,
                                                            dose=dose_highdose, 
                                                            noise_bg=noise_bg_highdose)
  


  diff_highnoise = np.linalg.norm(signal - (shot_noise_sample_highnoise/dose_highnoise - noise_bg_highnoise))
  diff_highdose = np.linalg.norm(signal - (shot_noise_sample_highdose/dose_highdose - noise_bg_highdose))
  assert  diff_highdose < diff_highnoise

test_apply_poisson_shot_noise_sample()

def test_apply_complex_ctf_to_exit_wave():
  N_random = np.random.uniform(low=50,high=100)
  N = int(2*(N_random//2)) # even N
  i0 = apply_complex_ctf_to_exit_wave(exit_wave_f=np.ones((N,N)),
                                 complex_ctf = np.ones((N,N))
                                 )
  assert i0.shape == (N,N)
test_apply_complex_ctf_to_exit_wave()

def test_apply_dqe():
  N_random = np.random.uniform(low=50,high=100)
  N = int(2*(N_random//2))

  i0 = raster_geometry.sphere([N,N,N],radius=N//8,position=0.25).astype(np.float32).sum(-1)
  ones = np.ones((N,N))
  zeros = np.zeros((N,N))
  i0_f = fourier.do_fft(i0,d=2)
  
  freq_A_2d,angles_rad = transfer.ctf_freqs(N,d=2)

  mtf_const = 1.5
  mtf2 = (np.sinc(freq_A_2d*mtf_const))**2
  ntf2 = np.sinc(freq_A_2d)**2
  dqe = mtf2/ntf2
  i0_dqe = apply_dqe(i0_f,dqe)
  assert i0_dqe.shape == (N,N)
test_apply_dqe()

def test_apply_ntf():
  N_random = np.random.uniform(low=50,high=100)
  N = int(2*(N_random//2))

  i0 = raster_geometry.sphere([N,N,N],radius=N//8,position=0.25).astype(np.float32).sum(-1)
  ones = np.ones((N,N))
  zeros = np.zeros((N,N))
  i0_f = fourier.do_fft(i0,d=2)
  
  freq_A_2d,angles_rad = transfer.ctf_freqs(N,d=2)

  ntf = np.sinc(freq_A_2d)

  i = apply_ntf(shot_noise_sample=ones,ntf=ntf)
  assert i.shape == (N,N)
test_apply_ntf()