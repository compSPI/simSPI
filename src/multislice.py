from simSPI import fourier
import numpy as np

def apply_complex_ctf_to_exit_wave(exit_wave_f,complex_ctf):
  """Apply complex contrast transfer function to multisclice exit wave.

  Fourier based convolution of exit wave and ctf, and collapse of wave function.

  Parameters
  ----------
  exit_wave_f : numpy.ndarray, shape (N,N)
    exit wave array.
  complex_ctf : numpy.ndarray, shape (N,N)
    complex ctf array.

  Returns
  -------
  i0 : numpy.ndarray, shape (N,N)
    exit wave ctf convolution (in real space)
  """
  i0 = np.abs(fourier.do_ifft(exit_wave_f*complex_ctf,d=2,only_real=False))
  return i0

def apply_dqe(i0_f,dqe):
  """Convolution with detector's DQE.

  Convolution of (ctf applied) exit wave with sqrt of the detective quantum efficiency

  Parameters
  ----------
  i0_f : numpy.ndarray, shape (N,N)
    image with ctf applied.
  dqe : numpy.ndarray, shape (N,N)
    detective quantum efficiency (in 2D).

  Returns
  -------
  i0_dqe : numpy.ndarray, shape (N,N)
    exit wave with dqe applied (in real space).
  """
  i0_dqe = fourier.do_ifft(i0_f*np.sqrt(dqe),d=2)
  return i0_dqe

def apply_poisson_shot_noise_sample(signal, dose, noise_bg=0):
  """Poisson shot noise.

  Poisson sampling of exit wave with ctf and dqe applied, to simulate shot noise

  Parameters
  ----------
  signal : numpy.ndarray, shape (N,N)
    input signal, (e.g. exit wave with ctf and dqe applied).
  dose : float
    multiplicative scaling factor to simulate electron dose.
  noise_bg : float
    additive factor to simulate dark current noise.

  Returns
  -------
  shot_noise_sample : numpy.ndarray, shape (N,N)
    Poisson sampled signal.
  """
  shot_noise_sample = np.random.poisson(dose*signal + noise_bg)
  return shot_noise_sample

def apply_ntf(shot_noise_sample,ntf):
  """Convolution with detector's NTF.

  Convolution of (ctf and dqe applied) exit wave with the noise transfer function.

  Parameters
  ----------
  shot_noise_sample : numpy.ndarray, shape (N,N)
    image with ctf applied.
  ntf : numpy.ndarray, shape (N,N)
    noise transfer function (in 2D).

  Returns
  -------
  i : numpy.ndarray, shape (N,N)
    exit wave with ntf applied (in real space).
  """
  i = fourier.do_ifft(fourier.do_fft(shot_noise_sample,d=2)*ntf,d=2)
  return i

def exit_wave_to_image(exit_wave_f,complex_ctf,dose,noise_bg,dqe,ntf):
  """Exit wave to image (ctf, detector dqe/ntf, and poisson shot noise)

  Convolution of (ctf, detector dqe/ntf,) with exit wave. 
  Incorporates Poisson shot noise and the collapse of the wave function.
  Forward model eqs 5-7 in 
    Vulović, M., Ravelli, R. B. G., van Vliet, L. J., Koster, A. J., Lazić, I., Lücken, U., … Rieger, B. (2013). 
    Image formation modeling in cryo-electron microscopy. 
    Journal of Structural Biology, 183(1), 19–32. http://doi.org/10.1016/j.jsb.2013.05.008

  Parameters
  ----------
  exit_wave_f : numpy.ndarray, shape (N,N)
    Fourier transform of exit wave.
  complex_ctf : numpy.ndarray, shape (N,N)
    complex ctf array.
  dose : float
    multiplicative scaling factor to simulate electron dose.
  noise_bg : float
    additive factor to simulate dark current noise.
  dqe : numpy.ndarray, shape (N,N)
    detective quantum efficiency (in 2D).
  ntf : numpy.ndarray, shape (N,N)
    noise transfer function (in 2D).

  Returns
  -------
  i : numpy.ndarray, shape (N,N)
    exit wave with ntf applied (in real space)
  """

  assert exit_wave_f.ndim == 2
  assert complex_ctf.ndim == 2
  assert dqe.ndim == 2
  assert ntf.ndim == 2

  i0 = apply_complex_ctf_to_exit_wave(exit_wave_f,complex_ctf)
  i0_f = fourier.do_fft(i0,d=2)
  i0_dqe = apply_dqe(i0_f,dqe)
  shot_noise_sample = apply_poisson_shot_noise_sample(i0_dqe, dose, noise_bg)
  i = apply_ntf(shot_noise_sample,ntf)

  return i, shot_noise_sample, i0_dqe, i0