"""Multislice."""

import numpy as np
import torch
import transforms


def apply_complex_ctf_to_exit_wave(exit_wave_f, complex_ctf):
    """Apply complex contrast transfer function to multisclice exit wave.

    Fourier based convolution of exit wave and ctf,
    and collapse of wave function.

    Parameters
    ----------
    exit_wave_f : numpy.ndarray, shape (n_pixels,n_pixels)
        Exit wave array.
    complex_ctf : numpy.ndarray, shape (n_pixels,n_pixels)
        Complex ctf array.

    Returns
    -------
    i0 : numpy.ndarray, shape (n_pixels,n_pixels)
        Exit wave ctf convolution (in real space).
    """
    n_pixels = exit_wave_f.shape[0]
    np_to_torch = torch.tensor(exit_wave_f * complex_ctf).reshape(
        1, 1, n_pixels, n_pixels
    )
    i0 = torch.abs(transforms.fourier_to_primal_2D(np_to_torch))
    return i0.detach().numpy().reshape(n_pixels, n_pixels)


def apply_dqe(i0_f, dqe):
    """Convolution with detector's DQE.

    Convolution of (ctf applied) exit wave
    with sqrt of the detective quantum efficiency.

    Parameters
    ----------
    i0_f : numpy.ndarray, shape (n_pixels,n_pixels)
        Image with ctf applied.
    dqe : numpy.ndarray, shape (n_pixels,n_pixels)
        Detective quantum efficiency (in 2D).

    Returns
    -------
    i0_dqe : numpy.ndarray, shape (n_pixels,n_pixels)
        Exit wave with dqe applied (in real space).
    """
    n_pixels = i0_f.shape[0]
    np_to_torch = torch.tensor(i0_f * np.sqrt(dqe)).reshape(1, 1, n_pixels, n_pixels)
    i0_dqe = transforms.fourier_to_primal_2D(np_to_torch)
    return i0_dqe.detach().numpy().reshape(n_pixels, n_pixels)


def apply_poisson_shot_noise_sample(signal, dose, noise_bg=0):
    """Poisson shot noise.

    Poisson sampling of exit wave with ctf and dqe applied,
    to simulate shot noise.

    Parameters
    ----------
    signal : numpy.ndarray, shape (n_pixels,n_pixels)
        Input signal, (e.g. exit wave with ctf and dqe applied).
    dose : float
        Multiplicative scaling factor to simulate electron dose.
    noise_bg : float
        Additive factor to simulate dark current noise.

    Returns
    -------
    shot_noise_sample : numpy.ndarray, shape (n_pixels,n_pixels)
        Poisson sampled signal.
    """
    shot_noise_sample = np.random.poisson(dose * signal + noise_bg)
    return shot_noise_sample


def apply_ntf(shot_noise_sample, ntf):
    """Convolution with detector's NTF.

    Convolution of (ctf and dqe applied) exit wave
    with the noise transfer function.

    Parameters
    ----------
    shot_noise_sample : numpy.ndarray, shape (n_pixels,n_pixels)
        Image with ctf applied.
    ntf : numpy.ndarray, shape (n_pixels,n_pixels)
        Noise transfer function (in 2D).

    Returns
    -------
    i : numpy.ndarray, shape (n_pixels,n_pixels)
        Exit wave with ntf applied (in real space).
    """
    n_pixels = shot_noise_sample.shape[0]
    shot_noise_sample_torch = torch.tensor(shot_noise_sample).reshape(
        1, 1, n_pixels, n_pixels
    )

    ntf_torch = torch.tensor(ntf).reshape(1, 1, n_pixels, n_pixels)
    i = transforms.fourier_to_primal_2D(
        transforms.primal_to_fourier_2D(shot_noise_sample_torch) * ntf_torch
    )
    return i.detach().numpy().reshape(n_pixels, n_pixels)


def exit_wave_to_image(exit_wave_f, complex_ctf, dose, noise_bg, dqe, ntf):
    """Exit wave to image (ctf, detector dqe/ntf, and poisson shot noise).

    Convolution of (ctf, detector dqe/ntf,) with exit wave.
    Incorporates Poisson shot noise and the collapse of the wave function.
    Forward model eqs 5-7 in [Vulović]_
    ..[Vulović] Vulović, M., Ravelli, R. B. G., van Vliet, L. J.,
                Koster, A. J., Lazić, I., Lücken, U., … Rieger, B.
                "Image formation modeling in cryo-electron microscopy."
                Journal of Structural Biology 183, no. 1 (2013): 19–32.
                http://doi.org/10.1016/j.jsb.2013.05.008.

    Parameters
    ----------
    exit_wave_f : numpy.ndarray, shape (n_pixels,n_pixels)
        Fourier transform of exit wave.
    complex_ctf : numpy.ndarray, shape (n_pixels,n_pixels)
        Complex ctf array.
    dose : float
        Multiplicative scaling factor to simulate electron dose.
    noise_bg : float
        Additive factor to simulate dark current noise.
    dqe : numpy.ndarray, shape (n_pixels,n_pixels)
        Detective quantum efficiency (in 2D).
    ntf : numpy.ndarray, shape (n_pixels,n_pixels)
        Noise transfer function (in 2D).

    Returns
    -------
    i : numpy.ndarray, shape (n_pixels,n_pixels)
        Exit wave with ntf applied (in real space).
    """
    i0 = apply_complex_ctf_to_exit_wave(exit_wave_f, complex_ctf)
    n_pixels = i0.shape[0]
    i0_torch = torch.tensor(i0).reshape(1, 1, n_pixels, n_pixels)
    i0_f = transforms.primal_to_fourier_2D(i0_torch)
    i0_dqe = apply_dqe(i0_f, dqe)
    shot_noise_sample = apply_poisson_shot_noise_sample(i0_dqe, dose, noise_bg)
    i = apply_ntf(shot_noise_sample, ntf)

    return i, shot_noise_sample, i0_dqe, i0
