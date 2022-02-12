"""Unit test for multislice."""

import numpy as np
import raster_geometry
import torch

from simSPI import multislice, transfer, transforms


def test_exit_wave_to_image():
    """High dose, no ctf/dqe/ntf."""
    n_pixels = 64
    sphere = raster_geometry.sphere(
        [n_pixels, n_pixels, n_pixels], radius=n_pixels // 8, position=0.25
    ).astype(np.float32)
    ones = np.ones((n_pixels, n_pixels))
    exit_wave = sphere.sum(-1)
    exit_wave_torch = torch.tensor(exit_wave).reshape(1, 1, n_pixels, n_pixels)
    exit_wave_f_torch = transforms.primal_to_fourier_2D(exit_wave_torch)
    exit_wave_f = exit_wave_f_torch.detach().numpy().reshape(n_pixels, n_pixels)
    high_dose = 1e6 * exit_wave.max()

    i, shot_noise_sample, i0_dqe, i0 = multislice.exit_wave_to_image(
        exit_wave_f=exit_wave_f,
        complex_ctf=ones,
        dose=high_dose,
        noise_bg=0,
        dqe=ones,
        ntf=ones,
    )

    assert i.shape == (n_pixels, n_pixels)
    assert shot_noise_sample.shape == (n_pixels, n_pixels)
    assert i0_dqe.shape == (n_pixels, n_pixels)
    assert i0.shape == (n_pixels, n_pixels)
    assert np.allclose(i / high_dose, exit_wave, atol=1e-2)


def test_apply_poisson_shot_noise_sample():
    """Poisson noise high vs low."""
    n_pixels = 64
    signal = np.ones((n_pixels, n_pixels))
    # hi noise
    dose_hin = 0.1
    noise_bg_hin = 1
    shot_noise_sample_hin = multislice.apply_poisson_shot_noise_sample(
        signal=signal, dose=dose_hin, noise_bg=noise_bg_hin
    )
    assert shot_noise_sample_hin.shape == (n_pixels, n_pixels)

    # hi dose
    dose_hid = 1
    noise_bg_hid = 0.1
    shot_noise_sample_hid = multislice.apply_poisson_shot_noise_sample(
        signal=signal, dose=dose_hid, noise_bg=noise_bg_hid
    )

    diff_hin = np.linalg.norm(
        signal - (shot_noise_sample_hin / dose_hin - noise_bg_hin)
    )
    diff_hid = np.linalg.norm(
        signal - (shot_noise_sample_hid / dose_hid - noise_bg_hid)
    )
    assert diff_hid < diff_hin


def test_apply_complex_ctf_to_exit_wave():
    """Test apply_complex_ctf_to_exit_wave."""
    n_random = np.random.uniform(low=50, high=100)
    n_pixels = int((n_random // 2) * 2)  # even N
    assert np.isclose(n_pixels % 2, 0), "must be even for test to work"
    i0 = multislice.apply_complex_ctf_to_exit_wave(
        exit_wave_f=np.ones((n_pixels, n_pixels)),
        complex_ctf=np.ones((n_pixels, n_pixels)),
    )
    assert i0.shape == (n_pixels, n_pixels)


def test_apply_dqe():
    """Test apply_dqe."""
    n_random = np.random.uniform(low=50, high=100)
    n_pixels = int(2 * (n_random // 2))
    assert np.isclose(n_pixels % 2, 0), "must be even for test to work"
    ones = np.ones((n_pixels, n_pixels))
    freq_A_2d = transfer.ctf_freqs(n_pixels, dim=2)[0]
    mtf_const = 1.5
    mtf2 = (np.sinc(freq_A_2d * mtf_const)) ** 2
    ntf2 = np.sinc(freq_A_2d) ** 2
    dqe = mtf2 / ntf2
    i0_dqe = multislice.apply_dqe(ones, dqe)
    assert i0_dqe.shape == (n_pixels, n_pixels)


def test_apply_ntf():
    """Test apply_ntf."""
    n_random = np.random.uniform(low=50, high=100)
    n_pixels = int(2 * (n_random // 2))
    assert np.isclose(n_pixels % 2, 0), "must be even for test to work"
    ones = np.ones((n_pixels, n_pixels))
    freq_A_2d = transfer.ctf_freqs(n_pixels, dim=2)[0]
    ntf = np.sinc(freq_A_2d)
    i = multislice.apply_ntf(shot_noise_sample=ones, ntf=ntf)
    assert i.shape == (n_pixels, n_pixels)
