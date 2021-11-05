"""Unit test for multislice."""

import numpy as np
import raster_geometry
from ioSPI import fourier

from simSPI import multislice, transfer


def test_exit_wave_to_image():
    """High dose, no ctf/dqe/ntf."""
    num_pix = 64
    sphere = raster_geometry.sphere(
        [num_pix, num_pix, num_pix], radius=num_pix // 8, position=0.25
    ).astype(np.float32)
    ones = np.ones((num_pix, num_pix))
    exit_wave = sphere.sum(-1)
    exit_wave_f = fourier.do_fft(exit_wave, dim=2)
    high_dose = 1e9 * exit_wave.max()

    i, shot_noise_sample, i0_dqe, i0 = multislice.exit_wave_to_image(
        exit_wave_f=exit_wave_f,
        complex_ctf=ones,
        dose=high_dose,
        noise_bg=0,
        dqe=ones,
        ntf=ones,
    )

    assert i.shape == (num_pix, num_pix)
    assert shot_noise_sample.shape == (num_pix, num_pix)
    assert i0_dqe.shape == (num_pix, num_pix)
    assert i0.shape == (num_pix, num_pix)
    assert np.allclose(i / high_dose, exit_wave, atol=1e-4)


def test_apply_poisson_shot_noise_sample():
    """Poisson noise high vs low."""
    num_pix = 64
    signal = np.ones((num_pix, num_pix))
    # hi noise
    dose_hin = 0.1
    noise_bg_hin = 1
    shot_noise_sample_hin = multislice.apply_poisson_shot_noise_sample(
        signal=signal, dose=dose_hin, noise_bg=noise_bg_hin
    )
    assert shot_noise_sample_hin.shape == (num_pix, num_pix)

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
    num_random = np.random.uniform(low=50, high=100)
    num_pix = int(2 * (num_random // 2))  # even N
    assert np.isclose(num_pix % 2, 0), "must be even for test to work"
    i0 = multislice.apply_complex_ctf_to_exit_wave(
        exit_wave_f=np.ones((num_pix, num_pix)), complex_ctf=np.ones((num_pix, num_pix))
    )
    assert i0.shape == (num_pix, num_pix)


def test_apply_dqe():
    """Test apply_dqe."""
    num_random = np.random.uniform(low=50, high=100)
    num_pix = int(2 * (num_random // 2))
    assert np.isclose(num_pix % 2, 0), "must be even for test to work"
    ones = np.ones((num_pix, num_pix))
    freq_A_2d = transfer.ctf_freqs(num_pix, dim=2)[0]
    mtf_const = 1.5
    mtf2 = (np.sinc(freq_A_2d * mtf_const)) ** 2
    ntf2 = np.sinc(freq_A_2d) ** 2
    dqe = mtf2 / ntf2
    i0_dqe = multislice.apply_dqe(ones, dqe)
    assert i0_dqe.shape == (num_pix, num_pix)


def test_apply_ntf():
    """Test apply_ntf."""
    num_random = np.random.uniform(low=50, high=100)
    num_pix = int(2 * (num_random // 2))
    assert np.isclose(num_pix % 2, 0), "must be even for test to work"
    ones = np.ones((num_pix, num_pix))
    freq_A_2d = transfer.ctf_freqs(num_pix, dim=2)[0]
    ntf = np.sinc(freq_A_2d)
    i = multislice.apply_ntf(shot_noise_sample=ones, ntf=ntf)
    assert i.shape == (num_pix, num_pix)
