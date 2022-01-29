"""Transfer (CTF)."""

import logging

import numba as nb
import numpy as np


def ctf_freqs(n_pixels, psize=1.0, dim=2):
    """Make CTF Frequencies.

    Makes 1 or 2 D array of fourier space frequencies.
    Note that freq_pix_1d_safe ensures that we have
        [0,1/(2*n_pixels), 2/(2*n_pixels), ..., (n_pixels - 1) / (2*n_pixels)]
        otherwise np.arange gives an extra point at +0.5 for n_pixels=98
        instead of finishing at
        (n_pixels - 1) / (2*n_pixels) = (49-1)/(2*49) = 48/98 = 0.48979... < 0.5

    Parameters
    ----------
    n_pixels : int
        Number of pixels.
    dim : int, 1 or 2
        Dimension.
    psize : float
        Pixel size in Å

    Returns
    -------
    freq_1d : numpy.ndarray, shape (n_pixels//2,)
        Frequencies in 1D, with dc at left most.
        1 D goes from 0 to n_pixels//2-1
        e.g. n_pixels=6 goes from 0 1 2
        n_pixels//2 - 1 = 3 -1 = 2
        only the positive part is returned; we skip over the negative part -3 -2 -1.
    freq_mag_2d : numpy.ndarray, shape (N,N)
        Magnitude of 2D frequency vector, with -ve frequencies in left half,
        and dc and +ve frequencies in right half.
    angles_rad : numpy.ndarray, shape (N,N)
        Angle of 2D frequency vector.
    """
    if dim == 1:
        freq_pix_1d = np.arange(0, 0.5, 1 / n_pixels)
        freq_pix_1d_safe = freq_pix_1d[: n_pixels // 2]
        freq_1d = freq_pix_1d_safe * psize
        return freq_1d

    # assert d == 2
    freq_pix_1d = np.arange(-0.5, 0.5, 1 / n_pixels)
    freq_pix_1d_safe = freq_pix_1d[:n_pixels]
    x, y = np.meshgrid(freq_pix_1d_safe, freq_pix_1d_safe)
    rho = np.sqrt(x ** 2 + y ** 2)
    angles_rad = np.arctan2(y, x)
    freq_mag_2d = rho * psize
    return (freq_mag_2d, angles_rad)


def eval_ctf(s, a, def1, def2, angast=0, phase=0, kv=300, ac=0.1, cs=2.0, bf=0, lp=0):
    """
    Evaluate CTF.

    From https://github.com/asarnow/pyem/blob/master/pyem/ctf.py.


    s : np.ndarray, shape (N,N)
        Precomputed frequency magnitude grid for CTF evaluation.
    a : np.ndarray, shape (N,N)
        Precomputed frequency grid angles.
    def1 : float
        1st prinicipal underfocus distance (Å).
    def2 : float
        2nd principal underfocus distance (Å).
    angast : float
        Angle of astigmatism (deg) from x-axis to azimuth.
    phase : float
        Phase shift (deg).
    kv :  float
        Microscope acceleration potential (kV).
    ac : float
        Amplitude contrast in [0, 1.0].
    cs : float
        Spherical aberration (mm).
    bf : float
        B-factor (1/A^2), divided by 4 in exponential, lowpass positive.
        Positive B factor dampens.
    lp : float
        Hard low-pass filter (Å), should usually be Nyquist.
    """
    angast = np.deg2rad(angast)
    kv = kv * 1e3
    cs = cs * 1e7
    lamb = 12.2643247 / np.sqrt(kv * (1.0 + kv * 0.978466e-6))
    def_avg = -(def1 + def2) * 0.5
    def_dev = -(def1 - def2) * 0.5
    k1 = np.pi / 2.0 * 2 * lamb
    k2 = np.pi / 2.0 * cs * lamb ** 3
    k3 = np.sqrt(1 - ac ** 2)
    k4 = bf / 4.0  # B-factor, follows RELION convention.
    k5 = np.deg2rad(phase)  # Phase shift.
    if lp != 0:  # Hard low- or high-pass.
        s *= s <= (1.0 / lp)
    s_2 = s ** 2
    s_4 = s_2 ** 2
    dZ = def_avg + def_dev * (np.cos(2 * (a - angast)))
    gamma = (k1 * dZ * s_2) + (k2 * s_4) - k5
    ctf = -(k3 * np.sin(gamma) - ac * np.cos(gamma))
    if bf != 0:  # Enforce envelope.
        ctf *= np.exp(-k4 * s_2)
    return ctf


# separate out so can test eval_ctf
eval_ctf_jitted = nb.jit(cache=True, nopython=True, nogil=True)(eval_ctf)


def random_ctfs(
    n_pixels,
    psize,
    n_particles,
    df_min=15000,
    df_max=20000,
    df_diff_min=100,
    df_diff_max=500,
    df_ang_min=0,
    df_ang_max=360,
    kv=300,
    ac=0.1,
    cs=2.0,
    phase=0,
    bf=0,
    do_log=True,
):
    """
    Sample random CTFs.

    Sample num_particles random CTFs, where the defoci and
    angle of astigmatism are drawn from a uniform distribution.

    num_pixels : int
        Number of pixels along an axis.
    prise : float
        Pixel size (A).
    n_particles : int
        Number of random samples.
    df_min : float
        Minimum defocus (sampled from uniform rv).
    df_max : float
        Maximum defocus (sampled from uniform rv).
    df_diff_min : float
        Minimum difference in defocus (sampled from uniform rv).
    df_diff_max : float
        Maximum difference in defocus (sampled from uniform rv).
    df_ang_min : float
        Minimum angle of astigmatism (sampled from uniform rv).
    df_ang_max : float
        Maximum angle of astigmatism (sampled from uniform rv).
    kv :  float
        Microscope acceleration potential (kV).
    ac : float
        Amplitude contrast in [0, 1.0].
    cs : float
        Spherical aberration (mm).
    bf : float
        B-factor (1/A^2), divided by 4 in exponential, lowpass positive.
        Positive B factor dampens.
    do_log : bool (default True)
        Option to log progress.
    """
    dfs = np.random.uniform(low=df_min, high=df_max, size=n_particles)
    df_diff = np.random.uniform(low=df_diff_min, high=df_diff_max, size=n_particles)
    df1s = dfs - df_diff / 2
    df2s = dfs + df_diff / 2
    df_ang_deg = np.random.uniform(low=df_ang_min, high=df_ang_max, size=n_particles)
    ctfs = np.empty((n_particles, n_pixels, n_pixels))
    freq_mag_2d, angles_rad = ctf_freqs(n_pixels, psize, dim=2)
    for idx in range(n_particles):
        if do_log and idx % max(1, (n_particles // 10)) == 0:
            logging.debug(idx)  # needs work: logger
        ctfs[idx] = eval_ctf_jitted(
            freq_mag_2d,
            angles_rad,
            def1=df1s[idx],
            def2=df2s[idx],
            angast=df_ang_deg[idx],
            phase=phase,
            kv=kv,
            ac=ac,
            cs=cs,
            bf=bf,
        )
    return (ctfs, df1s, df2s, df_ang_deg)
