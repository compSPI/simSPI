"""Helper functions for tem.py.

Write particle stack data (slice of protein)
to .txt file containing coordinate (crd) information on particle.
"""

import logging
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def write_crd_file(
    n_particles,
    xrange=np.arange(-100, 110, 10),
    yrange=np.arange(-100, 110, 10),
    crd_file="crd.txt",
):
    """Write particle data to .txt file containing particle stack data.

    Particle center coordinates as well as its Euler angles is written to file.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    xrange : ndarray
        Range of particle center x-coordinates to write.
    yrange : ndarray
        Range of particle center y-coordinates to write.
    crd_file : str
        Relative path to output .txt file.
    """
    log = logging.getLogger()

    if os.path.exists(crd_file):
        log.info(crd_file + " already exists.")
    else:
        rotlist = get_rotlist(n_particles)
        with open(crd_file, "w") as crd:
            crd.write("# File created by TEM-simulator, version 1.3.\n")
            crd.write(f"{n_particles} 6\n")
            crd.write(
                "#            \
                x             \
                y             \
                z             \
                phi           \
                theta         \
                psi  \n"
            )
            for y in yrange:
                for x in xrange:
                    for i in range(n_particles):
                        crd_table = {
                            "x": x,
                            "y": y,
                            "z": 0,
                            "phi": rotlist[i][0],
                            "theta": rotlist[i][1],
                            "psi": rotlist[i][2],
                        }
                        crd.write(
                            f"{crd_table['x']:14.4f} \
                            {crd_table['y']:14.4f} \
                            {crd_table['z']:14.4f} \
                            {crd_table['phi']:14.4f} \
                            {crd_table['theta']:14.4f} \
                            {crd_table['psi']:14.4f}\n",
                        )


def get_rotlist(n_particles):
    """Return a rotation list containing Euler angles.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    """
    rotlist = []
    for _ in range(n_particles + 1):
        x = R.random(5).as_euler("xyz", degrees=True)
        rotlist.append(x)
    return rotlist
