"""Test sim_volumes."""
import numpy as np
import os
import torch
import sim_volumes as sv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestSimVolumes:
    def test_modify_weight(self):
        points = np.asarray([[1, 0, 0, 3], [0, 8, 0, 5], [0, 0, 7, 9]])
        vol_size = 64
        center = 2
        volume = np.zeros((vol_size,) * 3)
        volume = sv.modify_weight(points, volume, vol_size, center)
        assert volume.shape == (vol_size,) * 3
        assert volume[1][1][1] == 3.4866983294215885e-164

    def test_simulate_volumes(self):
        particules = np.asarray([[1, 0], [0, 8], [0, 0]])
        n_volumes = 1
        vol_size = 64
        volumes, qs = sv.simulate_volumes(particules, n_volumes, vol_size)
        assert volumes.shape == (1, 64, 64, 64)
        assert len(qs) == 1
        assert qs[0][0] == '['

    def test_save_volume(self):
        particules = np.asarray([[1, 0], [0, 8], [0, 0]])
        n_volumes = 1
        vol_size = 64
        main_dir = ""
        name = "2particules"
        volumes, qs = sv.save_volume(
            particules, n_volumes, vol_size, main_dir, name, center=2)
        assert volumes.shape == (1, 64, 64, 64)
