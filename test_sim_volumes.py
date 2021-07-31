"""Test sim_volumes."""
import os

import numpy as np

import sim_volumes

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class TestSimVolumes:
    @staticmethod
    def test_get_random_quat():
        quaternions = sim_volumes.get_random_quat(5)
        assert quaternions.shape == (5, 4)
        assert type(quaternions) is np.ndarray

    @staticmethod
    def test_uniform_rotations():
        rot, quat = sim_volumes.uniform_rotations(5)
        assert rot.shape == (5, 3, 3)
        assert quat.shape == (5, 4)

    @staticmethod
    def test_modify_weight():
        points = np.asarray([[1, 0, 0, 3], [0, 8, 0, 5], [0, 0, 7, 9]])
        vol_size = 64
        center = 2
        volume = np.zeros((vol_size,) * 3)
        volume = sim_volumes.modify_weight(points, volume, vol_size, center)
        assert volume.shape == (vol_size,) * 3
        assert volume[1][1][1] == 3.4866983294215885e-164

    @staticmethod
    def test_simulate_volumes():
        particules = np.asarray([[1, 0], [0, 8], [0, 0]])
        n_volumes = 1
        vol_size = 64
        volumes, qs = sim_volumes.simulate_volumes(particules, n_volumes, vol_size)
        assert volumes.shape == (1, 64, 64, 64)
        assert len(qs) == 1
        assert qs[0][0] == "["

    @staticmethod
    def test_save_volume():
        particules = np.asarray([[1, 0], [0, 8], [0, 0]])
        n_volumes = 1
        vol_size = 64
        main_dir = ""
        name = "2particules"
        volumes, qs = sim_volumes.save_volume(
            particules, n_volumes, vol_size, main_dir, name
        )
        assert volumes.shape == (1, 64, 64, 64)
        assert len(qs) == 1
        assert qs[0][0] == "["
