import numpy as np
import os
import tempfile

from simSPI import crd


def test_write_crd_file():
    """Test whether function creates and writes to file on given path."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = temp_dir.name + '/crd.txt'

    assert not os.path.isfile(temp_file)
    crd.write_crd_file(numpart=20, crd_file=temp_file)
    assert os.path.isfile(temp_file)


def test_get_rotlist():
    """Test whether randomly generated rotlist is of correct length."""
    numpart = 10
    assert len(crd.get_rotlist(numpart)) == len(range(0, numpart + 1))


def test_rotation_matrix_to_euler_angles():
    """Test correct generation of Euler angles for singular and non-singular rotation matrices."""
    # sample rotation data
    angle = np.pi / 6
    c = np.cos(angle)
    s = np.sin(angle)

    # generate sample non-singular (ns) and singular (s) rotation matrices
    ns_rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    s_rot = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

    # find expected Euler angles for singular matrix
    theta = -np.arcsin(ns_rot[2, 0])
    psi = np.arctan2(ns_rot[2, 1] / np.cos(theta), ns_rot[2, 2] / np.cos(theta))
    phi = np.arctan2(ns_rot[1, 0] / np.cos(theta), ns_rot[0, 0] / np.cos(theta))
    ns_angles = np.array([np.rad2deg(psi), np.rad2deg(theta), np.rad2deg(phi)])

    # find expected Euler angles for non-singular matrix
    theta_s = np.arctan2(-s_rot[1, 2], s_rot[1, 1])
    psi_s = np.arctan2(-s_rot[2, 0], 0)
    phi_s = 0
    s_angles = np.array([np.rad2deg(theta_s), np.rad2deg(psi_s), np.rad2deg(phi_s)])

    assert np.allclose(ns_angles, crd.rotation_matrix_to_euler_angles(ns_rot))
    assert np.allclose(s_angles, crd.rotation_matrix_to_euler_angles(s_rot))


def test_is_rotation_matrix():
    """Test whether is_rotation_matrix correctly determines if a matrix is a rotation matrix."""
    # sample rotation data
    angle = np.pi / 6
    c = np.cos(angle)
    s = np.sin(angle)

    rot_in_x = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    rot_in_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rot_in_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    not_rot = np.ones((3, 3, 3))

    assert crd.is_rotation_matrix(rot_in_x)
    assert crd.is_rotation_matrix(rot_in_y)
    assert crd.is_rotation_matrix(rot_in_z)
    assert not crd.is_rotation_matrix(not_rot)
