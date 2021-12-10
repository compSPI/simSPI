"""Unit tests for crd.py."""

import os
import tempfile

from simSPI import crd


def test_write_crd_file():
    """Test whether function creates and writes to file on given path."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_file = temp_dir.name + "/crd.txt"

    assert not os.path.isfile(temp_file)
    crd.write_crd_file(numpart=20, crd_file=temp_file)
    assert os.path.isfile(temp_file)

    file_already_exists = temp_file
    crd.write_crd_file(numpart=20, crd_file=file_already_exists)
    assert os.path.isfile(file_already_exists)


def test_get_rotlist():
    """Test whether randomly generated rotlist is of correct length."""
    n_particles = 10
    assert len(crd.get_rotlist(n_particles)) == n_particles
