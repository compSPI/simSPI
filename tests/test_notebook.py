"""Unit tests for demo notebook."""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_resources():
    """Return resources for testing."""
    test_files_path = "/work/tests/test_files"
    notebooks_path = "/work/notebooks/"
    cwd = os.getcwd()
    resources = {
        "files": {
            "notebook_path": str(Path(os.getcwd(), notebooks_path)),
            "path_yaml": str(Path(cwd, test_files_path, "path_config.yaml")),
            "sim_yaml": str(Path(cwd, test_files_path, "sim_config.yaml")),
        }
    }

    return resources


def _exec_notebook(path, path_yaml, sim_yaml):
    """Execute notebook on path using papermill to fill config paths."""
    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    args = [
        "papermill",
        path,
        file_name,
        "-p",
        "path_config",
        path_yaml,
        "-p",
        "sim_config",
        sim_yaml,
        "--execution-timeout",
        "180",
    ]
    subprocess.check_call(args)


def test_tem_tutorial(test_resources):
    """Test execution of tem_tutorial.ipynb notebook."""
    notebook_name = "/tem_tutorial.ipynb"

    notebook_path = test_resources["files"]["notebook_path"]
    sim_yaml = test_resources["files"]["sim_yaml"]
    path_yaml = test_resources["files"]["path_yaml"]

    try:
        _exec_notebook(notebook_path + notebook_name, path_yaml, sim_yaml)
    except subprocess.CalledProcessError as exc:
        assert False, f"{notebook_name} raised exception: {exc}"
