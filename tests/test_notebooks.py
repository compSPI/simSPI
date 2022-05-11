"""Unit tests for demo notebooks."""
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def test_resources_tem_sim():
    """Return resources for testing the sim_tutorial and the tem_tutorial notebooks."""
    test_files_path = "/work/tests/test_files/tem"

    notebooks_path = "/work/notebooks/"
    cwd = os.getcwd()
    resources = {
        "notebook_path": str(Path(cwd, notebooks_path)),
        "arg_paths": {
            "path_config": str(Path(cwd, test_files_path, "path_config.yaml")),
            "sim_config": str(Path(cwd, test_files_path, "sim_config.yaml")),
            "sim_yaml_clean": str(Path(cwd, test_files_path, "sim_config_clean.yaml")),
        },
    }

    return resources


@pytest.fixture
def test_resources_linsim():
    """Return resources for testing the linear_simulator_tutorial notebook."""
    test_data_path = "/work/tests/data/"
    notebooks_path = "/work/notebooks/"
    cwd = os.getcwd()
    resources = {
        "notebook_path": str(Path(os.getcwd(), notebooks_path)),
        "arg_paths": {
            "vol_path": str(Path(cwd, test_data_path, "linear_simulator_vol.mrc")),
            "data_path": str(Path(cwd, test_data_path, "linear_simulator_data.npy")),
        },
    }

    return resources


def _exec_notebook(path, arg_paths):
    """Execute notebook on path using papermill to fill config paths.

    Parameters
    ----------
    path : String
        The full path of the notebook to test.

    arg_paths: dict[String, String]
        Dictionary containing the path to the config files passed to papermill
        for running the current notebook, e.g. arg_paths["path_config"] = /work/...
    """
    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    args = [
        "papermill",
        path,
        file_name,
        "--execution-timeout",
        "180",
    ]

    for arg_name, arg_val in arg_paths.items():
        args.extend(["-p", arg_name, arg_val])

    print(" ".join(args))

    subprocess.check_call(args)


def test_tem_tutorial(test_resources_tem_sim):
    """Test execution of tem_tutorial.ipynb notebook."""
    notebook_name = "/tem_tutorial.ipynb"
    notebook_path = test_resources_tem_sim["notebook_path"]

    try:
        _exec_notebook(
            notebook_path + notebook_name, test_resources_tem_sim["arg_paths"]
        )
    except subprocess.CalledProcessError as exc:
        assert False, f"{notebook_name} raised exception: {exc}"


def test_sim_tutorial(test_resources_tem_sim):
    """Test execution of tem_tutorial.ipynb notebook."""
    notebook_name = "/sim_tutorial.ipynb"
    notebook_path = test_resources_tem_sim["notebook_path"]

    try:
        _exec_notebook(
            notebook_path + notebook_name, test_resources_tem_sim["arg_paths"]
        )
    except subprocess.CalledProcessError as exc:
        assert False, f"{notebook_name} raised exception: {exc}"


def test_linear_simulator(test_resources_linsim):
    """Test execution of the linearsimulator_tutorial.ipynb notebook."""
    notebook_name = "/linearsimulator_tutorial.ipynb"
    notebook_path = test_resources_linsim["notebook_path"]

    try:
        _exec_notebook(
            notebook_path + notebook_name, test_resources_linsim["arg_paths"]
        )
    except subprocess.CalledProcessError as exc:
        assert False, f"{notebook_name} raised exception: {exc}"
