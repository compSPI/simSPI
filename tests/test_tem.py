"""Unit test for TEM Simulator wrapper."""
import copy
import os
import random
import tempfile

import pytest
import yaml

from simSPI import tem


# def test_tem_main():
#     """Tests simulator class main function."""
#     tem.main()
#
#
# def test_basic_sim():
#     """Tests simulator class initialization and basic functionality."""
#     tmp_config = tempfile.NamedTemporaryFile(delete=False)
#     tmp_paths = tempfile.NamedTemporaryFile(delete=False)
#     tmp_pdb = tempfile.NamedTemporaryFile(delete=False)
#
#     try:
#         tmp_config.write(yaml.dump({0: True}).encode("utf-8"))
#         sim = tem.TEMSimulator(
#             tmp_paths.name,
#             tmp_config.name,
#         )
#         _ = sim.run(tmp_pdb.name)
#         tem.TEMSimulator.generate_path_dict(tmp_pdb.name)
#
#         sim.create_crd_file(0)
#         sim.get_image_data()
#         sim.generate_parameters_dictionary()
#         sim.write_inp_file()
#         sim.extract_particles(None, 0)
#         sim.export_particle_stack([])
#     finally:
#         tmp_config.close()
#         tmp_paths.close()
#         tmp_pdb.close()
#         os.unlink(tmp_config.name)
#         os.unlink(tmp_paths.name)
#         os.unlink(tmp_pdb.name)


def setup_test_yamls(temp_path, test_case):
    sim_yaml_template_path = "./test_files/sim_config.yaml"

    case = {"path": test_case[0], "sim": test_case[1]}

    template = {
        "path": {
            "pdb_dir": "./test_files",
            "micrograph_keyword": "_test",
            "pdb_keyword": "4v6x",
            "output_dir": str(temp_path),
            "simulator_dir": "../../../TEM-simulator/src/TEM-simulator",
        }
    }

    optional_keys = ["micrograph_keyword", "output_dir"]

    with open(sim_yaml_template_path, "r") as stream:
        template["sim"] = yaml.safe_load(stream)

    for yaml_type, case_type in case.items():

        if case_type is "complete":
            break
        elif case_type is "minimal":
            template[yaml_type] = {
                key: value
                for key, value in template[yaml_type].items()
                if key not in optional_keys
            }
        elif case_type is "missing_keys":
            random_key = random.choice(
                [key for key in template[yaml_type].keys() if key not in optional_keys]
            )
            print(random_key)
            template[yaml_type].pop(random_key, None)
        elif case_type is "missing_values":
            random_key = random.choice(
                [key for key in template[yaml_type].keys() if key not in optional_keys]
            )
            print(random_key)
            template[yaml_type][random_key] = None
        else:
            print("Unsupported case")
            raise ValueError

    path_yaml = os.path.join(temp_path, "path.yaml")
    sim_yaml = os.path.join(temp_path, "sim.yaml")

    with open(sim_yaml, "w") as outfile:
        yaml.dump(template["sim"], outfile)
    with open(path_yaml, "w") as outfile:
        yaml.dump(template["path"], outfile)

    return sim_yaml, path_yaml


def test_temsimulator_complete(tmp_path):
    output_path_dict_template = {
        "pdb_file": ".pdb",
        "crd_file": ".txt",
        "mrc_file": ".mrc",
        "log_file": ".log",
        "inp_file": ".inp",
        "h5_file": ".h5",
    }
    sim_dict_keys_template = {
        "beam_parameters": 4,
        "optics_parameters": 10,
        "detector_parameters": 11,
        "specimen_grid_params": 3,
        "molecular_model": 3,
    }
    parameters_dict_keys = [
        "simulation",
        "sample",
        "particle",
        "particleset",
        "beam",
        "optics",
        "detector",
    ]

    test_sim_yaml, test_path_yaml = setup_test_yamls(
        "./test_files", ("missing_values", "complete")
    )

    # test_sim_yaml = './test_files/sim_config.yaml'
    sim = tem.TEMSimulator(test_path_yaml, test_sim_yaml)

    assert set(sim.parameter_dict.keys()) == set(parameters_dict_keys)

    for key, value in sim.output_path_dict.items():
        assert key in output_path_dict_template.keys()
        directory, file = os.path.split(value)
        assert os.path.isdir(directory)
        assert output_path_dict_template[key] in value

    for key, value in sim.sim_dict.items():
        assert key in sim_dict_keys_template.keys()
        assert len(value) is sim_dict_keys_template[key]



# def test_temsimulator_missing_values(tmp_path):

@pytest.mark.parametrize("test_case", [("missing_keys", "complete"), ("complete", "missing_keys")])
def test_temsimulator_missing_keys(tmp_path, test_case):
    test_sim_yaml, test_path_yaml = setup_test_yamls(
        "./test_files", test_case
    )

    with pytest.raises(KeyError):
        sim = tem.TEMSimulator(test_path_yaml, test_sim_yaml)

