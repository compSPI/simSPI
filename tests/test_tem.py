"""Unit test for TEM Simulator wrapper."""
import os
import tempfile

import yaml

from simSPI import tem


def test_tem_main():
    """Tests simulator class main function."""
    tem.main()


def test_basic_sim():
    """Tests simulator class initialization and basic functionality."""
    tmp_config = tempfile.NamedTemporaryFile(delete=False)
    tmp_paths = tempfile.NamedTemporaryFile(delete=False)
    tmp_pdb = tempfile.NamedTemporaryFile(delete=False)

    try:
        tmp_config.write(yaml.dump({0: True}).encode("utf-8"))
        sim = tem.TEMSimulator(
            tmp_paths.name,
            tmp_config.name,
        )
        _ = sim.run(tmp_pdb.name)
        tem.TEMSimulator.generate_path_dict(tmp_pdb.name)

        sim.create_crd_file(0)
        sim.get_image_data()
        sim.generate_parameters_dictionary()
        sim.write_inp_file()
        sim.extract_particles(None, 0)
        sim.export_particle_stack([])
    finally:
        tmp_config.close()
        tmp_paths.close()
        tmp_pdb.close()
        os.unlink(tmp_config.name)
        os.unlink(tmp_paths.name)
        os.unlink(tmp_pdb.name)
