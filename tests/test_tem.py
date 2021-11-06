"""Unit test for TEM Simulator wrapper."""

from simSPI import tem


def test_tem_main():
    """Tests simulator class main function."""
    tem.main()


def test_basic_sim():
    """Tests simulator class initialization and basic functionality."""
    paths_yaml = '../test/paths.yml'
    sims_yaml = '../test/sim.yml'
    config_yaml = '../test/config.yml'
    test_pdb = 'test.pdb'

    simulator = tem.TEMSimulator(paths_yaml, sims_yaml)
    simulator.get_config_from_yaml(config_yaml)
    tem.TEMSimulator.generate_path_dict(test_pdb)

    simulator.run(test_pdb)

    simulator.create_crd_file(0)
    simulator.get_image_data()
    simulator.generate_parameters_dictionary()
    simulator.write_inp_file()
    simulator.extract_particles(None, 0)
    simulator.export_particle_stack([])
