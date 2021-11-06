"""Unit test for TEM Simulator wrapper."""

from simSPI import tem


def test_tem_main():
    """Tests simulator class main function."""
    tem.main()


def test_basic_sim():
    """Tests simulator class initialization and basic functionality."""
    simulator = tem.TEMSimulator("../test/paths.yml", "../test/sim.yml")
    simulator.get_config_from_yaml("../test/config.yml")
    tem.TEMSimulator.generate_path_dict("test.pdb")

    simulator.run("test.pdb")

    simulator.create_crd_file(0)
    simulator.get_image_data()
    simulator.generate_parameters_dictionary()
    simulator.write_inp_file()
    simulator.extract_particles(None, 0)
    simulator.export_particle_stack([])
