"""Wrapper for the TEM Simulator."""
import yaml


class TEMSimulator:
    """Wrapper for the TEM Simulator.

    Parameters
    ----------
    path_config : str
        Relative path to YAML file containing file paths for TEM Simulator.
    sim_config : str
        Relative path to YAML file containing simulator paths for TEM Simulator.

    """

    def __init__(self, path_config, sim_config):
        print('\n')
       # self.path_dict = self.get_config_from_yaml(path_config)
        print('\n')
        self.sim_dict = self.get_config_from_yaml(sim_config)
        self.placeholder = 0

    def run(self, pdb_file):
        """Run TEM simulator on input file and produce particle stacks with metadata.

        Parameters
        ----------
        pdb_file : str
            Relative file path to input .pdb file for sim

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """
        pdb_file = "placeholder_to_pass_tests"
        particles = [self.placeholder, pdb_file]
        return particles

    def get_config_from_yaml(self, config_yaml):
        """Create dictionary with parameters from YAML file and groups them into lists.

        Parameters
        ----------
        config_yaml : str
            Relative path to YAML file containing parameters for TEM Simulator
        Returns
        -------
        classified_params : dict
            Dictionary containing grouped parameters for TEM Simulator, with keys:
                seed : str maps to int
                    Seed for TEM Simulator
                particle_mrcout : str maps to bool
                    Flag for optional volume map of sample
                sample_dimensions : str maps to
                    List containing the specimen grid parameters
                beam_params : str maps to list
                    List containing the beam parameters
                detector_params : str maps to list
                    List containing the detector parameters
                optics_params : str maps to list
                    List containing the optic parameters
        """
        with open(config_yaml, "r") as stream:
            raw_params = yaml.safe_load(stream)
        
        print(raw_params)
        print('\n')
        classified_params = self.classify_input_config(raw_params)

        return classified_params

    @staticmethod
    def classify_input_config(raw_params):
        """Take dictionary of individual parameters and groups them into lists.

        Parameters
        ----------
        raw_params : dict of type str to {str, double, int}
            Dictionary of simulator parameters
        Returns
        -------
        classified_params : dict of type str to list
            Dictionary of grouped parameters
        """

        molecular_model = raw_params['molecular_model']
        specimen_grid_params = raw_params['specimen_grid_params']
        beam_parameters = raw_params['beam_parameters']
        optics_parameters = raw_params['optics_parameters']
        detector_parameters = raw_params['detector_parameters']

        classified_params = {
            'molecular_model': list(molecular_model.values()),
            'specimen_grid_params': list(specimen_grid_params.values()),
            'beam_parameters': list(beam_parameters.values()),
            'optics_parameters': list(optics_parameters.values()),
            'detector_parameters': list(detector_parameters.values())
        }

        return classified_params

    @staticmethod
    def generate_path_dict(pdb_file, output_dir=None, mrc_keyword=None):
        """Return the paths to pdb, crd, log, inp, and h5 files as strings.

        Parameters
        ----------
        pdb_file : str
            Relative path to the pdb file
        output_dir : str, (default = None)
            Relative path to output directory
        mrc_keyword : str, (default = None)
            user-specified keyword appended to output files
        Returns
        -------
        path_dict : dict of type str to str
            Dict of file paths that includes keys:
            pdb_file
                relative path to pdb input file
            crd_file
                relative path to desired output crd file
            h5_file
                relative path to desired output h5 file
            inp_file
                relative path to desired output inp file
            mrc_file
                relative path to desired output mrc file
            log_file
                relative path to desired output log file
        """
        # jed
        path_dict = {}
        return path_dict

    def create_crd_file(self, pad):
        """Format and write molecular model data to crd_file for use in TEM-simulator.

        Parameters
        ----------
        pad : double
            Pad to be added to maximal dimension of the object read from pdb_file
        """
        self.placeholder = 0
        return pad

    def get_image_data(self):
        """Run simulator and return data.

        Returns
        -------
        List containing parsed .mrc data from Simulator
        """
        self.placeholder = 0
        return []

    def generate_parameters_dictionary(self):
        """Compile experiment data into .inp friendly file for use in TEM-simulator.

        Returns
        -------
        param_dictionary : dict
            .inp friendly dictionary containing simulation input parameters.
        """
        self.placeholder = 0
        param_dictionary = {}
        return param_dictionary

    def write_inp_file(self):
        """Write simulation parameters to .inp file for use by the TEM-simulator.

        The .inp files contain the parameters controlling the simulation. These are text
        files whose format is described in the TEM Simulator documentation. They contain
        component headings which divide the files into different sections (e.g.
        different particles) and parameter assignments of the form
        "<parameter> = <value>".
        """
        self.placeholder = 0
        return self.placeholder

    def extract_particles(self, micrograph, pad):
        """Extract particle data from micrograph.

        Parameters
        ----------
        micrograph : arr
            Array containing TEM-simulator micrograph output
        pad : double
            Pad to be added to maximal dimension of the object read from pdb_file

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """
        self.placeholder = 0
        return [micrograph, pad]

    def export_particle_stack(self, particles):
        """Export extracted particle data to h5 file.

        Parameters
        ----------
        particles : arr
            Individual particle data extracted from micrograph

        """
        self.placeholder = 0
        return particles


def main():
    """Return 1 as a placeholder."""
    t = TEMSimulator('../temp_workspace/input/path_config.yaml', '../temp_workspace/input/sim_config.yaml')
    return 1


if __name__ == "__main__":
    main()
