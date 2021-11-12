"""Wrapper for the TEM Simulator."""
import yaml
import simutils


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
        self.path_dict = self.get_raw_config_from_yaml(path_config)
        self.raw_sim_dict = self.get_raw_config_from_yaml(sim_config)

        self.output_path_dict = self.generate_path_dict(self.path_dict)
        self.sim_dict = self.classify_sim_params(self.raw_sim_dict)
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

    @staticmethod
    def get_raw_config_from_yaml(config_yaml):
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

        return raw_params

    @staticmethod
    def classify_sim_params(raw_sim_params):
        """Take dictionary of individual simulation parameters and groups them into lists.

        Parameters
        ----------
        raw_sim_params : dict of type str to (dict of type str to {str, int, double})
            Dictionary of simulator parameters
        Returns
        -------
        classified_sim_params : dict of type str to list
            Dictionary of grouped simulator parameters
        """
        molecular_model = raw_sim_params['molecular_model']
        specimen_grid_params = raw_sim_params['specimen_grid_params']
        beam_parameters = raw_sim_params['beam_parameters']
        optics_parameters = raw_sim_params['optics_parameters']
        detector_parameters = raw_sim_params['detector_parameters']

        classified_sim_params = {
            'molecular_model': list(molecular_model.values()),
            'specimen_grid_params': list(specimen_grid_params.values()),
            'beam_parameters': list(beam_parameters.values()),
            'optics_parameters': list(optics_parameters.values()),
            'detector_parameters': list(detector_parameters.values())
        }

        return classified_sim_params

    @staticmethod
    def generate_path_dict(path_dict):
        """Return the paths to pdb, crd, log, inp, and h5 files as strings.

        Parameters
        ----------
        path_dict : dict of stype str to str
            Dict of user inputted path config parameters containing keys:
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

        file_path_dict = {}

        output_file_path = path_dict['output_dir'] \
                           + path_dict['pdb_keyword'] \
                           + path_dict['micrograph_keyword']

        file_path_dict['pdb_file'] = path_dict['pdb_dir'] + path_dict['pdb_keyword'] + '.pdb'
        file_path_dict['crd_file'] = output_file_path + '.txt'
        file_path_dict['mrc_file'] = output_file_path + '.mrc'
        file_path_dict['log_file'] = output_file_path + '.log'
        file_path_dict['inp_file'] = output_file_path + '.inp'
        file_path_dict['h5_file'] = output_file_path + '.h5'

        return file_path_dict

    def create_crd_file(self, pad):
        """Format and write molecular model data to crd_file for use in TEM-simulator.

        Parameters
        ----------
        pad : double
            Pad to be added to maximal dimension of the object read from pdb_file
        """

        x_range, y_range, num_part = simutils.define_grid_in_fov(
            self.sim_dict['sample_dimensions'],
            self.sim_dict['optics_params'],
            self.sim_dict['detector_params'],
            self.output_path_dict['pdb_file'],
            Dmax=30,
            pad=pad
        )

        simutils.write_crd_file(
            num_part,
            xrange=x_range,
            yrange=y_range,
            crd_file=self.output_path_dict['crd_file']
        )

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
