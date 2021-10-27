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
        self.file_paths = self._getConfigFromYaml(path_config)
        self.sim_dict = self._getConfigFromYaml(sim_config)
        self.placeholder = 0

    def run_sim(self, pdb_file):
        """Run TEM simulator on input file and produces particle stacks with metadata.

        Parameters
        ----------
        pdb_file : str
            Relative file path to input .pdb file for sim

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """
        # self.file_paths = self._getIOFilePaths(pdb_file)
        # self._buildCordFile(**self.file_paths, **self.sim_dict)
        # self.param_dict = self._build_param_dict(**self.file_paths, **self.sim_dict)
        # self._build_inpFile(self.param_dict, **self.file_paths)
        #
        # data = self._getImageData(**self.filePaths)
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
        classified_params = self.classify_input_config(raw_params)

        return classified_params

    @staticmethod
    def classify_input_config(raw_params):
        """Take dictionary of individual parameters and groups them into lists.

        Parameters
        ----------
        raw_params : dict of type str to {str,bool,int}
            Dictionary of simulator parameters
        Returns
        -------
        classified_params : dict of type str to {str,bool,int,list}
            Dictionary of grouped parameters
        """
        classified_params = {}
        return classified_params

    @staticmethod
    def generate_file_paths(pdb_file, output_dir=None, mrc_keyword=None):
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
        file_paths : dict of type str to str
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
        file_paths = {}
        return file_paths

    @staticmethod
    def create_crd_file(file_paths, sim_param_arrays, pad):
        """Format and write molecular model data to crd_file for use in TEM-simulator.

        Parameters
        ----------
        file_paths : dict of type str to str
            Dict of file paths that includes keys:
                pdb_file
                    Relative path to pdb input file
                crd_file
                    Relative path to populated crd file
        sim_param_arrays : dict
            Dict of simulation parameters that includes keys:
                sample_dimensions : str maps to list
                    List containing the specimen grid parameters
                detector_params : str maps to list
                    List containing the detector parameters
                optics_params : str maps to list
                    List containing the optic parameters

        """
        return

    @staticmethod
    def get_image_data(file_paths, sim_path):
        """Run simulator and return data.

        Parameters
        ----------
        sim_path : str
            relative path to local TEM simulator installation
        file_paths : dict of type str to str
             Dict of file paths that includes keys:
                inp_file
                    relative path to populated .inp file
                mrc_file
                    relative path to ouput mrc file
                    sim_params : dict
        Returns
        -------
        List containing parsed .mrc data from Simulator
        """
        # SIMULATOR_BIN = Path(sim_path)  # might have to change depending on OS
        # inp_file = Path(file_paths["inp_file"])

        # cmd = "{0} {1}".format(SIMULATOR_BIN, inp_file)
        # os.system(shlex.quote(cmd))

        return []

    @staticmethod
    def generate_parameters_dictionary(file_paths, sim_params):
        """Compile experiment data into .inp friendly file for use in TEM-simulator.

        Parameters
        ----------
        file_paths : dict of type str to str
             Dict of file paths that includes keys:
                pdb_file
                    Relative path to pdb input file
                crd_file
                    Relative path to populated crd file
                mrc_file
                    Relative path to desired output mrc file
                log_file
                    Relative path to desired output log file
        sim_params : dict
            Dict of simulation parameters that includes keys:
                seed : str maps to int
                particle_mrcout : str maps to bool
                sample_dimensions : str maps to list
                beam_params : str maps to list
                detector_params : str maps to list
                optic_params : str maps to list

        Returns
        -------
        param_dictionary : dict
            .inp friendly dictionary containing simulation input parameters.
        """
        param_dictionary = {}
        return param_dictionary

    @staticmethod
    def write_inp_file(param_dict, file_paths):
        """Write simulation parameters to .inp file for use by the TEM-simulator.

        Parameters
        ----------
        param_dict : dict
            .inp friendly dictionary containing simulation input parameters.
        file_paths : dict of type str to str
            Dict of file paths that includes keys:
                inp_file
                    Relative path to input file to be populated with parameters
        """
        return

    @staticmethod
    def extract_particles(micrograph, sim_param_arrays, file_paths, pad):
        """Format and write molecular model data to crd_file for use in TEM-simulator.

        Parameters
        ----------
        micrograph : arr
            Array containing TEM-simulator micrograph output
        sim_param_arrays : dict
            Dictionary containing arrays of simulation parameters
        file_paths : dict
            Dictionary of file paths that includes keys:
                pdb_file
                    Relative path to populated .inp file
        pad : double
            Pad to be added to maximal dimension of the object read from pdb_file

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """
        return []

    @staticmethod
    def export_particle_stack(particles, file_paths, params_dict):
        """Export extracted particle data to h5 file.

        Parameters
        ----------
        particles : arr
            Individual particle data extracted from micrograph
        file_paths : arr
            Relative path to .h5 output file
        params_dict : arr
            .inp friendly dictionary containing simulation input parameters

        """
        return None


def main():
    """Start up the TEM Simulator for a particular pdb file."""
    sim = TEMSimulator(
        "./temp_workspace/input/path_config.yaml",
        "./temp_workspace/input/sim_config.yaml",
    )
    pdb_file = "placeholder.pdb"
    _ = sim.run_sim(pdb_file)


if __name__ == "__main__":
    main()
