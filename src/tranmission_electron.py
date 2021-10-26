
class TEMSimulator:
    """Wrapper for TEMSimulator and relevant utilites

    Attributes
    ----------
    path_config_yaml : string
        Relative path to YAML file containing the following keys:
                pdb_dir : str
                    Relative path to directory containing input .pdb files
                output_dir : str, optional
                    Relative path to output directory, if not specified an output directory next to the input file
                micrograph_keyword : str, optional
                    User-specified keyword appended to output files
                simulator_dir : str
                    Relative path to local TEM sim file
    sim_config_yaml : string
        Relative directory to YAML file containing desired TEM simulator parameters
    Methods
    -------
    runSim(pdbFile)
        Runs the TEM simulator on pdbFile. Return particle stack with appropriate metadata.
    """

    def __init__(self, path_config_yaml, sim_config_yaml):
        self.path_config_yaml = path_config_yaml
        self.sim_config_yaml = sim_config_yaml

    def runSim(self, pdbFile):
        """ Runs TEM simulator on input file and produces particle stacks with metadata

        Parameters
        ----------
        pdbFile : str
            Relative file path to input .pdb file for sim

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """

        return None

    def get_config_from_yaml(self, config_yaml):
        """Creates dictionary with parameters from YAML file and groups them into lists

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

        classified_params = {}

        return classified_params

    @staticmethod
    def classify_input_config(raw_params):
        """Takes dictionary of individual parameters and groups them into lists

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
    def generate_file_paths(path_yaml):
        """Returns the paths to relevant to-be-generated pdb, crd, log, inp, and h5 files as strings

        Parameters
        ----------
        path_yaml : str
            Relative path to YAML file containing the following keys:
                pdb_dir : str
                    Relative path to directory containing input .pdb files
                output_dir : str, (default = None)
                    Relative path to output directory, if not specified an output directory next to the input file
                micrograph_keyword : str, (default = None)
                    User-specified keyword appended to output files
                simulator_dir : str
                    Relative path to local TEM sim file

        Returns
        -------
         file_paths : dict of type str to str
             Dict of file paths that includes keys:
                pdb_file
                    Relative path to pdb input file
                crd_file
                    Relative path to desired output crd file
                h5_file
                    Relative path to desired output h5 file
                inp_file
                    Relative path to desired output inp file
                mrc_file
                    Relative path to desired output mrc file
                log_file
                    Relative path to desired output log file
                sim_file
                    Relative path to local TEM sim file
        """

        file_paths = {}
        return file_paths

    @staticmethod
    def create_crd_file(file_paths, sim_param_arrays, pad):
        """Formats and writes molecular model data to crd_file for use in TEM-simulator.

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
               pad : double
                   Pad to be added to maximal dimension of the object read from pdb_file
              """


        return

    @staticmethod
    def get_image_data(file_paths):
        """Run simulator and return data

        Parameters
        ----------
        file_paths : dict of type str to str
             Dict of file paths that includes keys:
                inp_file
                    Relative path to populated .inp file
                mrc_file
                    Relative path to output mrc file
                sim_file
                    Relative path to local TEM sim file
        Returns
        -------
        mrc_data : array
            Contains particle data parsed TEM simulator ouput mrc
        """
        mrc_data = []
        return mrc_data

    @staticmethod
    def generate_parameters_dictionary(file_paths, sim_params):
        """Compiles all relevant experiment data into .inp friendly file for use in TEM-simulator.

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
    def write_inp_file(params_dict, file_paths):
        """Writes simulation parameters to .inp file for use by the TEM-simulator.

        Parameters
        ----------
        params_dict : dict
            .inp friendly dictionary containing simulation input parameters.
        file_paths : dict of type str to str
            Dict of file paths that includes keys:
                inp_file
                    Relative path to input file to be populated with parameters
        """
        return

    @staticmethod
    def extract_particles(micrograph, params_dict, file_paths, pad):
        """Formats and writes molecular model data to crd_file for use in TEM-simulator.

        Parameters
        ----------
        micrograph : arr
            Array containing TEM-simulator micrograph output
        params_dict : dict
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
        return
    @staticmethod
    def export_particle_stack(particles, file_paths, params_dict):
        """Exports extracted particle data to h5 file

        Parameters
        ----------
        particles : arr
            Individual particle data extracted from micrograph
        file_paths : arr
            Relative path to .h5 output file
        params_dict : arr
            .inp friendly dictionary containing simulation input parameters
        """
        return


def main():
    sim = TEMSimulator('./temp_workspace/input/path_config.yaml', './temp_workspace/input/sim_config.yaml')
    sim_data = sim.runSim()


if __name__ == "__main__":
    main()
