import os
import yaml

from TEM_utils.cryoemio import simio, mrc2data
from TEM_utils.simutils import define_grid_in_fov, write_crd_file, fill_parameters_dictionary, write_inp_file
from pathlib import Path


class TEMSimulator:

    # Most static functions are wrapper functions that can accept dictionaries for arguments.
    # Easier than dealing with a bunch of args
    #
    # TODO
    #   - add functionality to change configuration
    #   - What should naming convention be? camel case or underscores
    #   - Verify functionality
    #   - Establish compatibility for later python versions
    #   - Figure out simulator wrapper

    def __init__(self, path_config, sim_config):
        # Seperated constructor and running actual sim -> lets you run a  bunch of input files with same configs
        # not sure if the helper methods should be made static or not

        self.file_paths = self._getConfigFromYaml(path_config)
        self.sim_dict = self._getConfigFromYaml(sim_config)


    def run_sim(self, pdb_file):
        """ Runs TEM simulator on input file and produces particle stacks with metadata

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
                    Flag for optional volume map of sample        # TODO: verify this, I think I saw this get mapped to mrc_file path in notebook
                sample_dimensions : str maps to
                    List containing the specimen grid parameters
                beam_params : str maps to list
                    List containing the beam parameters
                detector_params : str maps to list
                    List containing the detector parameters
                optics_params : str maps to list
                    List containing the optic parameters
        """

        # Questions:
        # - How do people usually format their YAMLS? Is it in arrays or key:value for each parameter?
        # - If there is no standard YAML config format, do we specify? or do we support all sorts of config files?
        # TODO:
        # - Have default configurations, overwrite from YAML
        # - update YAML parsing based on answers to questions above (rn kinda scuffed)

        with open(config_yaml, 'r') as stream:
            raw_params = yaml.load(stream)
        classified_params = self.classify_input_config(raw_params)

        return classified_params

    @staticmethod
    def classify_input_config(self, raw_params):
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
    def generate_file_paths(self, pdb_file, output_dir=None, mrc_keyword=None):
        """Returns the paths to relevant to-be-generated pdb, crd, log, inp, and h5 files as strings

        Parameters
        ----------
        pdb_file : str
            Relative path to the pdb file
        output_dir : str, (default = None)
            Relative path to output directory, if not specified an output directory next to the input file
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
    def create_crd_file(self, file_paths, sim_param_arrays, pad):
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

        """
        x_range, y_range, num_part = define_grid_in_fov(sim_param_arrays["sample_dimensions"],
                                                       sim_param_arrays["optics_params"],
                                                       sim_param_arrays["detector_params"],
                                                       pdb_file=file_paths["pdb_file"],
                                                       Dmax=30, pad=pad)

        write_crd_file(num_part, xrange=x_range, yrange=y_range, crd_file=file_paths["crd_file"])

    @staticmethod
    def get_image_data(self, input_files, sim_path):
        """Run simulator and return data

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

        """

        SIMULATOR_BIN = Path(sim_path)  # might have to change depending on OS
        inp_file = Path(input_files["inp_file"])

        cmd = '{0} {1}'.format(SIMULATOR_BIN, inp_file)
        os.system(cmd)

        return mrc2data(mrc_file=input_files["mrc_file"])

    @staticmethod
    def generate_parameters_dictionary(self, file_paths, sim_params):
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
        return fill_parameters_dictionary(mrc_file=file_paths["mrc_file"],
                                          pdb_file=file_paths["pdb_file"],
                                          particle_mrcout=file_paths["particle_mrcout"],
                                          crd_file=file_paths["crd_file"],
                                          sample_dimensions=sim_params["sample_dimensions"],
                                          beam_params=sim_params["beam_params"],
                                          optics_params=sim_params["optic_params"],
                                          detector_params=sim_params["detector_params"],
                                          log_file=file_paths["log_file"],
                                          seed=sim_params["seed"])

    @staticmethod
    def write_inp_file(self, param_dict, file_paths):
        """Writes simulation parameters to .inp file for use by the TEM-simulator.

        Parameters
        ----------
        paramDict : dict
            .inp friendly dictionary containing simulation input parameters.
        file_paths : dict of type str to str
            Dict of file paths that includes keys:
                inp_file
                    Relative path to input file to be populated with parameters
        """
        write_inp_file(inp_file=param_dict["inpFile"], dict_params=param_dict)

    @staticmethod
    def extract_particles(self, micrograph, sim_param_arrays, file_paths, pad):
        """Formats and writes molecular model data to crd_file for use in TEM-simulator.

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
        return
        """

    @staticmethod
    def export_particle_stack(particles,file_paths,params_dict):
        """Exports extracted particle data to h5 file

        Parameters
        ----------
        particles : arr
            Individual particle data extracted from micrograph
        file_paths : arr
            Relative path to .h5 output file
        params_dict : arr
            .inp friendly dictionary containing simulation input parameters

        Returns
        -------

        """
        return None


def main():
    sim = TEMSimulator('./temp_workspace/input/path_config.yaml', './temp_workspace/input/sim_config.yaml')
    sim_data = sim.run_sim()


if __name__ == "__main__":
    main()
