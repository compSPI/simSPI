import os
from TEM_utils.cryoemio import simio, mrc2data
from TEM_utils.simutils import define_grid_in_fov, write_crd_file, fill_parameters_dictionary, write_inp_file
from pathlib import Path
import yaml

# Have more verbose comments rn instead of docstrings

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


    def runSim(self, pdbFile):
        self.filePaths = self._getIOFilePaths(pdbFile)
        self._buildCordFile(**self.file_paths, **self.sim_dict)
        self.paramDict = self._buildParamDict(**self.file_paths, **self.sim_dict)
        self._buildInpFile(self.paramDict, **self.file_paths)

        data = self._getImageData(**self.filePaths)
        return data


    @staticmethod
    def _getConfigFromYaml(config):
        #
        # Parses Yaml file and returns the following : seed,sample_dimensions,beam_params,optics_params,detector_params
        # ( seed is a random int)
        #
        # get configurations from yaml.
        # For now, yaml must specify all configurations in the notebook in this format (sim_config.yaml)
        #
        # Questions:
        # - How do people usually format their YAMLS? Is it in arrays or key:value for each parameter?
        # - If there is no standard YAML config format, do we specify? or do we support all sorts of config files?
        # TODO:
        # - Have default configurations, overwrite from YAML
        # - update YAML parsing based on answers to questions above (rn kinda scuffed)

        with open(config, 'r') as stream:
            parsed_config = yaml.load(stream)

        return parsed_config



    @staticmethod
    def _getIOFilePaths(pdbFile):
        #
        # return dict of file paths with keys [ pdb_file, mrc_file, crd_file, log_file, inp_file,h5_file ]
        #
        # Use the input file to create ouput dir in same location
        # possibly add an output dir?
        # Implement

        filePaths = {}
        return filePaths

    @staticmethod
    def _buildCordFile(crd_file, pdb_file, sample_dimensions, optics_params, detector_params, **kwargs):
        # use simutils to build co-ordinate file
        # Creates .crd file in the crd_file path
        x_range, y_range, numpart = define_grid_in_fov(sample_dimensions,
                                                       optics_params,
                                                       detector_params,
                                                       pdb_file=pdb_file,
                                                       Dmax=30, pad=5.)

        write_crd_file(numpart, xrange=x_range, yrange=y_range, crd_file=crd_file)

    @staticmethod
    def _getImageData(inp_file, mrc_file, **kwargs):
        # generate command and run simulator, return mrcData
        # create mrc file

        SIMULATOR_BIN = Path(SIMULATOR)  # might have to change depending on OS
        inp_file = Path(inp_file)

        cmd = '{0} {1}'.format(SIMULATOR_BIN, inp_file)
        os.system(cmd)

        return mrc2data(mrc_file=mrc_file)

    @staticmethod
    def _buildParamDict(mrcFile, pdbFile, crdFile, sampleDimensions, beamParams, opticsParams, detectorParams, logFile,
                        seed, **kwargs):
        return fill_parameters_dictionary(mrc_file=mrcFile,
                                          pdb_file=pdbFile,
                                          particle_mrcout=mrcFile,
                                          crd_file=crdFile,
                                          sample_dimensions=sampleDimensions,
                                          beam_params=beamParams,
                                          optics_params=opticsParams,
                                          detector_params=detectorParams,
                                          log_file=logFile,
                                          seed=seed)

    @staticmethod
    def _buildInpFile(paramDict, inpFile, **kwargs):
        # mutates inpFile
        write_inp_file(inp_file=inpFile, dict_params=paramDict)


def main():
    sim = TEMSimulator('./temp_workspace/input/path_config.yaml', './temp_workspace/input/sim_config.yaml')
    sim_data = sim.runSim()


if __name__ == "__main__":
    main()
