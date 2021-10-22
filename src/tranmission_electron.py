import os
import numpy as np
from TEM_utils.cryoemio import simio, mrc2data
from TEM_utils.simutils import define_grid_in_fov,write_crd_file,fill_parameters_dictionary,write_inp_file
from pathlib import Path
import yaml


SIMULATOR='../Material/tem-Simulator/TEM-simulator/Release/TEM-simulator.exe'
PDBFILE = './temp_workspace/input/4v6x.pdb'
CONFIG = './temp_workspace/input/configurations.yaml'

def getConfigFromYaml(configFile):
    #
    # Parses Yaml file and returns the following : seed,sample_dimensions,beam_params,optics_params,detector_params
    # ( seed is a random int)
    #
    # get configurations from yaml.
    # For now, yaml must specify all configurations in the notebook in this format (configurations.yaml)
    #
    # Questions:
    # - How do people usually format their YAMLS? Is it in arrays or key:value for each parameter?
    # - If there is no standard YAML config format, do we specify? or do we support all sorts of config files?
    # TODO:
    # - Have default configurations, overwrite from YAML
    # - update YAML parsing based on answers to questions above (rn kinda scuffed)


    with open(configFile, 'r') as stream:
        parsedConfig = yaml.load(stream)
    parsedConfig['seed'] = 1234;

    return list(parsedConfig.values())

def getIOFilePaths(pdbFile):
#
# return list of file paths in order [ pdb_file, mrc_file, crd_file, log_file, inp_file ]
#
# trying to create output dir next to input file.
# Not working. paths needing fixing.

    if not os.path.isfile(pdbFile):
        raise Exception("Error opening File")

    # suffix = str(datetime.now().strftime("%m-%d-%Y %H %M %S"))
    suffix = ''
    inputFile = pdbFile
    inputDir = os.path.dirname(pdbFile) + '/'
    inputName = os.path.splitext(os.path.basename(inputFile))[0]

    outDir = inputDir + f"{inputName} simulation output {suffix}/"

    # if not os.path.exists(outDir):
    #     os.makedirs(Path(outDir))

    return simio(inputDir,inputName, outDir, inputName)

def buildCordFile(crd_file,pdb_file,sample_dimensions, optics_params, detector_params):
    # use simutils to build co-ordinate file
    x_range, y_range, numpart = define_grid_in_fov(sample_dimensions,
                                                            optics_params,
                                                            detector_params,
                                                            pdb_file=pdb_file,
                                                            Dmax=30, pad=5.)

    write_crd_file(numpart, xrange=x_range, yrange=y_range, crd_file=crd_file)

def runSimulator(inp_file,mrc_file):
    # generate command and run simulator, return mrcData
    SIMULATOR_BIN = Path(SIMULATOR) # might have to change depending on OS
    inp_file = Path(inp_file)

    cmd = '{0} {1}'.format(SIMULATOR_BIN, inp_file)
    os.system(cmd)

    return mrc2data(mrc_file = mrc_file)


def TEMSimulator(configFile, pdbFile):

    # Using more verbose comments instead of docstring for notes (for now)
    #
    #
    # Arguments -> configFile, pdbFile
    #
    # Target Output -> A particle stack with meta data (maybe look into .hdf5 )


    beam_params,detector_params,optics_params,sample_dimensions,seed  = getConfigFromYaml(configFile)
    pdb_file, mrc_file, crd_file, log_file, inp_file,h5_file = getIOFilePaths(pdbFile)
    buildCordFile(crd_file,pdb_file,sample_dimensions, optics_params, detector_params )
    #
    params_dictionary = fill_parameters_dictionary(mrc_file = mrc_file,
                                                        pdb_file = pdb_file,
                                                        particle_mrcout = mrc_file,
                                                        crd_file = crd_file,
                                                        sample_dimensions = sample_dimensions,
                                                        beam_params = beam_params,
                                                        optics_params = optics_params,
                                                        detector_params = detector_params,
                                                        log_file = log_file,
                                                        seed=1234)

    write_inp_file(inp_file=inp_file, dict_params=params_dictionary)

    mrc_data = runSimulator(inp_file,mrc_file)
    print(mrc_data.shape)

    return

def main():
    # General Notes
    # unpacking/packing arguments feels kinda complicated and prone to error.
    #TEMSimulatr() - main function. rest are helpers

    TEMSimulator(CONFIG,PDBFILE)
    # print(getIOFilePaths(PDBFILE))

if __name__ == "__main__":
    main()