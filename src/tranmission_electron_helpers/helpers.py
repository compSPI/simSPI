
# The following methods are largely refactored from simultils.py, their headers and docstrings are below.

def generate_file_paths():
    '''
    Returns the paths to relevant to-be-generated pdb, crd, log, inp, and h5 files as strings

        Parameters:
            PDB_DIR (str): relative path to directory containing PDB files
            PDB_KEYWORD (str): ID of PDB file found in pdbdir
            OUTPUT_DIR (str): relative path to output file directory
            MRC_KEYWORD (str): user-specified keyword

        Returns:
            pdb_file (str): relative path to pdb input file
            mrc_file (str): relative path to desired mrc output file
            crd_file (str): relative path to desired crd output file
            log_file (str): relative path to desired log output file
            inp_file (str): relative path to desired inp output file
            h5_file  (str): relative path to desired h5 output file
    '''
    return


def classify_input_config():
    '''
    Converts simulator inputs in config file to array format

        Parameters:
            sim_config (dict): dictionary containing simulator config data, converted from user-passed yaml file

        Returns:
            sim_param_arrays (dict): dictionary containing the following arrays:
                sample_dimensions (arr): arr containing the specimen grid parameters
                beam_params (arr):       arr containing the beam parameters
                optics params (arr):     arr containing the optic parameters
                detector_params (arr):   arr containing the detector parameters
    '''
    return

def create_crd_file():
    '''
    Formats and writes molecular model data to crd_file for use in TEM-simulator.

        Parameters:
            sim_param_arrays (dict): dictionary containing arrays of simulation parameters
            pdb_file (str): relative path to pdb input file
            crd_file (str): relative path to desired crd_output file
            pad (double): pad to be added to maximal dimension of the object read from pdb_file
    '''
    return

def generate_parameters_dictionary():
    '''
    Compiles all relevant experiment data into .inp friendly file for use in TEM-simulator.

        Parameters:
            pdb_file (str): relative path to pdb input file
            mrc_file (str): relative path to desired mrc output file
            crd_file (str): relative path to desired mrc output file
            log_file (str): relative path to desired mrc output file
            particle_mrcout (bool): flag for optional volume map of sample
            sim_param_arrays (dict): dictionary containing relevant simulation parameters
            seed (int):

        Returns:
            param_dictionary (dict): .inp friendly dictionary containing simulation input parameters.
    '''
    return

def write_inp_file():
    '''
    Writes simulation parameters to .inp file for use by the TEM-simulator.

        Parameters:
            inp_file (str): relative path to .inp file.
            dict_params (dict): .inp friendly dictionary containing simulation input parameters.
    '''
    return

def parse_mrc_data():
    '''
    Reads TEM-simulator .mrc ouput and converts into array.

        Parameters:
            mrc_file (str): relative path to mrc output file.

        Returns:
            micrograph_data (arr): array containing TEM-simulator micrograph output.
    '''
    return

def extract_particles():
    '''
    Extracts individual particles from full TEM-simulator generated micrograph.

        Parameters:
            micrograph (arr): array containing TEM-simulator micrograph output
            sim_param_arrays (dict): dictionary containing arrays of simulation parameters
            pdb_file (str): relative path to .pdb input file
            pad (double): pad to be added to maximal dimension of the object read from pdb_file

        Returns:
            particles (arr): individual particle data extracted from micrograph
    '''
    return

def export_particle_stack():
    '''
    Exports extracted particle data to h5 file.
        Parameters:
            particles (arr): individual particle data extracted from micrograph
            h5_file (str): relative path to .h5 output file
            params_dict (dict): .inp friendly dictionary containing simulation input parameters.
    '''
    return

