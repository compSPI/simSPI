"""Wrapper for the TEM Simulator."""
import random
import string
from pathlib import Path

import crd
import fov
import yaml
from ioSPI.ioSPI import cryoemio as io


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
        self.output_path_dict = self.generate_path_dict(path_config)
        self.sim_dict = self.get_config_from_yaml(sim_config)

        self.parameter_dict = io.fill_parameters_dictionary(
            sim_config,
            self.output_path_dict["mrc_file"],
            self.output_path_dict["pdb_file"],
            self.output_path_dict["crd_file"],
            self.output_path_dict["log_file"],
        )

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
        """Take dictionary of unordered parameters and groups them into lists.

        Parameters
        ----------
        raw_params : dict of type str to {str,bool,int}
            Dictionary of simulator parameters
        Returns
        -------
        classified_params : dict of type str to {str,bool,int,list}
            Dictionary of grouped parameters
        """
        sim_params_structure = {
            "molecular_model": ["voxel_size", "particle_name", "particle_mrcout"],
            "specimen_grid_params": [
                "hole_diameter",
                "hole_thickness_center",
                "hole_thickness_edge",
            ],
            "beam_parameters": [
                "voltage",
                "energy_spread",
                "electron_dose",
                "electron_dose_std",
            ],
            "optics_parameters": [
                "magnification",
                "spherical_aberration",
                "chromatic_aberration",
                "aperture_diameter",
                "focal_length",
                "aperture_angle",
                "defocus",
                "defocus_syst_error",
                "defocus_nonsyst_error",
                "optics_defocusout",
            ],
            "detector_parameters": [
                "detector_Nx",
                "detector_Ny",
                "detector_pixel_size",
                "detector_gain",
                "noise",
                "detector_Q_efficiency",
                "MTF_params",
            ],
        }

        def flatten_detector_array(arr):

            flattened_params = []

            for i in range(6):
                flattened_params.append(arr[i])

            for i in range(5):
                flattened_params.append(arr[6][i])

            return flattened_params

        classified_sim_params = {}

        for param_type, param_order in sim_params_structure.items():
            if param_type != "detector_parameters":
                classified_sim_params[param_type] = [
                    raw_params[param_type].get(key) for key in param_order
                ]
            elif param_type == "detector_parameters":
                ordered_list = [raw_params[param_type].get(key) for key in param_order]
                classified_sim_params[param_type] = flatten_detector_array(ordered_list)

        return classified_sim_params

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
        path_dict = {}

        if output_dir is None:
            output_dir = str(Path(pdb_file).parent)

        if mrc_keyword is None:
            mrc_keyword = str(Path(pdb_file).stem) + "".join(
                random.choices(string.ascii_uppercase + string.digits, k=5)
            )
        output_file_path = output_dir + mrc_keyword

        path_dict["pdb_file"] = pdb_file
        path_dict["crd_file"] = output_file_path + ".txt"
        path_dict["mrc_file"] = output_file_path + ".mrc"
        path_dict["log_file"] = output_file_path + ".log"
        path_dict["inp_file"] = output_file_path + ".inp"
        path_dict["h5_file"] = output_file_path + ".h5"

        return path_dict

    def create_crd_file(self, pad):
        """Format and write molecular model data to crd_file for use in TEM-simulator.

        Parameters
        ----------
        pad : double
            Pad to be added to maximal dimension of the object read from pdb_file
        """
        x_range, y_range, num_part = fov.define_grid_in_fov(
            self.sim_dict["optics_parameters"],
            self.sim_dict["detector_parameters"],
            self.output_path_dict["pdb_file"],
            pad=pad,
        )

        crd.write_crd_file(
            num_part,
            xrange=x_range,
            yrange=y_range,
            crd_file=self.output_path_dict["crd_file"],
        )

    def write_inp_file(self):
        """Write simulation parameters to .inp file for use by the TEM-simulator.

        The .inp files contain the parameters controlling the simulation. These are text
        files whose format is described in the TEM Simulator documentation. They contain
        component headings which divide the files into different sections (e.g.
        different particles) and parameter assignments of the form
        "<parameter> = <value>".
        """
        io.write_inp_file(
            inp_file=self.output_path_dict["inp_file"], dict_params=self.parameter_dict
        )

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
        particles = fov.micrograph2particles(
            micrograph,
            self.sim_dict["optics_parameters"],
            self.sim_dict["detector_parameters"],
            pdb_file=self.output_path_dict["pdb_file"],
            pad=pad,
        )

        return particles

    def export_particle_stack(self, particles):
        """Export extracted particle data to h5 file.

        Parameters
        ----------
        particles : arr
            Individual particle data extracted from micrograph

        """
        io.data_and_dic2hdf5(
            particles,
            self.output_path_dict["h5_file"],
        )
