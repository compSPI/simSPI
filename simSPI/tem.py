"""Wrapper for the TEM Simulator."""
import os

import crd
import fov
import io_cryo
import matplotlib.pyplot as plt
import numpy as np
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
        self.path_dict = self.get_raw_config_from_yaml(path_config)
        self.raw_sim_dict = self.get_raw_config_from_yaml(sim_config)

        self.output_path_dict = self.generate_path_dict(self.path_dict)
        self.sim_dict = self.classify_sim_params(self.raw_sim_dict)

        self.parameter_dict = self.generate_parameter_dict(
            self.output_path_dict, self.sim_dict, self.raw_sim_dict, seed=1234
        )

        self.placeholder = 0

    def run(self, display_data=False, export_particles=True):
        """Run TEM simulator on input file and produce particle stacks with metadata.

        Parameters
        ----------
        display_data : bool
            Flag to determine whether to export particle data after extraction
        export_particles : bool
            Flag to determine whether to export extracted particle data

        Returns
        -------
        particle_data : arr
            Individual particle data extracted from micrograph
        """
        self.create_crd_file(pad=5)
        self.create_inp_file()

        micrograph_data = self.get_image_data(display_data=display_data)
        particle_data = self.extract_particles(
            micrograph_data,
            export_particles=export_particles,
            display_data=display_data,
        )

        return particle_data

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
            Dictionary containing grouped parameters from yaml config file
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
                    raw_sim_params[param_type].get(key) for key in param_order
                ]
            elif param_type == "detector_parameters":
                ordered_list = [
                    raw_sim_params[param_type].get(key) for key in param_order
                ]
                classified_sim_params[param_type] = flatten_detector_array(ordered_list)

        return classified_sim_params

    @staticmethod
    def generate_path_dict(path_dict):
        """Return the paths to pdb, crd, log, inp, and h5 files as strings.

        Parameters
        ----------
        path_dict : dict of type str to str
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

        output_file_path = (
            path_dict["output_dir"]
            + path_dict["pdb_keyword"]
            + path_dict["micrograph_keyword"]
        )

        file_path_dict["pdb_file"] = (
            path_dict["pdb_dir"] + path_dict["pdb_keyword"] + ".pdb"
        )
        file_path_dict["crd_file"] = output_file_path + ".txt"
        file_path_dict["mrc_file"] = output_file_path + ".mrc"
        file_path_dict["log_file"] = output_file_path + ".log"
        file_path_dict["inp_file"] = output_file_path + ".inp"
        file_path_dict["h5_file"] = output_file_path + ".h5"

        return file_path_dict

    def create_crd_file(self, pad):
        """Format and write molecular model data to crd_file for use in TEM-simulator.

        Parameters
        ----------
        pad : double
            Pad to be added to maximal dimension of the object read from pdb_file

        Reference
        ---------
        Leverages methods developed in:
            https://github.com/slaclab/cryoEM-notebooks/blob/master/src/simutils.py
        """
        x_range, y_range, num_part = fov.define_grid_in_fov(
            self.sim_dict["optics_parameters"],
            self.sim_dict["detector_parameters"],
            self.output_path_dict["pdb_file"],
            dmax=30,
            pad=pad,
        )

        crd.write_crd_file(
            num_part,
            xrange=x_range,
            yrange=y_range,
            crd_file=self.output_path_dict["crd_file"],
        )

    def get_image_data(self, display_data=False):
        """Run simulator and return data.

        Parameters
        ----------
        display_data : bool
            Flag to determine whether to display particle data

        Returns
        -------
        List containing parsed .mrc data from Simulator

        Reference
        ---------
        Leverages methods developed in https://github.com/slaclab/cryoEM-notebooks
        """
        os.system(
            "{} {}".format(
                self.path_dict["simulator_dir"], self.output_path_dict["inp_file"]
            )
        )

        data = io_cryo.mrc2data(self.output_path_dict["mrc_file"])
        micrograph = data[0, ...]

        if display_data:
            # fig = plt.figure(figsize=(18, 12))
            plt.imshow(micrograph, origin="lower", cmap="Greys")
            plt.colorbar()
            plt.show()

        return micrograph

    @staticmethod
    def generate_parameter_dict(output_path_dict, sim_dict, raw_sim_dict, seed=1234):
        """Generate class variable parameter_dict from cleaned user config data.

        Output is for use in class methods.

        Parameters
        ----------
        output_path_dict : dict
            Dictionary containing file paths to input pdb file and simulator generated
            output files.
        sim_dict : dict
            Dictionary of grouped simulator parameters.
        raw_sim_dict : dict
            Dictionary containing grouped parameters from yaml config file.
        seed : int
            Integer seed passed to TEM-Simulator through inp_file.

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph.

        Reference
        ---------
        Leverages methods developed in:
            https://github.com/slaclab/cryoEM-notebooks/blob/master/src/simutils.py
        """
        mrc_file = output_path_dict["mrc_file"]
        pdb_file = output_path_dict["pdb_file"]
        crd_file = output_path_dict["crd_file"]
        log_file = output_path_dict["log_file"]

        particle_mrcout = raw_sim_dict["molecular_model"]["particle_mrcout"]

        sample_dimensions = sim_dict["specimen_grid_params"]
        beam_params = sim_dict["beam_parameters"]
        optics_params = sim_dict["optics_parameters"]
        detector_params = sim_dict["detector_parameters"]

        parameter_dict = io_cryo.fill_parameters_dictionary(
            mrc_file=mrc_file,
            pdb_file=pdb_file,
            particle_mrcout=particle_mrcout,
            crd_file=crd_file,
            sample_dimensions=sample_dimensions,
            beam_params=beam_params,
            optics_params=optics_params,
            detector_params=detector_params,
            log_file=log_file,
            seed=seed,
        )

        return parameter_dict

    def create_inp_file(self):
        """Write simulation parameters to .inp file for use by the TEM-simulator.

        The .inp files contain the parameters controlling the simulation. These are text
        files whose format is described in the TEM Simulator documentation. They contain
        component headings which divide the files into different sections (e.g.
        different particles) and parameter assignments of the form
        "<parameter> = <value>".

        Reference
        ---------
        Leverages methods developed in:
            https://github.com/slaclab/cryoEM-notebooks/blob/master/src/simutils.py
        """
        inp_file = self.output_path_dict["inp_file"]
        io_cryo.write_inp_file(inp_file=inp_file, dict_params=self.parameter_dict)

    def extract_particles(self, micrograph, export_particles=True, display_data=False):
        """Extract particle data from micrograph.

        Parameters
        ----------
        micrograph : arr
            Array containing TEM-simulator micrograph output
        export_particles : bool
            Boolean flag to determine whether to export particle data to h5 file
        display_data : bool
            Boolean flag to determine whether to display generated particle data

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph

        Reference
        ---------
        Leverages methods developed in:
            https://github.com/slaclab/cryoEM-notebooks/blob/master/src/simutils.py
            https://github.com/slaclab/cryoEM-notebooks/blob/master/src/cryoemio.py
        """
        particles = fov.micrograph2particles(
            micrograph,
            self.sim_dict["optics_parameters"],
            self.sim_dict["detector_parameters"],
            pdb_file=self.output_path_dict["pdb_file"],
            dmax=30,
            pad=5.0,
        )

        if display_data:
            self.view_particles(particles, ncol=5)

        if export_particles:
            io_cryo.data_and_dic_2hdf5(
                particles, self.output_path_dict["h5_file"], dic=self.parameter_dict
            )

        return particles

    @staticmethod
    def view_particles(data, slicing=(1, 1, 1), figsize=1, ncol=5):
        """Extract particle data from micrograph.

        Parameters
        ----------
        data : arr
            Array containing TEM-simulator micrograph output
        slicing : tuple

        figsize : int
            Integer scaling factor for rendered particle figures
        ncol : int
            Integer number of columns in particle view

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """
        view = data[:: slicing[0], :: slicing[1], :: slicing[2]]
        figsize = int(figsize * ncol)
        nrow = np.ceil(view.shape[0] / ncol)
        fig = plt.figure(figsize=(ncol * figsize, nrow * figsize))

        for i in np.arange(view.shape[0]):
            fig.add_subplot(int(nrow), int(ncol), int(i + 1))
            plt.imshow(view[i], cmap="Greys")

        plt.tight_layout()
        plt.show()


def main():
    """Return 1 as a placeholder."""
    t = TEMSimulator(
        "../path_config.yaml",
        "../sim_config.yaml",
    )
    t.run(display_data=True, export_particles=True)
    return 1


if __name__ == "__main__":
    main()
