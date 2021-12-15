"""Wrapper for the TEM Simulator."""
import os
import random
import string
from pathlib import Path

import cryoemio
import matplotlib.pyplot as plt
import numpy as np
import simutils
import yaml
from ioSPI import cryoemio as io

from simSPI import crd, fov


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

    # TODO: run DOES NOT EXIST in master, and is not up to date.
    #       requires deprecation (by more careful hands than my own).
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

        self.generate_metadata()

        micrograph_data = self.get_image_data(display_data=display_data)
        particle_data = self.extract_particles(
            micrograph_data,
            0.0,
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
    def classify_input_config(raw_params):
        """Take dictionary of unordered parameters and groups them into lists.

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
            "molecular_model": ["voxel_size_nm", "particle_name", "particle_mrcout"],
            "specimen_grid_params": [
                "hole_diameter_nm",
                "hole_thickness_center_nm",
                "hole_thickness_edge_nm",
            ],
            "beam_parameters": [
                "voltage_kv",
                "energy_spread_v",
                "electron_dose_e_nm2",
                "electron_dose_std_e_per_nm2",
            ],
            "optics_parameters": [
                "magnification",
                "spherical_aberration_mm",
                "chromatic_aberration_mm",
                "aperture_diameter_um",
                "focal_length_mm",
                "aperture_angle_mrad",
                "defocus_um",
                "defocus_syst_error_um",
                "defocus_nonsyst_error_um",
                "optics_defocusout",
            ],
            "detector_parameters": [
                "detector_nx_px",
                "detector_ny_px",
                "detector_pixel_size_um",
                "average_gain_count_per_electron",
                "noise",
                "detector_q_efficiency",
                "mtf_params",
            ],
        }

        classified_sim_params = {}

        for param_type, param_order in sim_params_structure.items():
            if param_type != "detector_parameters":
                classified_sim_params[param_type] = [
                    raw_params[param_type].get(key) for key in param_order
                ]
            elif param_type == "detector_parameters":
                ordered_params = [
                    raw_params[param_type].get(key) for key in param_order
                ]
                flattened_params = []

                for i in range(6):
                    flattened_params.append(ordered_params[i])
                for i in range(5):
                    flattened_params.append(ordered_params[6][i])

                classified_sim_params[param_type] = flattened_params

        return classified_sim_params

    @staticmethod
    def generate_path_dict(pdb_file, output_dir=None, mrc_keyword=None):
        """Return the paths to pdb, crd, log, inp, and h5 files as strings.

        Parameters
        ----------
        pdb_file : str
            Relative path to pdb file
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
            h5_file_noisy
                relative path to desired output h5 file with noise
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
        path_dict["h5_file_noisy"] = output_file_path + "-noisy.h5"

        return path_dict

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

        data = cryoemio.mrc2data(self.output_path_dict["mrc_file"])
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

        parameter_dict = simutils.fill_parameters_dictionary(
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
        io.write_inp_file(
            inp_file=self.output_path_dict["inp_file"], dict_params=self.parameter_dict
        )

    def extract_particles(
        self, micrograph, pad, export_particles=True, display_data=False
    ):
        """Extract particle data from micrograph.

        Parameters
        ----------
        micrograph : arr
            Array containing TEM-simulator micrograph output
        pad : double
            Pad to be added to maximal dimension of the object read from pdb_file
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
            pad=pad,
        )

        if display_data:
            self.view_particles(particles, ncol=5)

        if export_particles:
            cryoemio.data_and_dic2hdf5(particles, self.output_path_dict["h5_file"])

        return particles

    def apply_gaussian_noise(self, particles):
        """Apply gaussian noise to particle data.

        Returns
        -------
        noisy_particles : arr
            Individual particle data with gaussian noise applied
        """
        variance = np.var(particles)
        if "other" not in self.parameter_dict:
            return particles.copy()
        snr = 1.0
        try:
            snr = self.parameter_dict["other"]["signal_to_noise"]
        except KeyError:
            pass
        try:
            snr_db = self.parameter_dict["other"]["signal_to_noise_db"]
            snr = 10 ** (snr_db / 10)
        except KeyError:
            pass
        scale = np.sqrt(variance / snr)
        noisy_particles = np.random.normal(particles, scale)
        return noisy_particles

    def generate_metadata(self):
        """Generate metadata associated with picked particles from simulator.

        Notes
        -----
        Exports particle metadata in .star file to output directory specified
        in user config file.
        """
        particle_metadata = self.retrieve_rotation_metadata(
            self.output_path_dict["crd_file"]
        )

        file_name = (
            self.path_dict["pdb_keyword"]
            + self.path_dict["micrograph_keyword"]
            + ".star"
        )

        with open(self.path_dict["output_dir"] + file_name, "w") as f:
            for key, value in self.raw_sim_dict.items():
                f.write(f"{key}\n")
                for key0, value0 in value.items():
                    if type(value0) is list:
                        f.write("_" + "{0:24}{1}\n".format(key0, value0))
                    else:
                        f.write("_" + "{0:24}{1:>15}\n".format(key0, value0))
                f.write("\n")

            f.write("particle_rotation_angles\n")
            f.write("loop_\n")
            f.write("_phi\n")
            f.write("_theta\n")
            f.write("_psi\n")
            for angle in particle_metadata:
                f.write("{0[0]:13.4f}{0[1]:13.4f}{0[2]:13.4f}\n".format(angle))

    @staticmethod
    def retrieve_rotation_metadata(path):
        """Retrieve particle rotation data from pre-generated simulator crd file.

        Parameters
        ----------
        path : str
            String specifying path to crd file generated during simulation.

        Returns
        -------
        rotation_metadata : array-like, shape=[..., 3]
            N x 3 matrix representing the rotation angles , phi, theta, psi, of
            each particle in stack.
        """
        rotation_metadata = []
        lines = []
        with open(path) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if 4 < i:
                rotation_metadata.append([float(x) for x in line.split()[3:]])

        f.close()
        return rotation_metadata

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

        if "other" in self.parameter_dict:
            noisy_particles = self.apply_gaussian_noise(particles)
            if "h5_file_noisy" in self.output_path_dict:
                io.data_and_dic2hdf5(
                    noisy_particles,
                    self.output_path_dict["h5_file_noisy"],
                )
            else:
                io.data_and_dic2hdf5(
                    noisy_particles,
                    self.output_path_dict["h5_file"][:-3]
                    + "-noisy"
                    + self.output_path_dict["h5_file"][-3:],
                )

    @staticmethod
    def view_particles(data, slicing=(1, 1, 1), figsize=1, ncol=5):
        """Render picked particles in grid.

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
