"""Wrapper for the TEM Simulator."""
import os
import random
import string
from pathlib import Path

import numpy as np
import yaml
from ioSPI.ioSPI import cryoemio as io

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

        with open(path_config, "r") as stream:
            parsed_path_config = yaml.safe_load(stream)

        self.output_path_dict = self.generate_path_dict(**parsed_path_config)
        self.output_path_dict["local_sim_dir"] = parsed_path_config["local_sim_dir"]

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
    def generate_path_dict(pdb_file, output_dir=None, mrc_keyword=None, **kwargs):
        """Return the paths to pdb, crd, log, inp, and h5 files as strings.

        Parameters
        ----------
        pdb_file : str
            Relative path to the pdb file
        output_dir : str, (default = None)
            Relative path to output directory
        mrc_keyword : str, (default = None)
            user-specified keyword appended to output files
        kwargs
            Arbitrary keyword arguments.

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

        path_dict["pdb_file"] = str(Path(pdb_file))
        path_dict["crd_file"] = str(Path(output_dir, mrc_keyword + ".txt"))
        path_dict["mrc_file"] = str(Path(output_dir, mrc_keyword + ".mrc"))
        path_dict["log_file"] = str(Path(output_dir, mrc_keyword + ".log"))
        path_dict["inp_file"] = str(Path(output_dir, mrc_keyword + ".inp"))
        path_dict["h5_file"] = str(Path(output_dir, mrc_keyword + ".h5"))
        path_dict["h5_file_noisy"] = str(Path(output_dir, mrc_keyword + "-noisy.h5"))

        return path_dict

    def run(self, pad=5, export_particles=False):
        """Run TEM simulator on input file and produce particle stacks with metadata.

        Parameters
        ----------
        pad : double, (default = 5)
            Pad to be added to maximal dimension of the object read from pdb_file
        export_particles : boolean, (default = False)
            Particle data exported to .h5 if True.

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """
        self.create_crd_file(pad)
        self.create_inp_file()

        micrograph_data = self.get_image_data()
        particle_data = self.extract_particles(micrograph_data, pad=pad)

        if "other" in self.parameter_dict:
            if (
                self.parameter_dict["other"].get("signal_to_noise") is not None
                or self.parameter_dict["other"].get("signal_to_noise_db") is not None
            ):
                particle_data = self.apply_gaussian_noise(particle_data)

        if export_particles:
            self.export_particle_stack(particle_data)

        return particle_data

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

    def get_image_data(self):
        """Run simulator and return data.

        Returns
        -------
        List containing parsed .mrc data from Simulator

        Notes
        -----
        This method requires a local tem_sim installation to run.
        """
        os.system(
            "{} {}".format(
                self.output_path_dict["local_sim_dir"],
                self.output_path_dict["inp_file"],
            )
        )

        data = io.mrc2data(self.output_path_dict["mrc_file"])
        micrograph = data[0, ...]

        return micrograph

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

    def apply_gaussian_noise(self, particles):
        """Apply gaussian noise to particle data.

        Parameters
        ----------
        particles : arr
            Individual particle data extracted from micrograph

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
