"""Wrapper for the TEM Simulator."""
import random
import string
import subprocess
from pathlib import Path

import numpy as np
import yaml
from ioSPI import micrographs

from simSPI import crd, fov, tem_distribution_utils, tem_inputs


class TEMSimulator:
    """Wrapper for the TEM Simulator.

    Parameters
    ----------
    path_config : str
        Relative path to YAML file containing file paths for TEM Simulator.
    sim_config : str
        Relative path to YAML file containing simulator paths for TEM Simulator.

    Notes
    -----
    defocus_distribution_samples are in um.
    """

    def __init__(self, path_config, sim_config):

        with open(path_config, "r") as stream:
            parsed_path_config = yaml.safe_load(stream)

        self.sim_dict = self.get_config_from_yaml(sim_config)
        self.output_path_dict = self.generate_path_dict(**parsed_path_config)
        self.output_path_dict["local_sim_dir"] = parsed_path_config["local_sim_dir"]

        self.parameter_dict = tem_inputs.populate_tem_input_parameter_dict(
            sim_config,
            self.output_path_dict["mrc_file"],
            self.output_path_dict["pdb_file"],
            self.output_path_dict["crd_file"],
            self.output_path_dict["log_file"],
            self.output_path_dict["defocus_file"],
        )

        self.defocus_distribution_samples = []

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

    def generate_simulator_inputs(self, pad=5):
        """Generate input files for TEM simulator.

        Parameters
        ----------
        pad : double, (default = 5)
            Pad to be added to maximal dimension of the object read from pdb_file

        Returns
        -------
        particles : arr
            Individual particle data extracted from micrograph
        """
        self.create_crd_file(pad)
        self.create_defocus_file()  # TODO: private?
        self.create_inp_file()

    def create_defocus_file(self):
        """Sample defocus parameters and generate corresponding defocus file."""
        defocus_params = self.parameter_dict["ctf"]
        n_samples = self.parameter_dict["geometry"]["n_tilts"]

        distribution_generator = tem_distribution_utils.DistributionGenerator(
            defocus_params["distribution_type"],
            defocus_params["distribution_parameters"],
        )
        samples = distribution_generator.draw_samples_1d(n_samples).tolist()
        samples = [round(num, 4) for num in samples]

        tem_inputs.write_tem_defocus_file_from_distribution(
            self.output_path_dict["defocus_file"], samples
        )

        self.defocus_distribution_samples = samples

    def create_crd_file(self):
        """Format and write molecular model data to crd_file for use in TEM-simulator.
        """
        pad = 5  # TODO: get this from sim config
        x_range, y_range, num_part = fov.define_grid_in_fov(
            self.sim_dict["optics_parameters"],
            self.sim_dict["detector_parameters"],
            self.output_path_dict["pdb_file"],
            pad = pad,
        )

        crd.write_crd_file(
            num_part,
            xrange = x_range,
            yrange = y_range,
            crd_file = self.output_path_dict["crd_file"],
        )

    def create_inp_file(self):
        """Write simulation parameters to .inp file for use by the TEM-simulator.

        The .inp files contain the parameters controlling the simulation. These are text
        files whose format is described in the TEM Simulator documentation. They contain
        component headings which divide the files into different sections (e.g.
        different particles) and parameter assignments of the form
        "<parameter> = <value>".
        """
        tem_inputs.write_tem_inputs_to_inp_file(
            path = self.output_path_dict["inp_file"], tem_inputs = self.parameter_dict
        )

    def run_simulator(self):
        """Run TEM simulator.

        Raises
        ------
        subprocess.CalledProcessError
            Raised if shell command exits with non-zero code.

        Notes
        -----
        This method requires a local tem_sim installation to run.
        """
        sim_executable = f"{self.output_path_dict['local_sim_dir']}"
        input_file_arg = f"{self.output_path_dict['inp_file']}"

        subprocess.run([sim_executable, input_file_arg], check = True)

    def parse_simulator_data(self):
        """Extract particle data from micrograph.

        Parameters
        ----------

        Returns
        -------
        particles : arr #TODO:docstring
            Individual particle data extracted from micrograph
        """
        pad = 5.0  # TODO: get from sim_condif
        micrograph_data = micrographs.read_micrograph_from_mrc(
            self.output_path_dict["mrc_file"]
        )
        particle_stacks = []

        for i in range(self.parameter_dict["geometry"]["n_tilts"]):
            particles = fov.micrograph2particles(
                micrograph_data[i],
                self.sim_dict["optics_parameters"],
                self.sim_dict["detector_parameters"],
                pdb_file = self.output_path_dict["pdb_file"],
                pad = pad,
            )
            particle_stacks.append(particles)

        particle_stacks = self.apply_gaussian_noise(particle_stacks)
        return micrograph_data, np.array(particle_stacks)

    def apply_gaussian_noise(self, particle_stacks):
        """Apply gaussian noise to particle data.

        Parameters
        ----------
        particle_stacks : arr
            Individual particle data extracted from micrograph

        Returns
        -------
        noisy_particles : arr
            Individual particle data with gaussian noise applied
        """
        noisy_particles = []

        if "other" not in self.parameter_dict:
            return particle_stacks

        for i in range(self.parameter_dict["geometry"]["n_tilts"]):
            variance = np.var(particle_stacks[i])
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
            noisy_particles.append(np.random.normal(particle_stacks[i], scale))

        return np.array(noisy_particles)

    def export_simulated_data(self, particles):
        """Export extracted particle data to h5 file and generate meta data.

        Parameters
        ----------
        particles : arr
            Individual particle data extracted from micrograph #TODO: docstring and test

        """
        flattened_particles = np.ndarray.flatten(particles)
        micrographs.write_data_dict_to_hdf5(
            self.output_path_dict["h5_file"], flattened_particles
        )

        self.generate_metadata()

        return self.output_path_dict["h5_file"], self.output_path_dict["star_file"]

    def generate_metadata(self):
        """Generate metadata associated with picked particles from simulator.

        Notes
        -----
        Exports particle metadata in .star file to output directory specified
        in user config file.
        """
        particle_metadata = tem_inputs.retrieve_rotation_metadata(
            self.output_path_dict["crd_file"]
        )

        with open(Path(self.output_path_dict["metadata_params_file"]), "r") as stream:
            metadata_fields = yaml.safe_load(stream)

        n_samples = self.parameter_dict["geometry"]["n_tilts"]

        with open(self.output_path_dict["star_file"], "w") as f:
            mtf_params = {}
            for key, value in self.parameter_dict.items():
                f.write(f"data_{key}\n")
                for key0, value0 in value.items():
                    if key0[:3] == "mtf":
                        mtf_params[key0[4:]] = value0
                    else:
                        key_fixed = (
                            metadata_fields[key0] if key0 in metadata_fields else key0
                        )
                        if type(value0) is list:
                            f.write("_" + "{0:24}{1}\n".format(key_fixed, value0))
                        else:
                            f.write("_" + "{0:24}{1:>15}\n".format(key_fixed, value0))
                f.write("\n")

            f.write("loop_\n")
            f.write("_mtf_params\n")
            for c in ("a", "b", "c", "alpha", "beta"):
                if c in mtf_params:
                    f.write(f"{mtf_params[c]:13.4f}\n")

            for i in range(n_samples):

                f.write(f"particle_rotation_angles: {i + 1}\n")
                f.write("loop_\n")
                f.write("_defocus\n")
                f.write("_x\n")
                f.write("_y\n")
                f.write("_z\n")
                f.write("_phi\n")
                f.write("_theta\n")
                f.write("_psi\n")

                for coord in particle_metadata:
                    f.write(
                        "{0:13.4f}{1[0]:13.4f}{1[1]:13.4f}{1[2]:13.4f}"
                        "{1[3]:13.4f}{1[4]:13.4f}"
                        "{1[5]:13.4f}\n".format(
                            self.defocus_distribution_samples[i], coord
                        )
                    )
