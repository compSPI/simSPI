"""Wrapper for the TEM Simulator."""
import random
import string
import subprocess
from pathlib import Path

import numpy as np
import yaml
from ioSPI import micrographs

from simSPI import crd, fov, tem_inputs


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

        stream.close()

        self.sim_dict = self.get_config_from_yaml(sim_config)
        self.output_path_dict = self.generate_path_dict(**parsed_path_config)
        self.output_path_dict["local_sim_dir"] = parsed_path_config["local_sim_dir"]

        self.parameter_dict = tem_inputs.populate_tem_input_parameter_dict(
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

        stream.close()

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
    def generate_path_dict(
        pdb_file, metadata_params_file, output_dir=None, mrc_keyword=None, **kwargs
    ):
        """Return dict containing path_config paths and output files as strings.

        Parameters
        ----------
        pdb_file : str
            Relative path to the pdb file
        metadata_params_file : str
            Relative path to metadata params file
        output_dir : str, (default = None)
            Relative path to output directory
        mrc_keyword : str, (default = None)
            user-specified keyword appended to output files
        kwargs
            compatibility for arbitrary extra parameters.

        Returns
        -------
        path_dict : dict of type str to str
            Dict of file paths that includes keys:
            pdb_file
                relative path to pdb input file
            metadata_params_file
                relative path to metadata_params file
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
            star_file
                relative poth to desured output star file
        """
        path_dict = {}

        if output_dir is None:
            output_dir = str(Path(pdb_file).parent)

        pdb_keyword = Path(pdb_file).stem

        if mrc_keyword is None:
            mrc_keyword = "_" + "".join(
                random.choices(string.ascii_uppercase + string.digits, k=5)
            )

        path_dict["pdb_file"] = str(Path(pdb_file))
        path_dict["metadata_params_file"] = str(Path(metadata_params_file))
        path_dict["crd_file"] = str(
            Path(output_dir, pdb_keyword + mrc_keyword + ".txt")
        )
        path_dict["mrc_file"] = str(
            Path(output_dir, pdb_keyword + mrc_keyword + ".mrc")
        )
        path_dict["log_file"] = str(
            Path(output_dir, pdb_keyword + mrc_keyword + ".log")
        )
        path_dict["inp_file"] = str(
            Path(output_dir, pdb_keyword + mrc_keyword + ".inp")
        )
        path_dict["h5_file"] = str(Path(output_dir, pdb_keyword + mrc_keyword + ".h5"))
        path_dict["h5_file_noisy"] = str(
            Path(output_dir, pdb_keyword + mrc_keyword + "-noisy.h5")
        )
        path_dict["star_file"] = str(
            Path(output_dir, pdb_keyword + mrc_keyword + ".star")
        )

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
        self.write_inp_file()
        self.generate_metadata()

        particle_data = self.get_image_data()

        if export_particles:
            particle_data = self.extract_particles(particle_data, pad=pad)
            self.export_particle_stack(particle_data)

            if "other" in self.parameter_dict and (
                self.parameter_dict["other"].get("signal_to_noise") is not None
                or self.parameter_dict["other"].get("signal_to_noise_db") is not None
            ):
                particle_data = self.apply_gaussian_noise(particle_data)

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
        tem_inputs.write_tem_inputs_to_inp_file(
            path=self.output_path_dict["inp_file"], tem_inputs=self.parameter_dict
        )

    def get_image_data(self):
        """Run simulator and return data.

        Returns
        -------
        List containing parsed .mrc data from Simulator

        Raises
        ------
        subprocess.CalledProcessError
            Raised if shell commmand exits with non zero code.

        Notes
        -----
        This method requires a local tem_sim installation to run.
        """
        sim_executable = f"{self.output_path_dict['local_sim_dir']}"
        input_file_arg = f"{self.output_path_dict['inp_file']}"

        subprocess.run([sim_executable, input_file_arg], check=True)

        data = micrographs.read_micrograph_from_mrc(self.output_path_dict["mrc_file"])
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
        micrographs.write_data_dict_to_hdf5(self.output_path_dict["h5_file"], particles)

        if "other" in self.parameter_dict:
            noisy_particles = self.apply_gaussian_noise(particles)
            # TODO: update choosing of noisy params in output path dict to only have
            #       to add "h5_file_noisy" to path dict only when noise is in sim_config
            if "h5_file_noisy" in self.output_path_dict:
                micrographs.write_data_dict_to_hdf5(
                    self.output_path_dict["h5_file_noisy"], noisy_particles
                )
            else:
                micrographs.write_data_dict_to_hdf5(
                    self.output_path_dict["h5_file"][:-3]
                    + "-noisy"
                    + self.output_path_dict["h5_file"][-3:],
                    noisy_particles,
                )

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

        with open(Path(self.output_path_dict["metadata_params_file"]), "r") as stream:
            metadata_fields = yaml.safe_load(stream)

        stream.close()

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

            f.write("\n")
            f.write("loop_\n")
            f.write("_x\n")
            f.write("_y\n")
            f.write("_z\n")
            f.write("_phi\n")
            f.write("_theta\n")
            f.write("_psi\n")

            for coord in particle_metadata:
                f.write(
                    "{0[0]:13.4f}{0[1]:13.4f}{0[2]:13.4f}"
                    "{0[3]:13.4f}{0[4]:13.4f}"
                    "{0[5]:13.4f}\n".format(coord)
                )

        f.close()

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

        f.close()

        for i, line in enumerate(lines):
            if i >= 4:
                rotation_metadata.append([float(x) for x in line.split()[:]])

        return rotation_metadata
