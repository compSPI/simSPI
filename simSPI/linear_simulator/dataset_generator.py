"""runs dataset generator and saves it."""
import os

import mrcfile
import pandas as pd
import starfile
import torch

from .modules import simulator
from .params_utils import ParamsFactory
from .starfile_utils import return_names, starfile_data

"""Module to generate and save dataset (including metadata) in the output directory."""


class dataset_gen(torch.nn.Module):
    """class to generate and save dataset (including metadata) in the output directory.

    Parameters
    ----------
    config: class
        Class containing parameters of the dataset generation and simulator.

    """

    def __init__(self, config):
        super(dataset_gen, self).__init__()

        self.config = config
        self.simulator = simulator(config)  # instantiating the simulator
        self.dataframe = []
        self.init_dir()  # initializing the output folder
        self.print_statements()
        self.params_generator = ParamsFactory.get_params_generator(
            config
        )  # instantiating params generator

    def run(self):
        """Generate a chunk of projection and save it."""
        for iterations in range(
            self.config.datasetSize // self.config.chunks
        ):  # TODO: last few images will be left
            rot_params, ctf_params, shift_params = self.params_generator.get_params()
            projections = self.simulator(rot_params, ctf_params, shift_params)
            self.save_mrc(projections, iterations)
            self.starfile_data(rot_params, ctf_params, shift_params, iterations)
        self.save_starfile()
        self.save_configfile()

    def save_mrc(self, projections, iterations):
        """Save the projection chunks as an mrcs file in the output directory.

        Parameters
        ----------
        projections: torch.Tensor
            projection from the simulator (chunks,1, sidelen, sidelen)
        iterations: int
            iteration number of the loop. Used in naming the mrcs file.`
        """
        image_path = os.path.join(
            self.config.output_path, str(iterations).zfill(4) + ".mrcs"
        )
        projections = projections.detach().cpu().numpy()
        with mrcfile.new(image_path, overwrite="True") as m:
            m.set_data(projections.astype("float32"))

    def save_starfile(self):
        """Save the metadata in a starfile in the output directory."""
        df = pd.DataFrame(
            data=self.dataframe,
            index=[idx for idx in range(len(self.dataframe))],
            columns=return_names(self.config),
        )
        starfile.write(
            df, os.path.join(self.config.output_path, "Simulated.star"), overwrite=True
        )
        print(f"Saving star file with the parameters of the generated dataset..")

    def save_configfile(self):
        """Save the config as .txt and .cfg in the output directory."""
        with open(os.path.join(self.config.output_path, "config.cfg"), "w") as fp:
            self.config.config.write(fp)
        with open(os.path.join(self.config.output_path, "config.txt"), "w") as fp:
            self.config.config.write(fp)
        print(f"Saving configs..")

    def init_dir(self):
        """Make the output directory and puts the path in the config.output_path."""
        self.config.output_path = os.path.join(os.getcwd(), self.config.output_path)

        self.config.output_path = os.path.join(self.config.output_path, "Datasets")
        if os.path.exists(self.config.output_path) is False:
            os.mkdir(self.config.output_path)

        self.config.output_path = os.path.join(
            self.config.output_path, self.config.name
        )
        if os.path.exists(self.config.output_path) is False:
            os.mkdir(self.config.output_path)

    def starfile_data(self, rot_params, ctf_params, shift_params, iterations):
        """Append the dataframe with the simulator parameters.

        Parameters
        ----------
        rot_params: dict of type str to {tensor}
            Dictionary of rotation parameters for a projection chunk
        ctf_params: dict of type str to {tensor}
            Dictionary of Contrast Transfer Function (CTF)
            parameters for a projection chunk
        shift_params: dict of type str to {tensor}
            Dictionary of shift parameters for a projection chunk
        iterations: int
            iteration number of the loop. Used in naming the mrcs file.

        Returns
        -------
        dataframe: list
            list containing the metadata of the projection chunks which
            is then used to save the starfile.
        """
        self.dataframe = starfile_data(
            self.dataframe,
            rot_params,
            ctf_params,
            shift_params,
            iterations,
            self.config,
        )

    def print_statements(self):
        """Print statements about the data."""
        print(f"The size of the dataset is {self.config.datasetSize}")
        L = self.config.sidelen
        print(f"Size of the volume is {L}x{L}x{L}")
        print(
            f"Size of each projection is {self.config.sidelen}x{self.config.sidelen}\n"
        )
        print(f"Output directory is '{self.config.output_path}'\n")
