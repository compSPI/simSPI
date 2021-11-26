# imports wp simulator and generates data scpecified by the .cfg file
import os

import mrcfile
import numpy as np
import pandas as pd
import starfile
import torch
from pytorch3d.transforms import matrix_to_euler_angles
from simulation_module import simulator
from simulation_utils import ParamsFactory, return_names


class dataset_gen(torch.nn.Module):
    def __init__(self, config):
        super(dataset_gen, self).__init__()

        self.config = config
        self.simulator = simulator(config)
        self.dataframe = []
        self.variable_names = return_names()
        self.params_generator = ParamsFactory.get_params_generator(config)

    def forward(self):
        # iterating to save the data over a loop

        for iterations in range(
            self.config.datasetSize // self.config.chunks
        ):  # TODO: last few images will be left
            rot_params, ctf_params, shift_params = self.params_generator.get_params()
            projections = self.simulator(rot_params, ctf_params, shift_params)
            projections = projections.detach().cpu().numpy()
            self.save_mrc(projections, iterations)
            self.starfile_data(rot_params, ctf_params, shift_params, iterations)
        self.save_starfile()
        self.save_configfile()

    def save_mrc(self, projections, iterations):
        image_path = os.path.join(
            self.config.output_path, str(iterations).zfill(4) + ".mrcs"
        )
        with mrcfile.new(image_path, overwrite="True") as m:
            m.set_data(projections.astype("float32"))

    def save_starfile(self):
        df = pd.DataFrame(
            data=self.dataframe,
            index=[idx for idx in range(len(self.dataframe))],
            columns=self.variable_names,
        )
        starfile.write(
            df, os.path.join(self.config.output_path, "Simulated.star"), overwrite=True
        )

    def save_configfile(self):
        self.config.output_path = os.path.join(os.getcwd(), self.config.output_path)
        if os.path.exists(self.config.output_path) == False:
            os.mkdir(self.config.output_path)
        with open(os.path.join(self.config.output_path, "config.cfg"), "w") as fp:
            self.config.config.write(fp)
        with open(os.path.join(self.config.output_path, "config.txt"), "w") as fp:
            self.config.config.write(fp)

    def starfile_data(self, rot_params, ctf_params, shift_params, iterations):
        euler = np.degrees(rot_params["euler"].numpy())
        ImageName = [
            str(idx).zfill(3) + "@" + str(iterations).zfill(4) + ".mrcs"
            for idx in range(self.config.chunks)
        ]
        for num in range(self.config.chunks):
            self.dataframe.append(
                [
                    ImageName[num],
                    euler[num, 0],
                    euler[num, 1],
                    euler[num, 2],
                    shift_params["shiftX"][num].item(),
                    shift_params["shiftY"][num].item(),
                    ctf_params["defocusU"][num].item(),
                    ctf_params["defocusV"][num].item(),
                    ctf_params["defocusAngle"][num].item(),
                    self.config.kV,
                    self.config.pixel_size,
                    self.config.cs,
                    self.config.amplitude_contrast,
                    self.config.bfactor,
                ]
            )
