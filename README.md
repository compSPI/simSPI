[![Build](https://github.com/compSPI/simSPI/actions/workflows/build.yml/badge.svg)](https://github.com/compSPI/simSPI/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/compSPI/ioSPI/branch/master/graph/badge.svg?token=OBVOV3ZM1O)](https://codecov.io/gh/compSPI/simSPI)
[![DeepSource](https://deepsource.io/gh/compSPI/simSPI.svg/?label=active+issues&show_trend=true&token=9eFu6aig3-oXQIuhdDoYTEq-)](https://deepsource.io/gh/compSPI/simSPI/?ref=repository-badge)

# simSPI

Methods and tools for simulating SPI data.

# Download

First create a conda environment with the required dependencies using the `enviroment.yml` file as follows:

    conda env create --file environment.yml
or

    conda env create -f environment.yml


If the conda enivronment already exists but new packages have been added to environment.yml then update it using

    conda env update --prefix ./env --file environment.yml  --prune
Activate the environment using

    conda activate simSPI

Then download:

    git clone https://github.com/compSPI/ioSPI.git



# Linear Simulator
There are two types of simulator in this package:
Linear and TEM.

The linear simulator uses the linear model of the Cryo-EM physics to generate data.
This involves fist rotating the biomolecule and obtaining its 2D tomographic projection.
This 2D projection is then modulated with a CTF (Contrast Transfer Function) in fourier domain (convolution with corresponding PSF in image domain) and then shifted using scpecified shift values.

=====================================

###### Input:

Config file: containing conditions/parameters/specifications to run the simulator

MRC file: containing the 3D pixel-domain volume/biomolecule.
If the mrc file is not given then by default simulator initializes the shape of the biomolecule as a cube.

###### Output:

Dataset saved in ".mrcs" format.

The metadata is saved in a ".star" file

Copy of input .cfg file is also saved in .txt and .cfg format.

=====================================

**Run**

To create dataset using linear simulator move to base directory and run

    python -m simSPI.linear_simulator.main "path_to_config_file"


The run will create _Datasets_ directory in _simSPI/linear_simulator_ if it doesn't already exist
and will save the data in a new folder inside it. The name of the folder can be specified in the config file.

There are two main mode of generating the dataset:


_Starfile-based_

In this mode the parameters of the linear simulator for each projection are obtained from a specified star file whose path is given in the config file.

_Distributional_

In this mode the parameters of the linear simulator for each projection are obtained using specified distribution.
To choose this mode, leave the input star file path in config file blank.


**Example Scripts**

Example scripts are given in _example_scripts_ folder

To run them move to base folder of the directory and run, for example,

    ./example_scripts/generate_betagal_data_distributional.sh

This runs

    simSPI/linear_simulator/dataset_generator.py
using

    simSPI/linear_simulator/configs/betagal_distribution.cfg
file.

To modify the parameters of this dataset change the .cfg file or create a new one.

# TEM Simulator

...
# Contribute

We strongly recommend installing our pre-commit hook, to ensure that your code
follows our Python style guidelines. To do so, just run the following command line at the root of this repository:

    pre-commit install

With this hook, your code will be automatically formatted with our style conventions. If the pre-commit hooks black and isort mark "Failed" upon commit, this means that your code was not correctly formatted, but has been re-formatted by the hooks. Re-commit the result and check that the pre-commit marks "Passed".

Note that the hooks do not reformat docstrings automatically. If hooks failed with an error related to the documentation, you should correct this error yourself, commit the result and check that the pre-commit marks "Passed".


See our [contributing](https://github.com/compspi/compspi/blob/master/docs/contributing.rst) guidelines!
