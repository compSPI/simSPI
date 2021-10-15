[![Build](https://github.com/compSPI/simSPI/actions/workflows/build.yml/badge.svg)](https://github.com/compSPI/simSPI/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/compSPI/ioSPI/branch/master/graph/badge.svg?token=OBVOV3ZM1O)](https://codecov.io/gh/compSPI/simSPI)
[![DeepSource](https://deepsource.io/gh/compSPI/simSPI.svg/?label=active+issues&show_trend=true&token=9eFu6aig3-oXQIuhdDoYTEq-)](https://deepsource.io/gh/compSPI/simSPI/?ref=repository-badge)

# simSPI

Methods and tools for simulating SPI data.

# Download

First create a conda environment with the required dependencies using the `enviroment.yml` file as follows:

    conda env create --file environment.yml

Then download:

    git clone https://github.com/compSPI/ioSPI.git

# Contribute

We strongly recommend installing our pre-commit hook, to ensure that your code
follows our Python style guidelines. To do so, just run the following command line at the root of this repository:

    pre-commit install

With this hook, your code will be automatically formatted with our style conventions. If the pre-commit hooks black and isort mark "Failed" upon commit, this means that your code was not correctly formatted, but has been re-formatted by the hooks. Re-commit the result and check that the pre-commit marks "Passed".

Note that the hooks do not reformat docstrings automatically. If hooks failed with an error related to the documentation, you should correct this error yourself, commit the result and check that the pre-commit marks "Passed".

See our [contributing](https://github.com/compspi/compspi/blob/master/docs/contributing.rst) guidelines!