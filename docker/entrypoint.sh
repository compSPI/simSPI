#!/bin/bash -l

export PATH="/work/TEM-simulator/src":$PATH
echo $PATH

source /opt/anaconda/etc/profile.d/conda.sh
conda activate base
which python

exec "$@"