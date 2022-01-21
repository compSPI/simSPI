#!/usr/bin/env bash

export PATH="/work/TEM-simulator/src":$PATH

source /opt/anaconda/etc/profile.d/conda.sh
conda activate base

exec "$@"