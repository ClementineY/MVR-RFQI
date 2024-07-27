#!/bin/bash

# Load MATLAB module
module load matlab/full/r2023a

# Create and activate Conda environment
conda env create -f my_environment.yml --verbose
conda activate rfqi_w_matlab

# Install matlabengine
pip install matlabengine==9.14.3

# Set environment variables
export LD_PRELOAD=/shared/ucl/apps/Matlab/R2023a/full/bin/glnxa64/glibc-2.17_shim.so
export LD_LIBRARY_PATH=/shared/ucl/apps/gcc/9.2.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/lustre/home/ucabjy6/raptor/raptor/plasma-control-ucl/MVR-RFQI:$PYTHONPATH


echo "Environment setup complete. MATLAB module loaded and environment created."
