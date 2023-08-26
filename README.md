# WindSeer

This repository contains the tools to predict the wind using a neural network.

## Structure
### Benchmark Planner
The planning benchmark tools used to compare the planning performance using different wind predictions.

### Data Generation
The pipeline to generate the training data to learn the wind prediction.

### Libs
Submodules with alternative optimizers.

### WindSeer
The tools to train and evaluate the networks for the wind prediction.

## Installation
This guide explains how to set up the environment in Ubuntu to make the scripts in the repository run.

1. Python3 (at least version 3.6) is required, usually it is already present. If not install it with:
   `sudo apt-get install python3.6`

2. Install (Cuda)[https://developer.nvidia.com/cuda-zone] if a graphics card is present and (PyTorch)[https://pytorch.org/get-started/locally/] according to your CUDA, Ubuntu, and Python version. At least PyTorch 1.9.0 is required.
   Earlier Pytorch Version have some Cudnn issues with Cuda 11.1.

3. Install the following required python packages:
   `pip3 install scipy sklearn pyproj h5py matplotlib GDAL netCDF4 tqdm pandas tensorboard pyulog interpolation mayavi mpl-scatter-density PyQt5 yapf`

4. Clone the repo and update the submodules with:
   `git clone --recurse-submodules https://github.com/ethz-asl/intel_wind.git`

5. Install the `windseer` package in developer mode. To do so change into the `intel_wind` directory and execute the following command:
   `pip3 install -e .`

6. Install the LV03 projection to pyproj:
   `python3 windseer/proj_definitions/install_ch_defs.py`

7. Install RAdam and Ranger if you want to use the Ranger optimizer. To do so change into respective folders and install the packages using:
   `pip3 install . --user --no-deps`

### llmvlite build errors
If you have trouble installing `llvmlite` on Ubuntu 18.04 (a dependency of `interpolate`), it may be due to a version mismatch. Try this:
   ```
   sudo apt-get install llvm-10-dev
   export LLVM_CONFIG='/usr/bin/llvm-config-10'
   pip3 install interpolation
   ```
