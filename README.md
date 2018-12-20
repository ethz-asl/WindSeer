# intel_wind

This repository contains the tools to predict the wind using a neural network.

The `data_generation` folder provides the necessary scripts to generate the training data using OpenFoam while the `wind_prediction` folder contains the network and the scripts to train and predict.

## Installation
This guide explains how to set up the environment in Ubuntu to make the scripts in the repository run.

1. Python3 is required:
   `sudo apt-get install python3.6`
2. Install PyTorch v4.1 or newer:

   `pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl`
   `pip3 install torchvision`

3. Install the following required python packages:
   `pip3 install tensorboardX lz4 numpy tqdm`

4. Install the `nn_wind_prediction` package in developer mode. To do so change into the `intel_wind` directory and execute the following command:
    `pip install -e wind_prediction/`

## Guidelines
### Branches
- master: Current main release, i.e. under development but bench-tested and thus operational.
- features/MYFEATURENAME: A new feature branch. All new features shall be added like this.
- fix/MYFIX: A new fix. All new fixes shall be added like this.

### Other
- Do not include any file larger than 10 MB to a commit. If the data needs to be copied for example from the cluster computer to the local machine use:
    `scp -r USERNAME@login.leonhard.ethz.ch:DIR_TO_COPY* TARGET_DIR`