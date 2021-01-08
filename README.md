# intel_wind

This repository contains the tools to predict the wind using a neural network.

## Structure
### Benchmark Planner
This folder contains the planning benchmark tools used to compare the planning performance using different wind predictions.

### Data Generation
This folder contains the pipeline to generate the training data to learn the wind prediction

### Test
This folder contains some test functions for the python scripts.

### Wind Prediction
This folder contains the tools to train and evaluate the networks for the wind prediction.

## Installation
This guide explains how to set up the environment in Ubuntu to make the scripts in the repository run.

1. Python3 (at least version 3.6) is required, usually it is already present. If not install it with:
   `sudo apt-get install python3.6`

2. Install (Cuda)[https://developer.nvidia.com/cuda-zone] if a graphics card is present and (PyTorch)[https://pytorch.org/get-started/locally/] according to your CUDA, Ubuntu, and Python version. At least PyTorch 1.0.1 is required.

3. Install the following required python packages:
   `pip3 install tensorboardX lz4 numpy tqdm matplotlib scipy pandas h5py interpolation termcolor pyyaml scikit-learn`
   
4. Install python tkinter
   `sudo apt-get install python3-tk`

5. Clone the repo and update the submodules with:
   `git clone --recurse-submodules https://github.com/ethz-asl/intel_wind.git`

5. Install the `nn_wind_prediction` package in developer mode. To do so change into the `intel_wind` directory and execute the following command:
   `pip3 install -e wind_prediction/`

6. Install RAdam and Ranger if you want to use the Ranger optimizer. To do so change into respective folders and install the packages using:
   `pip3 install . --user --no-deps`

TODO: add installation for the planning benchmark locally and on the cluster

### llmvlite build errors
If you have trouble installing `llvmlite` on Ubuntu 18.04 (a dependency of `interpolate`), it may be due to a version mismatch. Try this:
   ```
   sudo apt-get install llvm-10-dev
   export LLVM_CONFIG='/usr/bin/llvm-config-10'
   pip3 install interpolation
   ```   


## Working with Leonhard

On the leonhard cluster, you need to perform a few setup steps

1. Load python 3.6.4 for gpu and h5py python package:
   `module load python_gpu/3.6.4 hdf5/1.10.1`
   
2. Install required packages:
   `python -m pip install --user tensorboardX==1.4 lz4 tqdm interpolation`
   
3. Setup to use PyTorch 1.0.1:
   ~~~
   bsub -Is -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" bash
   module load magma/2.2.0 libffi/3.2.1 python_gpu/3.6.4 eth_proxy
   pip install --user torch==1.0.1
   python -c 'import torch; print(torch.__version__); print("cuda avail: {0}".format(torch.cuda.is_available()))'
   ~~~
   (Packages will be installed in `$HOME/.local/lib64/python3.6/site-packages`)


## Guidelines
### Branches
- master: Current main release, i.e. under development but bench-tested and thus operational.
- features/MYFEATURENAME: A new feature branch. All new features shall be added like this.
- fix/MYFIX: A new fix. All new fixes shall be added like this.

### Other
- Do not include any file larger than 10 MB to a commit. If the data needs to be copied for example from the cluster computer to the local machine use:
    `scp -r USERNAME@login.leonhard.ethz.ch:DIR_TO_COPY* TARGET_DIR`
