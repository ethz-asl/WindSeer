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

1. Python3 is required:
   `sudo apt-get install python3.6`

2. Install PyTorch v4.1 or newer:

   `pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp35-cp35m-linux_x86_64.whl`
   `pip3 install torchvision`

3. Install the following required python packages:
   `pip3 install tensorboardX lz4 numpy tqdm`

4. Install the `nn_wind_prediction` package in developer mode. To do so change into the `intel_wind` directory and execute the following command:
    `pip install -e wind_prediction/`

TODO: add installation for the planning benchmark locally and on the cluster

## Working with Leonhard

On the leonhard cluster, you need to perform a few setup steps

1. Load python 3.6.4 for gpu:
   `module load python_gpu/3.6.4`
   
2. Install required packages:
   `python -m pip install --user tensorboardX==1.4 lz4 tqdm`
   
3. Setup to use PyTorch 0.4.1:
   ~~~
   bsub -Is -W 4:00 -R "rusage[mem=4096, ngpus_excl_p=1]" bash
   module load magma/2.2.0 libffi/3.2.1 python_gpu/3.6.4 eth_proxy
   pip install --user torch==0.4.1
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
