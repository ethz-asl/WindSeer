#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.models as models
import nn_wind_prediction.utils as utils
import numpy as np
import time
import torch
import os
import gc

# ----  Default Params --------------------------------------------------------------
model_name = 'test_model'
model_version = 'latest'
# -----------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Script to plot a prediction of the network')
parser.add_argument('-model_name', dest='model_name', default=model_name, help='The model name')
parser.add_argument('-model_version', dest='model_version', default=model_version, help='The model version')
parser.add_argument('-g', dest='n_grid', type=int, default=64, help='The number of cells in the grid in each dimension')
parser.add_argument('-n', dest='n_iter', type=int, default=100, help='The number of forward passes for the benchmarking')
parser.add_argument('--mp', dest='mixedprecision', action='store_true', help='Mixed precision inference')
parser.add_argument('--no_gpu', dest='no_gpu', action='store_true', help='Mixed precision inference')

args = parser.parse_args()

# check if gpu available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.no_gpu:
    device = torch.device("cpu")

# load the model parameter
params = utils.EDNNParameters('trained_models/' + args.model_name + '/params.yaml')

# load the model and its learnt parameters
NetworkType = getattr(models, params.model['model_type'])
net = NetworkType(**params.model_kwargs())

# load state dict
state_dict = torch.load('trained_models/' + args.model_name + '/' + args.model_version + '.model',
                                                                            map_location=lambda storage, loc: storage)

# load params
net.load_state_dict(state_dict)
net.to(device)
net.eval()

# get input
input = torch.randn(4,args.n_grid,args.n_grid,args.n_grid).to(device)

def start_timer():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    return time.time()

with torch.no_grad():
    with torch.cuda.amp.autocast(args.mixedprecision):
        inference_times = []
        for i in range(args.n_iter):
            torch.cuda.synchronize()
            start_time = start_timer()
            output = net(input.unsqueeze(0))
            torch.cuda.synchronize()
            end_time = time.time()
            inference_times.append(end_time - start_time)
            print('Current inference time: ', end_time - start_time)

print('-------------------')
print('Inference time: ', np.mean(inference_times), '+-', np.std(inference_times))
