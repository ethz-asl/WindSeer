#!/usr/bin/env python

from __future__ import print_function

import argparse
import windseer.nn.models as models
import windseer.utils as utils
import numpy as np
import time
import torch
import os
import gc

parser = argparse.ArgumentParser(
    description='Script to plot a prediction of the network'
    )
parser.add_argument(
    '-model', dest='model_dir', required=True, help='The model directory'
    )
parser.add_argument(
    '-model_version', dest='model_version', default='latest', help='The model version'
    )
parser.add_argument(
    '-g',
    dest='n_grid',
    type=int,
    default=64,
    help='The number of cells in the grid in each dimension'
    )
parser.add_argument(
    '-n',
    dest='n_iter',
    type=int,
    default=100,
    help='The number of forward passes for the benchmarking'
    )
parser.add_argument(
    '--mp',
    dest='mixedprecision',
    action='store_true',
    help='Mixed precision inference'
    )
parser.add_argument(
    '--no_gpu', dest='no_gpu', action='store_true', help='Mixed precision inference'
    )

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.no_gpu:
    device = torch.device("cpu")

net, params = utils.load_model(
    args.model_dir, args.model_version, None, device, eval=True
    )

# get input
input = torch.randn(4, args.n_grid, args.n_grid, args.n_grid).to(device).unsqueeze(0)

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
            output = net(input)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_times.append(end_time - start_time)
            print('Current inference time: ', end_time - start_time)

print('-------------------')
print('Inference time: ', np.mean(inference_times), '+-', np.std(inference_times))
