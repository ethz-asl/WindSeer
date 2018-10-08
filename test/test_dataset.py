#!/usr/bin/env python
'''
Script to test and benchmark the implementation of MyDataset
'''

import nn_wind_prediction.data as data
import nn_wind_prediction.utils as utils
import numpy as np
import sys
import time
import torch
from torch.utils.data import DataLoader

#------ Params to modidify ---------------------------
compressed = False
input_dataset = 'test.tar'
uhor_scaling = 5.5
uz_scaling = 0.28
turbulence_scaling = 1.05
plot_sample_num = 0
dataset_rounds = 0
use_turbulence = True
stride_hor = 1
stride_vert = 1
compute_dataset_statistics = True
#-----------------------------------------------------

db = data.MyDataset(input_dataset, stride_hor = stride_hor, stride_vert = stride_vert, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling, compressed = compressed)

dbloader = torch.utils.data.DataLoader(db, batch_size=1,
                                          shuffle=True, num_workers=0)

if compute_dataset_statistics:
    ux = []
    uy = []
    uz = []
    turb = []
    reflow_ratio = []
    dataset_rounds = 1

start_time = time.time()
for j in range(dataset_rounds):
    for i, data in enumerate(dbloader):
        input, label = data

        if compute_dataset_statistics:
            ux.append(label[:,0,:].abs().mean().item())
            uy.append(label[:,1,:].abs().mean().item())
            uz.append(label[:,2,:].abs().mean().item())
            if use_turbulence:
                turb.append(label[:,3,:].abs().mean().item())

            # compute if a reflow is happening in the simulated flow
            if label[:,0,:].mean().abs().item() > label[:,1,:].mean().abs().item():
                # general flow in x-direction
                max_vel = label[:,0,:].max().item()
                min_vel = label[:,0,:].min().item()
            else:
                # general flow in y-direction
                max_vel = label[:,1,:].max().item()
                min_vel = label[:,1,:].min().item()

            max_v = max(abs(max_vel), abs(min_vel))
            min_v = min(abs(max_vel), abs(min_vel))

            if (max_vel * min_vel < 0):
                reflow_ratio.append(min_v / max_v)
            else:
                reflow_ratio.append(0.0)

if compute_dataset_statistics:
    print('INFO: Mean ux:   {}'.format(np.mean(ux)))
    print('INFO: Mean uy:   {}'.format(np.mean(uy)))
    print('INFO: Mean uz:   {}'.format(np.mean(uz)))
    if use_turbulence:
        print('INFO: Mean turb: {}'.format(np.mean(turb)))
    print('INFO: Number of cases with a reflow ratio of > 0.05: {}'.format(sum(i > 0.05 for i in reflow_ratio)))
    print('INFO: Number of cases with a reflow ratio of > 0.10: {}'.format(sum(i > 0.10 for i in reflow_ratio)))
    print('INFO: Number of cases with a reflow ratio of > 0.20: {}'.format(sum(i > 0.20 for i in reflow_ratio)))
    print('INFO: Number of cases with a reflow ratio of > 0.30: {}'.format(sum(i > 0.30 for i in reflow_ratio)))
    print('INFO: Number of cases with a reflow ratio of > 0.40: {}'.format(sum(i > 0.40 for i in reflow_ratio)))
    print('INFO: Number of cases with a reflow ratio of > 0.50: {}'.format(sum(i > 0.50 for i in reflow_ratio)))
    print('INFO: Number of cases with a reflow ratio of > 0.75: {}'.format(sum(i > 0.75 for i in reflow_ratio)))
    print('INFO: Number of cases with a reflow ratio of > 1.00: {}'.format(sum(i > 1.00 for i in reflow_ratio)))

print('INFO: Time to get all samples in the dataset', dataset_rounds, 'times took', (time.time() - start_time), 'seconds')

try:
    input, label = db[plot_sample_num]
except:
    print('The plot_sample_num needs to be a value between 0 and', len(db)-1, '->' , plot_sample_num, ' is invalid.')
    sys.exit()

print(' ')
print('----------------------------------')
print('Input size:')
print(input.size())
print('Output size:')
print(label.size())
print('----------------------------------')
print(' ')

# plot the sample
utils.plot_sample(input, label, input[0,:])
