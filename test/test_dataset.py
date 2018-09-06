#!/usr/bin/env python
'''
Script to test and benchmark the implementation of MyDataset
'''

import sys
import time
import torch
from torch.utils.data import DataLoader
import utils

#------ Params to modidify ---------------------------
compressed = True
input_dataset = 'data/test.tar'
uhor_scaling = 1.0
uz_scaling = 1.0
turbulence_scaling = 1.0
plot_sample_num = 0
dataset_rounds = 0
use_turbulence = True
stride_hor = 1
stride_vert = 1
#-----------------------------------------------------

db = utils.MyDataset(input_dataset, stride_hor = stride_hor, stride_vert = stride_vert, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling, compressed = compressed)

dbloader = torch.utils.data.DataLoader(db, batch_size=32,
                                          shuffle=True, num_workers=2)

start_time = time.time()
for j in range(dataset_rounds):
    for data in dbloader:
        input, label = data
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
utils.plot_sample(input, label)
