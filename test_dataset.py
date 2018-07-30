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
input_dataset = 'data/converted_3d.tar'
uhor_scaling = 9.0
uz_scaling = 2.5
turbulence_scaling = 5.0
plot_sample_num = 0
dataset_rounds = 1
use_turbulence = True
#-----------------------------------------------------

db = utils.MyDataset(input_dataset, turbulence_label = use_turbulence, scaling_uhor = uhor_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)

dbloader = torch.utils.data.DataLoader(db, batch_size=32,
                                          shuffle=True, num_workers=4)

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

print(input.size())
print(label.size())

utils.plot_sample(input, label)
