#!/usr/bin/env python
'''
Script to test and benchmark the implementation of MyDataset
'''

import matplotlib.pyplot as plt
import sys
import time
import torch
from torch.utils.data import DataLoader
import utils


#------ Params to modidify ---------------------------
input_dataset = 'data/converted_train.tar'
ux_scaling = 9.0
uz_scaling = 2.5
turbulence_scaling = 5.0
plot_sample_num = 0
dataset_rounds = 30
#-----------------------------------------------------


db = utils.MyDataset(input_dataset,  scaling_ux = ux_scaling, scaling_uz = uz_scaling, scaling_nut = turbulence_scaling)

dbloader = torch.utils.data.DataLoader(db, batch_size=32,
                                          shuffle=True, num_workers=4)

start_time = time.time()
for j in range(dataset_rounds):
    for data in dbloader:
        input, label = data
print('INFO: Time to get all samples in the dataset', dataset_rounds, 'times took', (time.time() - start_time), 'seconds')

input, label = db[plot_sample_num]

try:
    input, label = db[plot_sample_num]
except:
    print('The plot_sample_num needs to be a value between 0 and', len(db)-1, '->' , plot_sample_num, ' is invalid.')
    sys.exit()

fh_in, ah_in = plt.subplots(3, 2)
fh_in.set_size_inches([6.2, 10.2])

h_ux_in = ah_in[0][0].imshow(input[1,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
h_uz_in = ah_in[0][1].imshow(input[2,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
ah_in[0][0].set_title('Ux in')
ah_in[0][1].set_title('Uz in')
fh_in.colorbar(h_ux_in, ax=ah_in[0][0])
fh_in.colorbar(h_uz_in, ax=ah_in[0][1])

h_ux_in = ah_in[1][0].imshow(label[0,:,:], origin='lower', vmin=label[0,:,:].min(), vmax=label[0,:,:].max())
h_uz_in = ah_in[1][1].imshow(label[1,:,:], origin='lower', vmin=label[1,:,:].min(), vmax=label[1,:,:].max())
ah_in[1][0].set_title('Ux label')
ah_in[1][1].set_title('Uz label')
fh_in.colorbar(h_ux_in, ax=ah_in[1][0])
fh_in.colorbar(h_uz_in, ax=ah_in[1][1])

h_ux_in = ah_in[2][0].imshow(input[0,:,:], origin='lower', vmin=input[0,:,:].min(), vmax=input[0,:,:].max())
h_uz_in = ah_in[2][1].imshow(label[2,:,:], origin='lower', vmin=label[2,:,:].min(), vmax=label[2,:,:].max())
ah_in[2][0].set_title('Terrain')
ah_in[2][1].set_title('Turbulence viscosity label')
fh_in.colorbar(h_ux_in, ax=ah_in[2][0])
fh_in.colorbar(h_uz_in, ax=ah_in[2][1])

plt.show()
