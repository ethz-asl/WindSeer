#!/usr/bin/env python
'''
Script to test and benchmark the implementation of FullDataset
'''

import nn_wind_prediction.data as nn_data
import time
import torch
from torch.utils.data import DataLoader

#------ Params to modidify ---------------------------
compressed = False
tensor_dataset = '../wind_prediction/data/MD_bigger_test.tar'
hdf5_dataset = '../wind_prediction/data/bigger_test.hdf5'
nx = 64
ny = 64
nz = 64
input_mode = 0
subsample = False
augmentation = False
ux_scaling = 1.0
uy_scaling = 1.0
uz_scaling = 1.0
turbulence_scaling = 1.0
p_scaling = 1.0
epsilon_scaling = 1.0
nut_scaling = 1.0
terrain_scaling = 64.0
stride_hor = 1
stride_vert = 1
autoscale = False
dataset_rounds = 5
input_channels = ['terrain', 'ux', 'uy', 'uz']
label_channels = ['ux', 'uy', 'uz', 'turb']
use_turbulence = 'turb' in label_channels
use_pressure = 'p' in label_channels
use_epsilon = 'epsilon' in label_channels
use_nut = 'nut' in label_channels
aug_mode = 1
#-----------------------------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    TensorDataset = nn_data.MyDataset(tensor_dataset, nx = nx, ny = ny, nz = nz, input_mode = input_mode,
                           subsample = subsample, augmentation = augmentation, autoscale = autoscale,
                           stride_hor = stride_hor, stride_vert = stride_vert, augmentation_mode=aug_mode,
                           turbulence_label = use_turbulence, pressure_label = use_pressure, epsilon_label = use_epsilon,
                           nut_label = use_nut, scaling_ux = ux_scaling, scaling_uy = uy_scaling, scaling_terrain = terrain_scaling,
                           scaling_uz = uz_scaling, scaling_turb = turbulence_scaling, scaling_p = p_scaling,
                           scaling_epsilon = epsilon_scaling, scaling_nut = nut_scaling, compressed = compressed,
                        return_grid_size = True, verbose=True)

    TDloader = torch.utils.data.DataLoader(TensorDataset, batch_size=1,
                                              shuffle=False, num_workers=4)

    start_time = time.time()
    for j in range(dataset_rounds):
        for i, data in enumerate(TDloader):
            if autoscale:
                input, label, scale, ds = data
            else:
                input, label, ds = data

    print('Input shape: ', input.shape)
    print('Label shape: ', label.shape)

    print('INFO: Time to get all samples in the tensor dataset', dataset_rounds, 'times took', (time.time() - start_time),
          'seconds')

    HDF5Dataset = nn_data.HDF5Dataset(hdf5_dataset, input_channels = input_channels, label_channels = label_channels,
                                        nx=nx, ny=ny, nz=nz, input_mode=input_mode, augmentation_mode=aug_mode,
                                        subsample=subsample, augmentation=augmentation, autoscale=autoscale,
                                        stride_hor=stride_hor, stride_vert=stride_vert,
                                        nut_label=use_nut, scaling_ux = ux_scaling, scaling_uy=uy_scaling,
                                        scaling_terrain=terrain_scaling,
                                        scaling_uz=uz_scaling, scaling_turb=turbulence_scaling, scaling_p=p_scaling,
                                        scaling_epsilon=epsilon_scaling, scaling_nut=nut_scaling,
                                        return_grid_size=True,verbose=True)

    HDloader = torch.utils.data.DataLoader(HDF5Dataset, batch_size=1,
                                           shuffle=False, num_workers=4)

    start_time = time.time()
    for j in range(dataset_rounds):
        for i, data in enumerate(HDloader):
            if autoscale:
                input, label, scale, ds = data
            else:
                input, label, ds = data
    print('Input shape: ', input.shape)
    print('Label shape: ', label.shape)

    print('INFO: Time to get all samples in the hdf5 dataset', dataset_rounds, 'times took', (time.time() - start_time),
          'seconds')

if __name__ == '__main__':
    main()
