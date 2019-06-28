#!/usr/bin/env python
'''
Script to test and benchmark the implementation of MyDataset
'''

import matplotlib.pyplot as plt
import nn_wind_prediction.data as nn_data
import nn_wind_prediction.utils as utils
import numpy as np
import sys
import time
import torch
from torch.utils.data import DataLoader

#------ Params to modidify ---------------------------
compute_dataset_statistics = True
plot_sample_num = 0
compressed = False
input_dataset = '../wind_prediction/data/test.hdf5'
nx = 64
ny = 64
nz = 64
input_mode = 1
subsample = False
augmentation = False
ux_scaling = 1.0
uy_scaling = 1.0
uz_scaling = 1.0
turbulence_scaling = 1.0
p_scaling = 1.0
epsilon_scaling = 1.0
nut_scaling = 1.0
terrain_scaling = 1.0
stride_hor = 1
stride_vert = 1
autoscale = False
aug_mode = 1
input_channels = ['terrain', 'ux', 'uy', 'uz']
label_channels = ['ux', 'uy', 'uz', 'turb']
use_turbulence = 'turb' in label_channels
use_pressure = 'p' in label_channels
use_epsilon = 'epsilon' in label_channels
use_nut = 'nut' in label_channels
plot_divergence = True
#-----------------------------------------------------

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    db = nn_data.HDF5Dataset(input_dataset, input_channels=input_channels, label_channels=label_channels,
                                      nx=nx, ny=ny, nz=nz, input_mode=input_mode, augmentation_mode=aug_mode,
                                      subsample=subsample, augmentation=augmentation, autoscale=autoscale,
                                      stride_hor=stride_hor, stride_vert=stride_vert,
                                      scaling_ux=ux_scaling, scaling_uy=uy_scaling,
                                      scaling_terrain=terrain_scaling,
                                      scaling_uz=uz_scaling, scaling_turb=turbulence_scaling, scaling_p=p_scaling,
                                      scaling_epsilon=epsilon_scaling, scaling_nut=nut_scaling,
                                      return_grid_size=True, verbose=True)

    dbloader = torch.utils.data.DataLoader(db, batch_size=1,
                                              shuffle=False, num_workers=4)

    if compute_dataset_statistics:
        ux = []
        uy = []
        uz = []
        turb = []
        ux_std = []
        uy_std = []
        uz_std = []
        turb_std = []
        ux_max = []
        uy_max = []
        uz_max = []
        turb_max = []
        reflow_ratio = []
        global dataset_rounds
        dataset_rounds = 1
        dx = []
        dy = []
        dz = []
        max_div = []
        mean_div = []
        terrain = []

    start_time = time.time()
    for j in range(dataset_rounds):
        for i, data in enumerate(dbloader):
            if autoscale:
                input, label, scale, ds = data
            else:
                input, label, ds = data

            ds = ds.squeeze()

            if compute_dataset_statistics:
                ux.append(label[:,0,:].abs().mean().item())
                uy.append(label[:,1,:].abs().mean().item())
                uz.append(label[:,2,:].abs().mean().item())
                ux_max.append(label[:,0,:].abs().max().item())
                uy_max.append(label[:,1,:].abs().max().item())
                uz_max.append(label[:,2,:].abs().max().item())
                ux_std.append(label[:,0,:].abs().std().item())
                uy_std.append(label[:,1,:].abs().std().item())
                uz_std.append(label[:,2,:].abs().std().item())

                if use_turbulence:
                    turb.append(label[:,3,:].abs().mean().item())
                    turb_max.append(label[:,3,:].abs().max().item())
                    turb_std.append(label[:,3,:].abs().std().item())

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

                dx.append(ds[0].item())
                dy.append(ds[1].item())
                dz.append(ds[2].item())

                divergence = utils.divergence(label[:3], ds.squeeze()).unsqueeze(1).unsqueeze(0)
                mean_div.append(divergence.abs().mean())
                max_div.append(divergence.abs().max().item())

                idx = input.shape[2] - 1
                while(idx >= 0 and input[0,0,idx,:,:].min() > 0):
                    idx -= 1

                terrain.append(idx * ds[2].item())

    if compute_dataset_statistics:
        print('------------------------------------------------------------------------------')
        print('INFO: Mean ux:   {} m/s'.format(np.mean(ux)))
        print('INFO: Mean uy:   {} m/s'.format(np.mean(uy)))
        print('INFO: Mean uz:   {} m/s'.format(np.mean(uz)))
        print('INFO: Max ux:    {} m/s'.format(np.mean(ux_max)))
        print('INFO: Max uy:    {} m/s'.format(np.mean(uy_max)))
        print('INFO: Max uz:    {} m/s'.format(np.mean(uz_max)))
        print('INFO: Std ux:    {} m/s'.format(np.mean(ux_std)))
        print('INFO: Std uy:    {} m/s'.format(np.mean(uy_std)))
        print('INFO: Std uz:    {} m/s'.format(np.mean(uz_std)))
        if use_turbulence:
            print('INFO: Mean turb: {} J/kg'.format(np.mean(turb)))
            print('INFO: Max turb:  {} J/kg'.format(np.mean(turb_max)))
            print('INFO: Std turb:  {} J/kg'.format(np.mean(turb_std)))
        print('INFO: Number of cases with a reflow ratio of > 0.05: {}'.format(sum(i > 0.05 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 0.10: {}'.format(sum(i > 0.10 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 0.20: {}'.format(sum(i > 0.20 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 0.30: {}'.format(sum(i > 0.30 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 0.40: {}'.format(sum(i > 0.40 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 0.50: {}'.format(sum(i > 0.50 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 0.75: {}'.format(sum(i > 0.75 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 1.00: {}'.format(sum(i > 1.00 for i in reflow_ratio)))
        print('INFO: Number of cases with a reflow ratio of > 1.00: {}'.format(sum(i > 1.00 for i in reflow_ratio)))
        print('INFO: Min dx:   {} m'.format(np.min(dx)))
        print('INFO: Max dx:   {} m'.format(np.max(dx)))
        print('INFO: Min dy:   {} m'.format(np.min(dy)))
        print('INFO: Max dy:   {} m'.format(np.max(dy)))
        print('INFO: Min dz:   {} m'.format(np.min(dz)))
        print('INFO: Max dz:   {} m'.format(np.max(dz)))
        print('INFO: Average divergence: {}'.format(np.mean(mean_div)))
        print('INFO: Maximum divergence: {}'.format(np.max(max_div)))
        print('INFO: Min terrain height: {}'.format(np.min(terrain)))
        print('INFO: Max terrain height: {}'.format(np.max(terrain)))
        print('------------------------------------------------------------------------------')

        dataset_stats = {
            'ux': ux,
            'uy': uy,
            'uz': uz,
            'turb': turb,
            'reflow_ratio': reflow_ratio,
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'max_div': max_div,
            'mean_div': mean_div,
            'terrain': terrain
            }
        np.save('dataset_stats.npy', dataset_stats)

        # plotting of the statistics
        plt.figure()
        plt.subplot(2, 3, 1)
        plt.hist(reflow_ratio, 10, facecolor='r')
        plt.grid(True)
        plt.xlabel('Reflow Ratio []')
        plt.ylabel('N')

        plt.subplot(2, 3, 2)
        plt.hist(dx, 10, facecolor='g')
        plt.grid(True)
        plt.xlabel('dx [m]')
        plt.ylabel('N')

        plt.subplot(2, 3, 3)
        plt.hist(dy, 10, facecolor='b')
        plt.grid(True)
        plt.xlabel('dy [m]')
        plt.ylabel('N')

        plt.subplot(2, 3, 4)
        plt.hist(dz, 10, facecolor='y')
        plt.grid(True)
        plt.xlabel('dz [m]')
        plt.ylabel('N')

        plt.subplot(2, 3, 5)
        plt.hist(terrain, 10, facecolor='y')
        plt.grid(True)
        plt.xlabel('Terrain height [m]')
        plt.ylabel('N')

        plt.figure()
        plt.subplot(2, 3, 1)
        plt.hist(max_div, 10, facecolor='r')
        plt.grid(True)
        plt.xlabel('Maximum divergence')
        plt.ylabel('N')

        plt.subplot(2, 3, 2)
        plt.hist(mean_div, 10, facecolor='g')
        plt.grid(True)
        plt.xlabel('Mean divergence')
        plt.ylabel('N')

        plt.subplot(2, 3, 3)
        plt.hist(turb, 10, facecolor='y')
        plt.grid(True)
        plt.xlabel('Turb. kin. energy [J/kg]')
        plt.ylabel('N')

        plt.subplot(2, 3, 4)
        plt.hist(ux, 10, facecolor='m')
        plt.grid(True)
        plt.xlabel('Ux [m/s]')
        plt.ylabel('N')

        plt.subplot(2, 3, 5)
        plt.hist(uy, 10, facecolor='b')
        plt.grid(True)
        plt.xlabel('Uy [m/s]')
        plt.ylabel('N')

        plt.subplot(2, 3, 6)
        plt.hist(uz, 10, facecolor='k')
        plt.grid(True)
        plt.xlabel('Uz [m/s]')
        plt.ylabel('N')
        plt.draw()

    print('INFO: Time to get all samples in the dataset', dataset_rounds, 'times took', (time.time() - start_time), 'seconds')

    try:
        if autoscale:
            input, label, scale, ds = db[plot_sample_num]
        else:
            input, label, ds = db[plot_sample_num]
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
    provided_channels = ['ux_in', 'uy_in', 'uz_in', 'terrain', 'ux_cfd', 'uy_cfd', 'uz_cfd', 'turb']
    terrain = input[:1]
    input = torch.cat((input[1:4], terrain, label), 0)
    utils.plot_sample(provided_channels, 'all', input, None, terrain.squeeze(), plot_divergence, nn_data.get_grid_size(input_dataset))

if __name__ == '__main__':
#     try:
#         torch.multiprocessing.set_start_method('spawn')
#     except RuntimeError:
#         pass
    main()
