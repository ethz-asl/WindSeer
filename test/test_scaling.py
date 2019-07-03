#!/usr/bin/env python
'''
Script to test scaling of data
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
compressed = False
input_dataset = 'test.hdf5'
input_channels = ['terrain', 'ux', 'uy', 'uz']
label_channels = ['ux', 'uy', 'uz']
#-----------------------------------------------------

def main():
    db = nn_data.HDF5Dataset(input_dataset, input_channels = input_channels, label_channels= label_channels,
                            autoscale = True, input_mode = 1)
    dbloader = torch.utils.data.DataLoader(db, batch_size=1,
                                              shuffle=False, num_workers=0)

    scale_normalized = []
    
    start_time = time.time()
    for i, data in enumerate(dbloader):
        input, label, scale = data

        name = db.get_name(i)
        windspeed = float(name.split("_", 5)[4][1:])

        scale_normalized.append(scale/windspeed)

    # violin plot
    violin_plot('scale normalized', [scale_normalized], 'method', 'scale/windspeed', [0.5, 2.5])
    
    plt.show()

def violin_plot(labels, data, xlabel, ylabel, ylim):
    index = np.arange(len(labels))

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')

    # need to manually set the factor and make sure that it is not too small, otherwise a numerical underflow will happen
    factor = np.power(len(data[0]), -1.0 / (len(data) + 4))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, points=300, bw_method=np.max([factor, 0.6]))

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1 = []
    medians = []
    quartile3 = []
    for channel in data:
        quartile1_channel, medians_channel, quartile3_channel = np.percentile(channel, [25, 50, 75])
        quartile1.append(quartile1_channel)
        medians.append(medians_channel)
        quartile3.append(quartile3_channel)

    whiskers = np.array([adjacent_values(sorted(sorted_array), q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(inds)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)
    fig.tight_layout()


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

if __name__ == '__main__':
#     try:
#         torch.multiprocessing.set_start_method('spawn')
#     except RuntimeError:
#         pass
    main()
