from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from windseer.plotting import plot_mpl_scatter_density, density_scatter


parser = argparse.ArgumentParser(description='Script to plot measurement campaigns benchmark results')
parser.add_argument(
    '-f',
    dest='filenames',
    nargs='+',
    type=str,
    help='Path to the individual files with the data'
    )
parser.add_argument(
    '-fig',
    dest='fig_size',
    nargs=2,
    default=(8.0,7.0),
    type=float,
    help='Set the figure size in inches'
    )
parser.add_argument(
    '-dpi',
    dest='dpi',
    default=15.0,
    type=float,
    help='Set the figure density plot in dpi'
    )
parser.add_argument(
    '-mpl',
    dest='mpl_scatter_density',
    action='store_true',
    help='Use the mpl_scatter_density function'
    )


args = parser.parse_args()

# load the data
all_data = {}
for i, name in enumerate(args.filenames):
    data = np.load(name, allow_pickle=True).item()

    for key in data.keys():
        if not key in all_data:
            all_data[key] = []

        all_data[key] += data[key]


all_data['Shor_pred'] = np.sqrt(np.array(all_data['u_pred'])**2 + np.array(all_data['v_pred'])**2).tolist()
all_data['Shor_meas'] = np.sqrt(np.array(all_data['u_meas'])**2 + np.array(all_data['v_meas'])**2).tolist()
all_data['S_pred'] = np.sqrt(np.array(all_data['u_pred'])**2 + np.array(all_data['v_pred'])**2 + np.array(all_data['w_pred'])**2).tolist()
all_data['S_meas'] = np.sqrt(np.array(all_data['u_meas'])**2 + np.array(all_data['v_meas'])**2 + np.array(all_data['w_meas'])**2).tolist()

# generate the plots
channels = ['u', 'v', 'w', 'Shor', 'S']
labels = ['u [m/s]', 'v [m/s]', 'w [m/s]', 'S horizontal [m/s]', 'S [m/s]']

if 'tke_meas' in all_data.keys():
    channels += ['tke']
    labels += ['TKE [m2/s2]']

for ch, lbl in zip(channels, labels):
    mask = np.isfinite(all_data[ch + '_meas'])
    data_meas = np.array(all_data[ch + '_meas'])[mask]
    data_pred = np.array(all_data[ch + '_pred'])[mask]
    fig, ax = plt.subplots(figsize=args.fig_size)
    ax.set_aspect('equal', 'box')
    max_val = max(max(data_meas), max(data_pred))
    min_val = min(min(data_meas), min(data_pred))
    if args.mpl_scatter_density:
        plot_mpl_scatter_density(fig, data_meas, data_pred, args.dpi)
    else:
        density_scatter(data_meas, data_pred, ax, bins=75, s=0.1)
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel('measurement ' + lbl)
    plt.ylabel('prediction ' + lbl)

    # compute metrics
    bias = np.mean(data_pred - data_meas)
    rmse = np.sqrt(np.mean((data_pred - data_meas)**2))
    rho, p = scipy.stats.pearsonr(data_meas, data_pred)
    print('max: ', np.max(data_meas))
    print(ch)
    print('\tBIAS:', bias)
    print('\tRMSE:', rmse)
    print('\tR:', rho, ' (p:', p, ')')

plt.show()