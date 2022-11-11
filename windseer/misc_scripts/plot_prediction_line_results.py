import argparse
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from itertools import compress
import os

parser = argparse.ArgumentParser(
    description='Create the measurements line plots from the WindSeer publication'
    )
parser.add_argument(
    '-d',
    dest='data_directory',
    required=True,
    help='Path to the directory with the saved profiles'
    )
parser.add_argument('-l', dest='line', required=True, help='Name of the line to plot')
parser.add_argument(
    '-e', dest='experiment', required=True, help='Name of the experiment to plot'
    )
parser.add_argument(
    '-m',
    dest='methods',
    nargs='+',
    type=str,
    required=True,
    help='Names of the models to plot'
    )
parser.add_argument(
    '-labels',
    dest='labels',
    nargs='+',
    type=str,
    required=True,
    help='Labels for each model'
    )
parser.add_argument(
    '-t',
    dest='towers',
    nargs='+',
    type=str,
    required=True,
    help='Names of input towers used/available'
    )
args = parser.parse_args()

n_sigma = 1.0

if 'bolund' in args.experiment:
    print_tke = True
    fig_size = (8.4, 1.9)

    # bolund specific plot settings
    if args.line == 'lineB_5m':
        line_height = 5
        measurements = {
            'M3': {
                'dist': 3.2,
                's': 10.887975699999998,
                'w': -0.31394859999999997,
                'tke': 2.8701171162999994,
                'sig_s': 1.70,
                'sig_w': 1.11
                },
            'M6': {
                'dist': -46.1,
                's': 14.0541009,
                'w': 0.09037629999999999,
                'tke': 2.068293283,
                'sig_s': 1.36,
                'sig_w': 0.77
                },
            'M7': {
                'dist': -66.9,
                's': 6.95325329,
                'w': 1.2864201,
                'tke': 1.642888709,
                'sig_s': 1.32,
                'sig_w': 0.61
                },
            'M8': {
                'dist': 92.0,
                's': 4.8134877,
                'w': -0.8830332,
                'tke': 5.3725694211,
                'sig_s': 2.35,
                'sig_w': 1.52
                },
            }

    tower_height = 23
    terrain_limits = [0, 30]
    x_lims = [-100, 150]

elif 'askervein' in args.experiment:
    # askervein specific plot settings
    print_tke = True
    fig_size = (8.4, 1.9)

    line_height = 10
    if args.line == 'lineA_10m':
        if 'TU03A' in args.experiment:
            measurements = {
                'ASW85': {
                    'dist': -841,
                    's': 8.6,
                    'w': 0.55,
                    'tke': 1.26,
                    'sig_s': 0.94,
                    'sig_w': 0.49
                    },
                'ASW60': {
                    'dist': -608,
                    's': 8.4,
                    'w': 0.18,
                    'tke': 1.47,
                    'sig_s': 1.01,
                    'sig_w': 0.45
                    },
                'ASW50': {
                    'dist': -492,
                    's': 7.4,
                    'w': 0.37,
                    'tke': 1.42,
                    'sig_s': 0.76,
                    'sig_w': 0.52
                    },
                'ASW35': {
                    'dist': -327,
                    's': 7.9,
                    'w': 1.53,
                    'tke': 1.85,
                    'sig_s': 1.10,
                    'sig_w': 0.64
                    },
                'ASW20': {
                    'dist': -186,
                    's': 11.5,
                    'w': 3.15,
                    'tke': 1.75,
                    'sig_s': 1.18,
                    'sig_w': 0.62
                    },
                'ASW10': {
                    'dist': -96,
                    's': 14.7,
                    'w': 3.78,
                    'tke': 1.81,
                    'sig_s': 1.29,
                    'sig_w': 0.67
                    },
                'HT': {
                    'dist': 0,
                    's': 17.9,
                    'w': 0.91,
                    'tke': 1.44,
                    'sig_s': 1.11,
                    'sig_w': 0.62
                    },
                'ANE10': {
                    'dist': 100,
                    's': 13.3,
                    'w': -2.46,
                    'tke': 2.38,
                    'sig_s': 1.25,
                    'sig_w': 0.58
                    },
                'ANE20': {
                    'dist': 198,
                    's': 5.9,
                    'w': -1.27,
                    'tke': 6.07,
                    'sig_s': 1.84,
                    'sig_w': 0.98
                    },
                'ANE40': {
                    'dist': 393,
                    's': 3.1,
                    'w': -0.39,
                    'tke': 5.64,
                    'sig_s': 2.09,
                    'sig_w': 1.34
                    },
                }

    tower_height = 170
    terrain_limits = [0, 250]
    x_lims = [-900, 450]

elif 'perdigao' in args.experiment:
    # perdigao specific plot settings
    print_tke = False
    fig_size = (8.4, 1.9)

    if args.line == 'lineTSE_30m':
        line_height = 30
        x_lims = [-1680, 810]
        if '20170608' in args.experiment:
            measurements = {
                'TSE01': {
                    'dist': -1495,
                    's': 4.41,
                    'w': 0.86,
                    'u': 4.01,
                    'v': 1.33,
                    'sig_u': 1.52,
                    'sig_v': 1.30,
                    'sig_w': 0.75
                    },
                'TSE02': {
                    'dist': -1140,
                    's': 4.52,
                    'w': 1.76,
                    'u': 3.92,
                    'v': 1.73,
                    'sig_u': 1.20,
                    'sig_v': 1.60,
                    'sig_w': 0.76
                    },
                'TSE04': {
                    'dist': -960,
                    's': 7.89,
                    'w': 2.12,
                    'u': 7.34,
                    'v': 2.48,
                    'sig_u': 1.17,
                    'sig_v': 1.43,
                    'sig_w': 0.76
                    },
                'TSE06': {
                    'dist': -630,
                    's': 3.17,
                    'w': 0.13,
                    'u': 0.44,
                    'v': -2.50,
                    'sig_u': 1.64,
                    'sig_v': 1.64,
                    'sig_w': 1.48
                    },
                'TSE07*': {
                    'dist': -480,
                    's': 2.02,
                    'w': 0.01,
                    'u': 0.35,
                    'v': -0.96,
                    'sig_u': 1.36,
                    'sig_v': 1.59,
                    'sig_w': 0.96
                    },
                'TSE08*': {
                    'dist': -265,
                    's': 2.62,
                    'w': -0.24,
                    'u': 1.59,
                    'v': -0.47,
                    'sig_u': 1.76,
                    'sig_v': 1.80,
                    'sig_w': 1.03
                    },
                'TSE09': {
                    'dist': 0,
                    's': 3.87,
                    'w': 0.09,
                    'u': 2.70,
                    'v': -1.61,
                    'sig_u': 1.70,
                    'sig_v': 2.43,
                    'sig_w': 1.03
                    },
                'TSE10': {
                    'dist': 145,
                    's': 3.64,
                    'w': 0.37,
                    'u': 2.92,
                    'v': -1.01,
                    'sig_u': 1.69,
                    'sig_v': 1.98,
                    'sig_w': 0.82
                    },
                'TSE11': {
                    'dist': 220,
                    's': 4.02,
                    'w': 0.64,
                    'u': 3.44,
                    'v': -0.72,
                    'sig_u': 1.78,
                    'sig_v': 1.95,
                    'sig_w': 0.85
                    },
                'TSE12*': {
                    'dist': 355,
                    's': 4.58,
                    'w': 1.51,
                    'u': 4.26,
                    'v': -0.34,
                    'sig_u': 1.80,
                    'sig_v': 1.63,
                    'sig_w': 0.82
                    },
                'TSE13': {
                    'dist': 465,
                    's': 6.93,
                    'w': 0.87,
                    'u': 6.72,
                    'v': 0.06,
                    'sig_u': 1.70,
                    'sig_v': 1.54,
                    'sig_w': 0.92
                    },
                }
        elif '20170520' in args.experiment:
            measurements = {
                'TSE01': {
                    'dist': -1495,
                    's': 2.32,
                    'w': 0.25,
                    'u': 0.42,
                    'v': 1.23,
                    'sig_u': 1.41,
                    'sig_v': 1.76,
                    'sig_w': 1.05
                    },
                'TSE02': {
                    'dist': -1140,
                    's': 1.73,
                    'w': 0.49,
                    'u': 0.82,
                    'v': 1.12,
                    'sig_u': 0.98,
                    'sig_v': 0.89,
                    'sig_w': 0.75
                    },
                'TSE04': {
                    'dist': -960,
                    's': 6.53,
                    'w': 0.33,
                    'u': -5.57,
                    'v': -3.04,
                    'sig_u': 1.54,
                    'sig_v': 1.36,
                    'sig_w': 0.83
                    },
                'TSE06': {
                    'dist': -630,
                    's': 4.46,
                    'w': 0.69,
                    'u': -3.90,
                    'v': -1.41,
                    'sig_u': 1.67,
                    'sig_v': 1.47,
                    'sig_w': 0.75
                    },
                'TSE07*': {
                    'dist': -480,
                    's': 2.27,
                    'w': 0.17,
                    'u': -1.66,
                    'v': -0.89,
                    'sig_u': 1.11,
                    'sig_v': 1.36,
                    'sig_w': 0.82
                    },
                'TSE08*': {
                    'dist': -265,
                    's': 3.45,
                    'w': 0.24,
                    'u': -2.78,
                    'v': -0.56,
                    'sig_u': 2.09,
                    'sig_v': 1.46,
                    'sig_w': 0.79
                    },
                'TSE09': {
                    'dist': 0,
                    's': 3.55,
                    'w': -0.10,
                    'u': -1.36,
                    'v': 2.30,
                    'sig_u': 1.97,
                    'sig_v': 2.05,
                    'sig_w': 1.28
                    },
                'TSE10': {
                    'dist': 145,
                    's': 3.14,
                    'w': 0.55,
                    'u': -0.37,
                    'v': 2.60,
                    'sig_u': 1.55,
                    'sig_v': 1.71,
                    'sig_w': 0.98
                    },
                'TSE11': {
                    'dist': 220,
                    's': 2.71,
                    'w': 0.53,
                    'u': -0.02,
                    'v': 2.28,
                    'sig_u': 1.23,
                    'sig_v': 1.66,
                    'sig_w': 1.03
                    },
                'TSE12*': {
                    'dist': 355,
                    's': 1.58,
                    'w': 0.17,
                    'u': -0.04,
                    'v': 0.91,
                    'sig_u': 1.05,
                    'sig_v': 1.18,
                    'sig_w': 0.90
                    },
                'TSE13': {
                    'dist': 465,
                    's': 6.54,
                    'w': 0.87,
                    'u': -6.01,
                    'v': -2.12,
                    'sig_u': 1.26,
                    'sig_v': 1.83,
                    'sig_w': 0.81
                    },
                }

    elif args.line == 'lineTNW_20m':
        x_lims = [-1450, 1100]
        line_height = 20
        if '20170608' in args.experiment:
            measurements = {
                'TNW01': {
                    'dist': -1220,
                    's': 3.98,
                    'w': 0.68,
                    'u': 3.49,
                    'v': 1.41,
                    'sig_u': 1.44,
                    'sig_v': 1.22,
                    'sig_w': 0.70
                    },
                'TNW02': {
                    'dist': -990,
                    's': 4.30,
                    'w': 1.64,
                    'u': 3.88,
                    'v': 1.33,
                    'sig_u': 1.41,
                    'sig_v': 1.19,
                    'sig_w': 0.67
                    },
                'TNW03*': {
                    'dist': -830,
                    's': 5.52,
                    'w': 1.60,
                    'u': 4.73,
                    'v': 2.52,
                    'sig_u': 1.63,
                    'sig_v': 1.26,
                    'sig_w': 0.68
                    },
                'TNW05': {
                    'dist': -495,
                    's': 4.87,
                    'w': 1.10,
                    'u': -0.09,
                    'v': -4.64,
                    'sig_u': 1.41,
                    'sig_v': 1.76,
                    'sig_w': 1.04
                    },
                'TNW06': {
                    'dist': -200,
                    's': 4.09,
                    'w': -0.70,
                    'u': 3.41,
                    'v': -1.67,
                    'sig_u': 1.60,
                    'sig_v': 1.39,
                    'sig_w': 0.99
                    },
                'TNW07': {
                    'dist': 0,
                    's': 4.06,
                    'w': -0.14,
                    'u': 3.70,
                    'v': -1.16,
                    'sig_u': 1.40,
                    'sig_v': 1.18,
                    'sig_w': 0.80
                    },
                'TNW08': {
                    'dist': 205,
                    's': 5.51,
                    'w': 0.97,
                    'u': 5.34,
                    'v': -0.41,
                    'sig_u': 1.35,
                    'sig_v': 1.27,
                    'sig_w': 0.85
                    },
                'TNW09*': {
                    'dist': 360,
                    's': 4.36,
                    'w': 1.09,
                    'u': 3.57,
                    'v': 2.16,
                    'sig_u': 1.32,
                    'sig_v': 1.38,
                    'sig_w': 0.85
                    },
                'TNW10': {
                    'dist': 455,
                    's': 6.04,
                    'w': 1.61,
                    'u': 5.41,
                    'v': 2.34,
                    'sig_u': 1.24,
                    'sig_v': 1.37,
                    'sig_w': 0.82
                    },
                'TNW11': {
                    'dist': 570,
                    's': 7.05,
                    'w': 0.41,
                    'u': 6.21,
                    'v': 2.99,
                    'sig_u': 1.32,
                    'sig_v': 1.54,
                    'sig_w': 0.83
                    },
                'TNW12': {
                    'dist': 625,
                    's': 2.25,
                    'w': -0.03,
                    'u': 1.72,
                    'v': 0.10,
                    'sig_u': 1.46,
                    'sig_v': 1.38,
                    'sig_w': 0.98
                    },
                'TNW13': {
                    'dist': 695,
                    's': 2.20,
                    'w': 0.36,
                    'u': 0.20,
                    'v': -1.86,
                    'sig_u': 1.01,
                    'sig_v': 1.19,
                    'sig_w': 0.85
                    },
                'TNW14': {
                    'dist': 785,
                    's': 2.25,
                    'w': 0.21,
                    'u': 0.31,
                    'v': -1.79,
                    'sig_u': 1.13,
                    'sig_v': 1.45,
                    'sig_w': 1.05
                    },
                'TNW15': {
                    'dist': 840,
                    's': 2.50,
                    'w': 0.32,
                    'u': 0.23,
                    'v': -1.96,
                    'sig_u': 1.34,
                    'sig_v': 1.52,
                    'sig_w': 0.89
                    },
                }
        elif '20170520' in args.experiment:
            measurements = {
                'TNW01': {
                    'dist': -1220,
                    's': 2.14,
                    'w': 0.24,
                    'u': 0.22,
                    'v': 1.42,
                    'sig_u': 1.47,
                    'sig_v': 1.36,
                    'sig_w': 0.83
                    },
                'TNW02': {
                    'dist': -990,
                    's': 1.53,
                    'w': 0.57,
                    'u': 0.85,
                    'v': 0.65,
                    'sig_u': 1.04,
                    'sig_v': 0.91,
                    'sig_w': 0.65
                    },
                'TNW03*': {
                    'dist': -830,
                    's': 5.74,
                    'w': 0.58,
                    'u': -4.73,
                    'v': -2.85,
                    'sig_u': 1.45,
                    'sig_v': 1.53,
                    'sig_w': 0.62
                    },
                'TNW05': {
                    'dist': -495,
                    's': 2.73,
                    'w': 0.00,
                    'u': -2.08,
                    'v': -0.05,
                    'sig_u': 1.47,
                    'sig_v': 1.54,
                    'sig_w': 0.81
                    },
                'TNW06': {
                    'dist': -200,
                    's': 3.13,
                    'w': -0.24,
                    'u': -1.69,
                    'v': 1.76,
                    'sig_u': 1.97,
                    'sig_v': 1.50,
                    'sig_w': 0.87
                    },
                'TNW07': {
                    'dist': 0,
                    's': 3.78,
                    'w': -0.06,
                    'u': -0.60,
                    'v': 3.28,
                    'sig_u': 1.79,
                    'sig_v': 1.53,
                    'sig_w': 0.96
                    },
                'TNW08': {
                    'dist': 205,
                    's': 2.88,
                    'w': 0.94,
                    'u': 0.37,
                    'v': 2.43,
                    'sig_u': 1.44,
                    'sig_v': 1.32,
                    'sig_w': 0.85
                    },
                'TNW09*': {
                    'dist': 360,
                    's': 1.97,
                    'w': 0.43,
                    'u': 0.34,
                    'v': 1.41,
                    'sig_u': 1.16,
                    'sig_v': 1.15,
                    'sig_w': 0.66
                    },
                'TNW10': {
                    'dist': 455,
                    's': 1.92,
                    'w': -0.01,
                    'u': -0.66,
                    'v': 0.66,
                    'sig_u': 1.53,
                    'sig_v': 1.34,
                    'sig_w': 1.05
                    },
                'TNW11': {
                    'dist': 570,
                    's': 5.89,
                    'w': 1.18,
                    'u': -5.44,
                    'v': -1.67,
                    'sig_u': 1.38,
                    'sig_v': 1.42,
                    'sig_w': 0.84
                    },
                'TNW12': {
                    'dist': 625,
                    's': 5.09,
                    'w': 1.68,
                    'u': -4.77,
                    'v': -1.15,
                    'sig_u': 1.32,
                    'sig_v': 1.33,
                    'sig_w': 0.77
                    },
                'TNW13': {
                    'dist': 695,
                    's': 2.83,
                    'w': 0.92,
                    'u': -2.46,
                    'v': -0.69,
                    'sig_u': 1.21,
                    'sig_v': 1.43,
                    'sig_w': 0.81
                    },
                'TNW14': {
                    'dist': 785,
                    's': 3.46,
                    'w': 0.92,
                    'u': -3.11,
                    'v': -0.63,
                    'sig_u': 1.38,
                    'sig_v': 1.35,
                    'sig_w': 0.70
                    },
                'TNW15': {
                    'dist': 840,
                    's': 3.60,
                    'w': 0.89,
                    'u': -3.25,
                    'v': -0.54,
                    'sig_u': 1.29,
                    'sig_v': 1.43,
                    'sig_w': 0.72
                    },
                'TNW16': {
                    'dist': 920,
                    's': 3.40,
                    'w': 0.61,
                    'u': -3.11,
                    'v': -0.11,
                    'sig_u': 1.29,
                    'sig_v': 1.41,
                    'sig_w': 0.66
                    },
                }

    tower_height = 330
    terrain_limits = [0, 530]
data = {}
for twr in args.towers:
    for mth, lb in zip(args.methods, args.labels):
        res = np.load(
            os.path.join(
                args.data_directory, args.experiment + '_' + twr + '_' + mth + '.npy'
                ),
            allow_pickle=True
            )
        label = lb + ', input: ' + twr

        if len(res) > 1:
            data[label] = res[1][args.line]

fig, ah = plt.subplots(4, 1, squeeze=False, figsize=fig_size)

styles = ['-', '-', '-', '--', '--', '--', ':', ':', ':']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors_mpl = prop_cycle.by_key()['color']
colors = [
    colors_mpl[0], colors_mpl[1], colors_mpl[2], colors_mpl[0], colors_mpl[1],
    colors_mpl[2], colors_mpl[0], colors_mpl[1], colors_mpl[2]
    ]
handles = []
for i, key in enumerate(data.keys()):
    if print_tke:
        h_s, = ah[0][0].plot(
            data[key]['dist'],
            data[key]['s_pred'],
            styles[i],
            label=key,
            color=colors[i],
            lw=1
            )
        ah[1][0].plot(
            data[key]['dist'],
            data[key]['w_pred'],
            styles[i],
            label=key,
            color=colors[i],
            lw=1
            )
        handles.append(h_s)

        if not 'GPR' in key:
            ah[2][0].plot(
                data[key]['dist'],
                data[key]['tke_pred'],
                styles[i],
                label=key,
                color=colors[i],
                lw=1
                )

    else:
        h_s, = ah[0][0].plot(
            data[key]['dist'],
            data[key]['u_pred'],
            styles[i],
            label=key,
            color=colors[i],
            lw=1
            )
        ah[1][0].plot(
            data[key]['dist'],
            data[key]['v_pred'],
            styles[i],
            label=key,
            color=colors[i],
            lw=1
            )
        ah[2][0].plot(
            data[key]['dist'],
            data[key]['w_pred'],
            styles[i],
            label=key,
            color=colors[i],
            lw=1
            )
        handles.append(h_s)

    if i == 0:
        for twr in measurements.keys():
            index_dist = np.argmin(
                np.abs(data[key]['dist'] - measurements[twr]['dist'])
                )
            terrain_height = data[key]['terrain'][index_dist]

            ah[3][0].annotate(
                twr,
                xy=(measurements[twr]['dist'], 0),
                xytext=(measurements[twr]['dist'], terrain_height + tower_height),
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="-"),
                verticalalignment="top",
                zorder=0
                )

            if print_tke:
                ah[0][0].errorbar(
                    measurements[twr]['dist'],
                    measurements[twr]['s'],
                    yerr=n_sigma * measurements[twr]['sig_s'],
                    fmt='D',
                    color='black',
                    ecolor='black',
                    barsabove=False,
                    markersize=2.5,
                    elinewidth=2,
                    capsize=3,
                    zorder=50
                    )
                ah[1][0].errorbar(
                    measurements[twr]['dist'],
                    measurements[twr]['w'],
                    yerr=n_sigma * measurements[twr]['sig_w'],
                    fmt='D',
                    color='black',
                    ecolor='black',
                    barsabove=False,
                    markersize=2.5,
                    elinewidth=2,
                    capsize=3,
                    zorder=50
                    )
                ah[2][0].errorbar(
                    measurements[twr]['dist'],
                    measurements[twr]['tke'],
                    fmt='D',
                    color='black',
                    ecolor='black',
                    barsabove=False,
                    markersize=4,
                    elinewidth=2,
                    capsize=3,
                    zorder=50
                    )
            else:
                ah[0][0].errorbar(
                    measurements[twr]['dist'],
                    measurements[twr]['u'],
                    yerr=n_sigma * measurements[twr]['sig_u'],
                    fmt='D',
                    color='black',
                    ecolor='black',
                    barsabove=False,
                    markersize=2.5,
                    elinewidth=2,
                    capsize=3,
                    zorder=50
                    )
                ah[1][0].errorbar(
                    measurements[twr]['dist'],
                    measurements[twr]['v'],
                    yerr=n_sigma * measurements[twr]['sig_v'],
                    fmt='D',
                    color='black',
                    ecolor='black',
                    barsabove=False,
                    markersize=2.5,
                    elinewidth=2,
                    capsize=3,
                    zorder=50
                    )
                ah[2][0].errorbar(
                    measurements[twr]['dist'],
                    measurements[twr]['w'],
                    yerr=n_sigma * measurements[twr]['sig_w'],
                    fmt='D',
                    color='black',
                    ecolor='black',
                    barsabove=False,
                    markersize=2.5,
                    elinewidth=2,
                    capsize=3,
                    zorder=50
                    )

        ah[3][0].plot(
            data[key]['dist'],
            data[key]['terrain'] + line_height,
            color='black',
            lw=1.0
            )
        ah[3][0].fill_between(
            data[key]['dist'], data[key]['terrain'], color='lightgrey', linewidth=0.0
            )
        ah[3][0].plot(data[key]['dist'], data[key]['terrain'], color='dimgrey', lw=0.3)

ah[0][0].axes.xaxis.set_visible(False)
ah[1][0].axes.xaxis.set_visible(False)
ah[2][0].axes.xaxis.set_visible(False)
ah[3][0].set_ylim(terrain_limits)
ah[3][0].set_ylabel(r'Terrain $[m]$')
ah[3][0].set_xlabel(r'Distance $[m]$')
if print_tke:
    ah[0][0].set_ylabel(r'$S$ $[m/s]$')
    ah[1][0].set_ylabel(r'$W$ $[m/s]$')
    ah[2][0].set_ylabel(r'$TKE$ $[m^2/s^2]$')
else:
    ah[0][0].set_ylabel(r'$U$ $[m/s]$')
    ah[1][0].set_ylabel(r'$V$ $[m/s]$')
    ah[2][0].set_ylabel(r'$W$ $[m/s]$')

for i in range(4):
    ah[i][0].set_xlim(x_lims)

plt.legend(
    bbox_to_anchor=(1.04, 2.0), loc="center left", borderaxespad=0, handles=handles
    )

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.8, top=0.98)
plt.show()
