from __future__ import print_function

import windseer.plotting as plotting

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import os

parser = argparse.ArgumentParser(description='Script to plot prediction errors')
parser.add_argument(
    '-data',
    dest='data_directory',
    default='',
    type=str,
    help='Directory where the prediction errors are located'
    )
parser.add_argument(
    '-m', dest='models', nargs='+', required=True, type=str, help='Model names'
    )
parser.add_argument(
    '-l',
    dest='labels',
    nargs='+',
    type=str,
    help='Custom labels for the plots (the same number of entries as models)'
    )
parser.add_argument('-loss', dest='loss', action='store_true', help='Plot the losses')
parser.add_argument(
    '-abs', dest='absolute', action='store_true', help='Plot the absolute errors'
    )
parser.add_argument(
    '-rel', dest='relative', action='store_true', help='Plot the relative errors'
    )
parser.add_argument(
    '-box', dest='boxplot', action='store_true', help='Generate the boxplot figures'
    )
parser.add_argument(
    '-errors',
    dest='show_errors',
    action='store_true',
    help='Plot the error statistics to the console'
    )
parser.add_argument(
    '-losses', dest='plot_losses', action='store_true', help='Plot the loss statistics'
    )
parser.add_argument(
    '-auto',
    dest='autorange',
    action='store_true',
    help='Automatically set the range of the plot to 10 % of the maximum value'
    )
parser.add_argument(
    '-fig',
    dest='fig_size',
    nargs=2,
    default=(8.2, 2.5),
    type=float,
    help='Set the figure size in inches'
    )
parser.add_argument(
    '-d',
    dest='domain',
    default='full',
    type=str,
    help='Select the plotting domain: full, low, high'
    )
parser.add_argument(
    '-p',
    dest='property',
    default='mean',
    type=str,
    help='Select the plotting property: mean, max, median'
    )
parser.add_argument(
    '-c',
    dest='channel',
    default='tot',
    type=str,
    help='Select the plotting channel: tot, hor, ver, turb'
    )

args = parser.parse_args()

# extract the data
labels = []
losses = {}
prediction_errors = {}

for i, name in enumerate(args.models):
    data = np.load(
        os.path.join(args.data_directory, 'prediction_errors_' + name + '.npz'),
        allow_pickle=True
        )
    loss = data['losses'].item()
    prediction_error = data['prediction_errors'].item()

    if args.labels is None:
        labels.append(name)
    else:
        labels.append(args.labels[i])

    for key in loss.keys():
        if not key in losses.keys():
            losses.update({key: []})

        losses[key].append(loss[key])

    for key in prediction_error.keys():
        if not key in prediction_errors.keys():
            prediction_errors.update({key: []})

        prediction_errors[key].append(prediction_error[key])

if args.plot_losses:
    for key in losses.keys():
        if args.autorange:
            val_max = 0.0
            for data in losses[key]:
                val_max = np.max([data.max(), val_max])

            ylim = [0, 0.1 * val_max]
        else:
            ylim = None

        plotting.violin_plot(labels, losses[key], 'models', key, ylim)

if args.show_errors:
    for channel in ['tot', 'hor', 'ver', 'turb']:
        key = 'all_' + channel + '_' + args.property + '_rel'

        for lbl, vals in zip(labels, prediction_errors[key]):
            print(
                channel, lbl, '| mean: ', np.mean(vals), '| median: ',
                np.median(vals), '| std: ', np.std(vals), '| 25percentile: ',
                np.percentile(vals, 25), '| 75percentile: ', np.percentile(vals, 75)
                )

for key in prediction_errors.keys():
    splitted = key.split('_')

    method = splitted[2]
    plot = True
    if splitted[2] == 'mean':
        method = 'average'
        if not (args.property == 'mean'):
            plot = False

    elif splitted[2] == 'median':
        method = 'median'
        if not (args.property == 'median'):
            plot = False

    elif splitted[2] == 'max':
        method = 'maximum'
        if not (args.property == 'max'):
            plot = False

    channel = splitted[1]
    unit = ''
    if splitted[1] == 'tot':
        channel = 'total velocity'
        unit = '[m/s]'
        if not (args.channel == 'tot'):
            plot = False

    elif splitted[1] == 'hor':
        channel = 'horizontal velocity'
        unit = '[m/s]'
        if not (args.channel == 'hor'):
            plot = False

    elif splitted[1] == 'ver':
        channel = 'vertical velocity'
        unit = '[m/s]'
        if not (args.channel == 'ver'):
            plot = False

    elif splitted[1] == 'turb':
        channel = 'turbulence'
        unit = '[J/kg]'
        if not (args.channel == 'turb'):
            plot = False

    domain = splitted[0]
    if splitted[0] == 'all':
        domain = 'over the full flow domain'
        if not (args.domain == 'full'):
            plot = False

    elif splitted[0] == 'low':
        domain = 'close to the terrain'
        if not (args.domain == 'low'):
            plot = False

    elif splitted[0] == 'high':
        domain = 'high above the terrain'
        if not (args.domain == 'high'):
            plot = False

    type = 'Absolute'
    try:
        if splitted[3] == 'rel':
            type = 'Relative'
            if not args.relative:
                plot = False

        elif not args.absolute:
            plot = False
    except:
        if not args.absolute:
            plot = False

    # get the ylim
    if args.autorange:
        val_max = 0.0
        for data in prediction_errors[key]:
            val_max = np.max([data.max(), val_max])

        if splitted[2] == 'mean':
            ylim = [0, 0.25 * val_max]

        elif splitted[2] == 'max':
            ylim = [0, val_max]

        elif splitted[2] == 'median':
            ylim = [0, 0.5 * val_max]

        else:
            ylim = None

    else:
        ylim = None

    if plot:
        ylabel = type + ' ' + method + ' ' + channel + ' error ' + domain + ' ' + unit
        plotting.violin_plot(labels, prediction_errors[key], 'models', ylabel, ylim)

        if args.boxplot:
            font = {'fontname': 'Myriad Pro'}
            fig, ax = plt.subplots(figsize=args.fig_size)
            bp = ax.boxplot(
                prediction_errors[key],
                notch=0,
                vert=1,
                whis=1.5,
                flierprops={
                    'marker': 's',
                    'markeredgecolor': 'none',
                    'markerfacecolor': 'black',
                    'markersize': 3,
                    'alpha': 0.05
                    }
                )
            ax.set(axisbelow=True)

            ax.yaxis.grid(
                True, linestyle='-', which='major', color='lightgrey', alpha=0.5
                )

            plt.ylabel('Relative Prediction Error [-]', **font)

            num_boxes = len(prediction_errors[key])
            for i in range(num_boxes):
                box = bp['boxes'][i]
                box_x = []
                box_y = []
                for j in range(5):
                    box_x.append(box.get_xdata()[j])
                    box_y.append(box.get_ydata()[j])
                    box_coords = np.column_stack([box_x, box_y])
                    ax.add_patch(Polygon(box_coords, facecolor='blue'))

                med = bp['medians'][i]
                median_x = []
                median_y = []
                for j in range(2):
                    median_x.append(med.get_xdata()[j])
                    median_y.append(med.get_ydata()[j])
                    ax.plot(median_x, median_y, 'k')

                ax.plot(
                    np.average(med.get_xdata()),
                    np.average(prediction_errors[key][i]),
                    color='w',
                    marker='*',
                    markeredgecolor='k'
                    )
                ax.set_xticklabels(labels, **font)
                if True:
                    ax.set_ylim(bottom=0, top=1.0)
                else:
                    plt.yscale("log")

plt.show()
