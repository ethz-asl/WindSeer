from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import nn_wind_prediction.utils as utils

# enter here the different models to compare
model_names = ['noise01', 'noise02', 'noise03', 'noise04', 'noise05', 'noise11', 'noise12', 'noise13', 'noise14', 'noise15']
# model_names = ['noise01', 'noise02_nn', 'noise03_nn', 'noise04_nn', 'noise05_nn', 'noise11_nn', 'noise12_nn', 'noise13_nn', 'noise14_nn', 'noise15_nn']
# model_names = ['trajectory01', 'trajectory02', 'trajectory03', 'trajectory04', 'trajectory05', 'trajectory06']
# model_names = ['trajectory01_short', 'trajectory02_short', 'trajectory03_short', 'trajectory04_short', 'trajectory05_short', 'trajectory06_short']
# model_names = ['trajectory01_long', 'trajectory02_long', 'trajectory03_long', 'trajectory04_long', 'trajectory05_long', 'trajectory06_long']
# model_names = ['noise01', 'depth01', 'depth02']
# model_names = ['noise01', 'smoothing01']
# model_names = ['noise01', 'vae01', 'vae03', 'noise03', 'vae02', 'vae04']
# model_names = ['noise01', 'noise06', 'noise07', 'noise08', 'uz01']
# model_names = ['trajectory01', 'trajectory02', 'trajectory03', 'trajectory01_short', 'trajectory02_short', 'trajectory03_short', 'trajectory01_long', 'trajectory02_long', 'trajectory03_long']
# model_names = ['noise01', 'pooling01', 'pooling02']

# model_names = ['final01', 'final02', 'final03', 'final04']
# model_names = ['final01_e1500', 'final02_e1500', 'final03_e1500', 'final04_e1500']
# model_names = ['final01_e1500', 'final02_e1500', 'final03_e1500', 'final04_e1500', 'final01', 'final02', 'final03', 'final04']
# model_names = ['noise01', 'pooling01', 'pooling02', 'uz01', 'trajectory03_short']

# model_names = ['final01_440', 'final02_440', 'final03_440', 'final04_440']
# model_names = ['final01_nn_440', 'final02_nn_440', 'final03_nn_440', 'final04_nn_440']

# model_names = ['final01_610', 'final02_610', 'final03_610', 'final04_610']
# model_names = ['final01_nn_610', 'final02_nn_610', 'final03_nn_610', 'final04_nn_610']

# model_names = ['final01_nn_1027', 'final02_nn_1027', 'final03_nn_1027', 'final04_nn_1027']

# model_names = ['final01_nn_3244', 'final02_nn_3244', 'final03_nn_3244', 'final04_nn_3244']

# model_names = ['final01_3784', 'final02_3784', 'final03_3784', 'final04_3784', 'final01_nn_3784', 'final02_nn_3784', 'final03_nn_3784', 'final04_nn_3784']
# model_names = ['final01_nn_3784', 'final02_nn_3784', 'final03_nn_3784', 'final04_nn_3784']


# model_names = ['final01_nn_1027', 'final02_nn_1027', 'final03_nn_1027', 'final04_nn_1027', 'final01_nn_610', 'final02_nn_610', 'final03_nn_610', 'final04_nn_610', 'final01_nn_440', 'final02_nn_440', 'final03_nn_440', 'final04_nn_440']


# default values --------------------------------------------------------
plot_losses = False

# full : full domain
# low  : low altitude
# high : high altitude
plot_domain = 'full'


# mean  : mean
# median: median
# max   : max
plot_property = 'mean'

# tot : total velocity
# hor : horizontal velocity
# ver : vertical velocity
# turb: turbulence
plot_channel = 'tot'

plot_absolute = True
plot_relative = True


show_errors = True

autorange = True

boxplot = True
fig_size = (8.2, 2.5)
# end default values ----------------------------------------------------

# parse the arguments
parser = argparse.ArgumentParser(description='Script to plot prediction errors')
parser.add_argument('-loss', dest='loss', action='store_true', help='Plot the losses')
parser.add_argument('-abs', dest='absolute', action='store_true', help='Plot the absolute errors')
parser.add_argument('-rel', dest='relative', action='store_true', help='Plot the relative errors')

parser.add_argument('-d', dest='domain', default=plot_domain, help='Select the plotting domain: full, low, high')
parser.add_argument('-p', dest='property', default=plot_property, help='Select the plotting property: mean, max, median')
parser.add_argument('-c', dest='channel', default=plot_channel, help='Select the plotting channel: tot, hor, ver, turb')

args = parser.parse_args()

plot_losses = plot_losses or args.loss
plot_absolute = plot_absolute or args.absolute
plot_relative = plot_relative or args.relative

plot_domain = args.domain
plot_property = args.property
plot_channel = args.channel

# extract the data
labels = []

losses = {}

prediction_errors = {}

for name in model_names:
    data = np.load('prediction_errors_' + name + '.npz', allow_pickle=True)
    loss = data['losses'].item()
    prediction_error = data['prediction_errors'].item()

    labels.append(name)

    for key in loss.keys():
        if not key in losses.keys():
            losses.update({key: []})

        losses[key].append(loss[key])

    for key in prediction_error.keys():
        if not key in prediction_errors.keys():
            prediction_errors.update({key: []})

        prediction_errors[key].append(prediction_error[key])

# plot the losses
if plot_losses:
    for key in losses.keys():
        if autorange:
            val_max = 0.0
            for data in losses[key]:
                val_max = np.max([data.max(), val_max])

            ylim = [0, 0.1 * val_max]
        else:
            ylim = None

        utils.violin_plot(labels, losses[key], 'models', key, ylim)

if show_errors:
    for channel in ['tot', 'hor', 'ver', 'turb']:
        key = 'all_' + channel + '_' + plot_property + '_rel'
        
        for lbl, vals in zip(labels, prediction_errors[key]):
            print(channel, lbl, '| mean: ', np.mean(vals), '| median: ', np.median(vals), '| std: ', np.std(vals), '| 25percentile: ', np.percentile(vals, 25), '| 75percentile: ', np.percentile(vals, 75))


# plot the velocity errors
for key in prediction_errors.keys():
    splitted = key.split('_')

    # generate the ylabel and decide if it should be plotted
    method = splitted[2]
    plot = True
    if splitted[2] == 'mean':
        method = 'average'
        if not (plot_property=='mean'):
            plot = False

    elif splitted[2] == 'median':
        method = 'median'
        if not (plot_property=='median'):
            plot = False

    elif splitted[2] == 'max':
        method = 'maximum'
        if not (plot_property=='max'):
            plot = False

    channel = splitted[1]
    unit = ''
    if splitted[1] == 'tot':
        channel = 'total velocity'
        unit = '[m/s]'
        if not (plot_channel=='tot'):
            plot = False

    elif splitted[1] == 'hor':
        channel = 'horizontal velocity'
        unit = '[m/s]'
        if not (plot_channel=='hor'):
            plot = False

    elif splitted[1] == 'ver':
        channel = 'vertical velocity'
        unit = '[m/s]'
        if not (plot_channel=='ver'):
            plot = False

    elif splitted[1] == 'turb':
        channel = 'turbulence'
        unit = '[J/kg]'
        if not (plot_channel=='turb'):
            plot = False

    domain = splitted[0]
    if splitted[0] == 'all':
        domain = 'over the full flow domain'
        if not (plot_domain=='full'):
            plot = False

    elif splitted[0] == 'low':
        domain = 'close to the terrain'
        if not (plot_domain=='low'):
            plot = False

    elif splitted[0] == 'high':
        domain = 'high above the terrain'
        if not (plot_domain=='high'):
            plot = False

    type = 'Absolute'
    try:
        if splitted[3] == 'rel':
            type = 'Relative'
            if not plot_relative:
                plot = False

        if not plot_absolute:
            plot = False
    except:
        if not plot_absolute:
            plot = False

    # get the ylim
    if autorange:
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
        utils.violin_plot(labels, prediction_errors[key], 'models', ylabel, ylim)

        if boxplot:
            # overwrite the labels
#             labels = ['BL', 'AP', 'MP', 'NUZ', 'LT']
            labels = ['BL', 'G10%', 'G30%', 'G50%', 'G80%', 'B10%', 'B30%', 'B50%', 'B10% + G10%', 'B10% + G30%']
#             labels = ['AD4', 'AD6', 'ZD4', 'ZD6', 'AD4', 'AD6', 'ZD4', 'ZD6', 'AD4', 'AD6', 'ZD4', 'ZD6']

            font = {'fontname':'Myriad Pro'}
            fig, ax = plt.subplots(figsize = fig_size)
            bp = ax.boxplot(prediction_errors[key], notch=0, vert=1, whis=1.5, flierprops={'marker': 's', 'markeredgecolor':'none', 'markerfacecolor':'black', 'markersize': 3, 'alpha': 0.05})
            ax.set(axisbelow=True)
            
            ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            
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

                ax.plot(np.average(med.get_xdata()), np.average(prediction_errors[key][i]),
                        color='w', marker='*', markeredgecolor='k')
                ax.set_xticklabels(labels, **font)
#                 ax.set_xticklabels(labels, rotation='vertical', **font)
                if True:
                    ax.set_ylim(bottom=0, top=1.0)
                else:
                    plt.yscale("log")

plt.show()
