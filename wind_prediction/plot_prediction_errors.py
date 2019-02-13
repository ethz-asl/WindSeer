from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import nn_wind_prediction.utils as utils

# enter here the different models to compare
model_names = ['ednn3d_1', 'ednn3d_2', 'ednn3d_3', 'ednn3d_4', 'ednn3d_5', 'ednn3d_6']

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

plot_absolute = False
plot_relative = False

autorange = True

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
    data = np.load('prediction_errors_' + name + '.npz')
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

plt.show()
