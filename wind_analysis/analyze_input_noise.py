import argparse
from matplotlib import pyplot as plt
import numpy as np
import random
import scipy
import torch

from analysis_utils import utils
import nn_wind_prediction.utils as nn_utils

parser = argparse.ArgumentParser(description='Optimize wind speed and direction from COSMO data using observations')
parser.add_argument('config_yaml', help='Input yaml config')
parser.add_argument('-p', '--plot', action='store_true', help='Plot the optimization results')
parser.add_argument('-s', '--seed', default=0, type=int, help='Seed of the random generators')
parser.add_argument('-r', '--rate', default=50, type=int, help='Sampling rate of the wind measurements')
args = parser.parse_args()

if args.seed > 0:
    random.seed(args.seed)
    np.random.seed(args.seed)

config = nn_utils.BasicParameters(args.config_yaml)
#nn_params = nn_utils.EDNNParameters(args.model_dir + '/params.yaml')

config.params['model'] = {}
config.params['model']['input_channels'] = ['terrain', 'ux', 'uy', 'uz']
config.params['model']['label_channels'] = ['ux', 'uy', 'uz']
config.params['model']['autoscale'] = False

measurement, _, _, mask, _, wind_data, _ = utils.load_measurements(config.params['measurements'], config.params['model'])

data = []
label = []

if wind_data is not None:
    # compute the fourier transforms of the wind data
    fig, ax = plt.subplots(3, 1)
    ax[0].plot(wind_data['time'] * 1e-6, wind_data['we'])
    ax[0].set_ylabel('we [m/s]')
    ax[0].set_xlabel('time [s]')

    ax[1].plot(wind_data['time'] * 1e-6, wind_data['wn'])
    ax[1].set_ylabel('wn [m/s]')
    ax[1].set_xlabel('time [s]')

    ax[2].plot(wind_data['time'] * 1e-6, wind_data['wd'])
    ax[2].set_ylabel('wd [m/s]')
    ax[2].set_xlabel('time [s]')

    fig.suptitle('Raw Wind Estimates')

    freq_fft = scipy.fft.rfftfreq(len(wind_data['we']), 1 / args.rate)
    we_fft = scipy.fft.rfft(wind_data['we'])
    wn_fft = scipy.fft.rfft(wind_data['wn'])
    wd_fft = scipy.fft.rfft(wind_data['wd'])

    fig, ax = plt.subplots(3, 1)
    ax[0].plot(freq_fft[1:], np.abs(we_fft[1:]))
    ax[0].set_ylabel('amplitude we')
    ax[0].set_xlabel('frequency [Hz]')
    ax[0].set_yscale('log')

    ax[1].plot(freq_fft[1:], np.abs(wn_fft[1:]))
    ax[1].set_ylabel('amplitude wn')
    ax[1].set_xlabel('frequency [Hz]')
    ax[1].set_yscale('log')

    ax[2].plot(freq_fft[1:], np.abs(wd_fft[1:]))
    ax[2].set_ylabel('amplitude wd')
    ax[2].set_xlabel('frequency [Hz]')
    ax[2].set_yscale('log')
    
    data += [wind_data['wn'], wind_data['we'], wind_data['wd']]
    label += ['wn raw', 'we raw', 'wd raw']

    print('Raw we:')
    print('\tmean: ', wind_data['we'].mean())
    print('\tstd:  ', wind_data['we'].std())
    print('Raw wn:')
    print('\tmean: ', wind_data['wn'].mean())
    print('\tstd:  ', wind_data['wn'].std())
    print('Raw wd:')
    print('\tmean: ', wind_data['wd'].mean())
    print('\tstd:  ', wind_data['wd'].std())



meas_z = torch.masked_select(measurement[0, 2], mask[0] > 0)
meas_y = torch.masked_select(measurement[0, 1], mask[0] > 0)
meas_x = torch.masked_select(measurement[0, 0], mask[0] > 0)

print('Binned ux:')
print('\tmean: ', meas_x.mean().item())
print('\tstd:  ', meas_x.std().item())
print('Binned uy:')
print('\tmean: ', meas_y.mean().item())
print('\tstd:  ', meas_y.std().item())
print('Binned uz:')
print('\tmean: ', meas_z.mean().item())
print('\tstd:  ', meas_z.std().item())

data += [meas_x.numpy(), meas_y.numpy(), meas_z.numpy()]
label += ['ux binned', 'uy binned', 'uz binned'] 
nn_utils.violin_plot(label, data, None, 'vel [m/s]', ylim=None)

plt.show()


