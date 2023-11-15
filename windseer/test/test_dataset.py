#!/usr/bin/env python
'''
Testcases for the HDF5Dataset
'''

from windseer.data import HDF5Dataset
import windseer.plotting as plotting
import windseer

import argparse
import copy
import numpy as np
import os
import random
import torch
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_filename = os.path.join(testdata_folder, 'test_dataset.hdf5')

default_config = {
    'filename': test_filename,
    'input_channels': ['terrain', 'ux', 'uy', 'uz'],
    'label_channels': ['ux', 'uy', 'uz', 'turb'],
    'nx': 64,
    'ny': 64,
    'nz': 64,
    'input_mode': 5,
    'augmentation_mode': 1,
    'augmentation': True,
    'autoscale': False,
    'augmentation_kwargs': {
        'subsampling': True,
        'rotating': True,
        },
    'stride_hor': 1,
    'stride_vert': 1,
    'device': 'cpu',
    'scaling_ux': 1.0,
    'scaling_uy': 1.0,
    'scaling_uz': 1.0,
    'scaling_turb': 1.0,
    'scaling_terrain': 1.0,
    'turbulence_scaling': 1.0,
    'p_scaling': 1.0,
    'epsilon_scaling': 1.0,
    'nut_scaling': 1.0,
    'terrain_scaling': 1.0,
    'return_name': False,
    'return_grid_size': False,
    'verbose': True,
    'loss_weighting_fn': 0,
    'loss_weighting_clamp': True,
    'input_smoothing': True,
    'input_smoothing_interpolation': False,
    'input_smoothing_interpolation_linear': False,
    'additive_gaussian_noise': True,
    'max_gaussian_noise_std': 0.0,
    'n_turb_fields': 1,
    'max_normalized_turb_scale': 0.0,
    'max_normalized_bias_scale': 0.0,
    'only_z_velocity_bias': True,
    'max_fraction_of_sparse_data': 0.05,
    'min_fraction_of_sparse_data': 0.001,
    'trajectory_min_length': 30,
    'trajectory_max_length': 300,
    'trajectory_min_segment_length': 3.0,
    'trajectory_max_segment_length': 20.0,
    'trajectory_step_size': 1.0,
    'trajectory_max_iter': 50,
    'trajectory_start_weighting_mode': 0,
    'trajectory_length_short_focus': True,
    'use_system_random': False,
    }


def plot_sample(input_mode):
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    config = copy.deepcopy(default_config)
    config['input_mode'] = input_mode

    dataset = HDF5Dataset(**config)
    data = dataset[0]
    input = data[0]
    label = data[1]

    # plot the sample
    input_channels_plotting = [
        s + '_in' if s in ['ux', 'uy', 'uz'] else s
        for s in dataset.get_input_channels()
        ]
    label_channels_plotting = [s + '_cfd' for s in config['label_channels']]
    if 'mask' in dataset.get_input_channels():
        input_mask = input[dataset.get_input_channels().index('mask')].squeeze()
    else:
        input_mask = None

    plotting.plot_sample(
        input_channels_plotting,
        input,
        label_channels_plotting,
        label,
        input_mask=input_mask,
        ds=dataset.get_ds()
        )


class TestHDF5Dataset(unittest.TestCase):

    def test_getitem(self):
        np.random.seed(0)
        torch.manual_seed(0)
        random.seed(0)
        config = copy.deepcopy(default_config)
        dataset = HDF5Dataset(**config)
        data = dataset[0]
        input = data[0]
        label = data[1]

        input_test = torch.load(os.path.join(testdata_folder, 'test_input.pt'))
        label_test = torch.load(os.path.join(testdata_folder, 'test_label.pt'))
        self.assertTrue(torch.equal(input, input_test))
        self.assertTrue(torch.equal(label, label_test))

    def test_getname(self):
        config = copy.deepcopy(default_config)
        dataset = HDF5Dataset(**config)
        name = dataset.get_name(0)
        self.assertEqual(name, 'batch04_F_101_E15x15_W01_t10610')

    def test_getds(self):
        config = copy.deepcopy(default_config)
        dataset = HDF5Dataset(**config)
        ds = dataset.get_ds()
        ds_test = torch.Tensor([
            16.64444351196289, 16.64444351196289, 11.578947067260742
            ])
        self.assertTrue(torch.equal(ds, ds_test))

    def test_length(self):
        config = copy.deepcopy(default_config)
        dataset = HDF5Dataset(**config)
        self.assertEqual(len(dataset), 1)

    def test_getinputchannels(self):
        config = copy.deepcopy(default_config)
        dataset = HDF5Dataset(**config)
        input_channels = dataset.get_input_channels()
        self.assertEqual(input_channels, ['terrain', 'ux', 'uy', 'uz', 'mask'])

    def test_getlabelchannels(self):
        config = copy.deepcopy(default_config)
        dataset = HDF5Dataset(**config)
        label_channels = dataset.get_label_channels()
        self.assertEqual(label_channels, ['ux', 'uy', 'uz', 'turb'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the HDF5Dataset class')
    parser.add_argument('--plot', action='store_true', help='')
    parser.add_argument('-i', dest='input_mode', type=int, default=5, help='Input mode')
    args = parser.parse_args()

    if args.plot:
        plot_sample(args.input_mode)
    else:
        unittest.main()
