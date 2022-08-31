#!/usr/bin/env python
'''
Testcases for the different models
'''

import windseer.nn.models as models

import torch
import torch.nn as nn
import unittest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

default_config = {
    'batchsize': 2,
    'n_x': 8,
    'n_y': 8,
    'n_z': 8,
    'n_downsample_layers': 2,
    'interpolation_mode': 'nearest',
    'align_corners': False,
    'skipping': True,
    'use_terrain_mask': True,
    'pooling_method': 'striding',
    'use_fc_layers': True,
    'fc_scaling': 2,
    'use_mapping_layer': False,
    'potential_flow': False,
    'activation_type': 'LeakyReLU',
    'activation_args': {
        'negative_slope': 0.1
        },
    'predict_uncertainty': False,
    'verbose': True,
    'submodel_type': 'ModelEDNN3D',
    'use_turbulence': True,
    'n_stacked': 3,
    'n_epochs': 3,
    'pass_full_output': False,
    'submodel_terrain_mask': False,
    'filter_kernel_size': 3,
    'use_pressure': False,
    'use_epsilon': False,
    'use_nut': False,
    'n_first_conv_channels': 8,
    'channel_multiplier': 2,
    'grid_size': [1, 1, 1],
    'vae': False,
    'use_uz_in': True,
    'logvar_scaling': 10,
    'use_sparse_mask': False,
    'use_sparse_convolution': False,
    'uncertainty_train_mode': 'alternating'
    }

configs = []
configs.append(default_config.copy())

config = default_config.copy()
config['batchsize'] = 8
configs.append(config)

config = default_config.copy()
config['n_x'] = 16
configs.append(config)

config = default_config.copy()
config['n_y'] = 16
configs.append(config)

config = default_config.copy()
config['n_z'] = 16
configs.append(config)

config = default_config.copy()
config['n_downsample_layers'] = 3
configs.append(config)

config = default_config.copy()
config['n_x'] = 32
config['n_y'] = 32
config['n_z'] = 32
config['n_downsample_layers'] = 4
configs.append(config)

config = default_config.copy()
config['skipping'] = False
configs.append(config)

config = default_config.copy()
config['use_terrain_mask'] = False
configs.append(config)

config = default_config.copy()
config['pooling_method'] = 'maxpool'
configs.append(config)

config = default_config.copy()
config['pooling_method'] = 'averagepool'
configs.append(config)

config = default_config.copy()
config['pooling_method'] = 'averagepool'
configs.append(config)

config = default_config.copy()
config['use_fc_layers'] = False
configs.append(config)

config = default_config.copy()
config['fc_scaling'] = 8
configs.append(config)

config = default_config.copy()
config['use_mapping_layer'] = True
configs.append(config)

config = default_config.copy()
config['activation_type'] = 'PReLU'
config['activation_args'] = {'init': 0.1}
configs.append(config)

config = default_config.copy()
config['use_mapping_layer'] = True
configs.append(config)

config = default_config.copy()
config['submodel_terrain_mask'] = True
configs.append(config)

config = default_config.copy()
config['potential_flow'] = True
configs.append(config)

config = default_config.copy()
config['predict_uncertainty'] = True
configs.append(config)

config = default_config.copy()
config['n_stacked'] = 2
config['n_epochs'] = 2
configs.append(config)

config = default_config.copy()
config['pass_full_output'] = True
configs.append(config)

config = default_config.copy()
config['submodel_terrain_mask'] = True
configs.append(config)

config = default_config.copy()
config['filter_kernel_size'] = 5
configs.append(config)

config = default_config.copy()
config['n_first_conv_channels'] = 12
configs.append(config)

config = default_config.copy()
config['channel_multiplier'] = 4
configs.append(config)

config = default_config.copy()
config['use_turbulence'] = False
configs.append(config)

config = default_config.copy()
config['use_epsilon'] = True
configs.append(config)

config = default_config.copy()
config['use_nut'] = True
configs.append(config)

config = default_config.copy()
config['vae'] = True
configs.append(config)

config = default_config.copy()
config['use_uz_in'] = False
configs.append(config)

config = default_config.copy()
config['logvar_scaling'] = 1
configs.append(config)

config = default_config.copy()
config['use_sparse_mask'] = True
configs.append(config)

config = default_config.copy()
config['use_sparse_convolution'] = True
configs.append(config)


class TestModels(unittest.TestCase):

    def run_test(self, Model, batch_size, **kwargs):
        loss_fn = torch.nn.MSELoss()

        net = Model(**kwargs).to(device)
        net.init_params()

        input = torch.randn(
            batch_size, net.get_num_inputs(), kwargs['n_z'], kwargs['n_y'],
            kwargs['n_x']
            ).to(device)
        labels = torch.randn(
            batch_size, net.get_num_outputs(), kwargs['n_z'], kwargs['n_y'],
            kwargs['n_x']
            ).to(device)

        output = net(input)

        loss = loss_fn(output['pred'], labels)
        loss.backward()

        return True

    def test_SparseConv(self):
        kwargs_conv = {
            'conv_type': nn.Conv3d,
            'mask_exclude_first_dim': True,
            'in_channels': 3,
            'out_channels': 3,
            'kernel_size': 3,
            'padding': 1
            }
        input = torch.zeros(1, 3, 12, 12, 12)
        input[0, :, 7, 7, 7] = 1
        sparse_conv = models.SparseConv(**kwargs_conv)
        output = sparse_conv(input)

        self.assertTrue(torch.equal(input.nonzero(), output.nonzero()))

    def test_ModelEDNN3D(self):
        for kwarg in configs:
            self.assertTrue(
                self.run_test(models.ModelEDNN3D, kwarg['batchsize'], **kwarg)
                )

    def test_ModelStacked(self):
        for kwarg in configs:
            self.assertTrue(
                self.run_test(models.ModelStacked, kwarg['batchsize'], **kwarg)
                )

    def test_ModelTwin(self):
        for kwarg in configs:
            self.assertTrue(
                self.run_test(models.ModelTwin, kwarg['batchsize'], **kwarg)
                )


if __name__ == '__main__':
    unittest.main()
