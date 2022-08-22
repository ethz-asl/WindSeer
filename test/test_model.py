import nn_wind_prediction.models as models
import torch
import argparse
from termcolor import colored

'''
Uncomment the @profile if it is run using mprof: mprof run --interval 0.001 test_model.py
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(Model, d3, batch_size, error_counter, test_counter, **kwargs):
    test_counter += 1
    loss_fn = torch.nn.MSELoss()
    print('\tTest #{}'.format(test_counter))
    print('\t\tConfig:')
    print('\t\t\tbatch_size: {}'.format(batch_size))
    print('\t\t\tModel: {}'.format(Model))
    print('\t\t\targs: {}'.format(kwargs))
    print('\t\tResult:')

    try:
        net = Model(**kwargs).to(device)
        net.init_params()

    except:
        print(colored('\t\t\tinit failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    if d3:
        input = torch.randn(batch_size, net.get_num_inputs(), kwargs['n_z'], kwargs['n_y'], kwargs['n_x']).to(device)
        labels = torch.randn(batch_size, net.get_num_outputs(), kwargs['n_z'], kwargs['n_y'], kwargs['n_x']).to(device)
    else:
        input = torch.randn(batch_size, net.get_num_inputs(), kwargs['n_z'], kwargs['n_x']).to(device)
        labels = torch.randn(batch_size, net.get_num_outputs(), kwargs['n_z'], kwargs['n_x']).to(device)

    try:
        output = net(input)

    except:
        print(colored('\t\t\tforward failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    try:
        loss = loss_fn(output['pred'], labels)
        loss.backward()

    except:
        print(colored('\t\t\tbackward failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    print(colored('\t\t\tpassed', 'green'))
    return error_counter, test_counter

if __name__ == "__main__":
    error_counter = 0
    test_counter = 0

    default_config = {'batchsize': 2, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False, 'filter_kernel_size': 3,
                    'use_pressure': False, 'use_epsilon': False, 'use_nut': False, 'n_first_conv_channels': 8,
                    'channel_multiplier': 2, 'grid_size': [1,1,1], 'vae': False, 'use_uz_in': True,
                    'logvar_scaling': 10, 'use_sparse_mask': False, 'use_sparse_convolution': False}

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

    print("--------------------------------------------------------")
    print("SplitNet tests")
    for kwarg in configs:
        error_counter, test_counter = test_model(models.SplitNet, True, kwarg['batchsize'], error_counter, test_counter, **kwarg)

    print("--------------------------------------------------------")
    print("ModelEDNN2D tests")
    for kwarg in configs:
        error_counter, test_counter = test_model(models.ModelEDNN2D, False, kwarg['batchsize'], error_counter, test_counter, **kwarg)

    print("--------------------------------------------------------")
    print("ModelEDNN3D tests")
    for kwarg in configs:
        error_counter, test_counter = test_model(models.ModelEDNN3D, True, kwarg['batchsize'], error_counter, test_counter, **kwarg)

    print("--------------------------------------------------------")
    print("ModelTwin tests")
    for kwarg in configs:
        error_counter, test_counter = test_model(models.ModelTwin, True, kwarg['batchsize'], error_counter, test_counter, **kwarg)

    print("--------------------------------------------------------")
    print("ModelStacked tests")
    for kwarg in configs:
        error_counter, test_counter = test_model(models.ModelStacked, True, kwarg['batchsize'], error_counter, test_counter, **kwarg)

    if (error_counter == 0):
        print(colored('{} out of {} test failed'.format(error_counter, test_counter), 'green'))
    else:
        print(colored('{} out of {} test failed'.format(error_counter, test_counter), 'red'))
