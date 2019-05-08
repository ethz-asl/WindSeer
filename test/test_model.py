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
        input = torch.randn(batch_size, net.num_inputs(), kwargs['n_z'], kwargs['n_y'], kwargs['n_x']).to(device)
        labels = torch.randn(batch_size, net.num_outputs(), kwargs['n_z'], kwargs['n_y'], kwargs['n_x']).to(device)
    else:
        input = torch.randn(batch_size, net.num_inputs(), kwargs['n_z'], kwargs['n_x']).to(device)
        labels = torch.randn(batch_size, net.num_outputs(), kwargs['n_z'], kwargs['n_x']).to(device)

    output = net(input)
    try:
        output = net(input)

    except:
        print(colored('\t\t\tforward failed', 'red'))
        error_counter += 1
        return error_counter, test_counter

    try:
        loss = loss_fn(output, labels)
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

    configs = []
    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 32, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 16, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 16, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 16, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 1,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 32, 'n_y': 32, 'n_z': 32, 'n_downsample_layers': 4,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': False, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': False, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'maxpool',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'averagepool',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': False, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 8, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': True, 'potential_flow': True,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'PReLU', 'activation_args': {'init': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D', 'use_grid_size': True, 'use_turbulence': True, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': False, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': False, 'n_stacked': 3, 'n_epochs': 3,
                    'pass_full_output': True, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': False, 'n_stacked': 6, 'n_epochs': 6,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

    configs.append({'batchsize': 1, 'n_x': 8, 'n_y': 8, 'n_z': 8, 'n_downsample_layers': 2,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 2, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': False, 'n_stacked': 6, 'n_epochs': 6,
                    'pass_full_output': False, 'submodel_terrain_mask': True})

    configs.append({'batchsize': 6, 'n_x': 32, 'n_y': 32, 'n_z': 32, 'n_downsample_layers': 4,'interpolation_mode': 'nearest',
                    'align_corners': False, 'skipping': True, 'use_terrain_mask': True, 'pooling_method': 'striding',
                    'use_fc_layers': True, 'fc_scaling': 16, 'use_mapping_layer': False, 'potential_flow': False,
                    'activation_type': 'LeakyReLU', 'activation_args': {'negative_slope': 0.1}, 'predict_uncertainty': False,
                    'verbose': True, 'submodel_type': 'ModelEDNN3D','use_turbulence': True, 'n_stacked': 2,  'n_epochs': 2,
                    'pass_full_output': False, 'submodel_terrain_mask': False})

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
