#!/usr/bin/env python

import h5py
from math import trunc
import nn_wind_prediction.utils as utils
import nn_wind_prediction.data as nn_data
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def dataset_prediction_error(net, device, params, loss_fn, loader_testset):
    #Compute the average loss
    with torch.no_grad():
        worst_index = -1
        maxloss = -200000.0 #arbitrary large negative number, the loss should always be higher than that number

        losses = {
            'loss_total': np.zeros(len(loader_testset)),
            'loss_ux': np.zeros(len(loader_testset)),
            'loss_uy': np.zeros(len(loader_testset)),
            'loss_uz': np.zeros(len(loader_testset)),
            'loss_turb': np.zeros(len(loader_testset)),
        }

        prediction_errors = {
            'all_tot_mean': [],
            'all_tot_max': [],
            'all_tot_median': [],
            'all_hor_mean': [],
            'all_hor_max': [],
            'all_hor_median': [],
            'all_ver_mean': [],
            'all_ver_max': [],
            'all_ver_median': [],
            'all_turb_mean': [],
            'all_turb_max': [],
            'all_turb_median': [],

            'all_tot_mean_rel': [],
            'all_tot_max_rel': [],
            'all_tot_median_rel': [],
            'all_hor_mean_rel': [],
            'all_hor_max_rel': [],
            'all_hor_median_rel': [],
            'all_ver_mean_rel': [],
            'all_ver_max_rel': [],
            'all_ver_median_rel': [],
            'all_turb_mean_rel': [],
            'all_turb_max_rel': [],
            'all_turb_median_rel': [],

            'low_tot_mean': [],
            'low_tot_max': [],
            'low_tot_median': [],
            'low_hor_mean': [],
            'low_hor_max': [],
            'low_hor_median': [],
            'low_ver_mean': [],
            'low_ver_max': [],
            'low_ver_median': [],
            'low_turb_mean': [],
            'low_turb_max': [],
            'low_turb_median': [],

            'low_tot_mean_rel': [],
            'low_tot_max_rel': [],
            'low_tot_median_rel': [],
            'low_hor_mean_rel': [],
            'low_hor_max_rel': [],
            'low_hor_median_rel': [],
            'low_ver_mean_rel': [],
            'low_ver_max_rel': [],
            'low_ver_median_rel': [],
            'low_turb_mean_rel': [],
            'low_turb_max_rel': [],
            'low_turb_median_rel': [],

            'high_tot_mean': [],
            'high_tot_max': [],
            'high_tot_median': [],
            'high_hor_mean': [],
            'high_hor_max': [],
            'high_hor_median': [],
            'high_ver_mean': [],
            'high_ver_max': [],
            'high_ver_median': [],
            'high_turb_mean': [],
            'high_turb_max': [],
            'high_turb_median': [],

            'high_tot_mean_rel': [],
            'high_tot_max_rel': [],
            'high_tot_median_rel': [],
            'high_hor_mean_rel': [],
            'high_hor_max_rel': [],
            'high_hor_median_rel': [],
            'high_ver_mean_rel': [],
            'high_ver_max_rel': [],
            'high_ver_median_rel': [],
            'high_turb_mean_rel': [],
            'high_turb_max_rel': [],
            'high_turb_median_rel': [],
        }

        try:
            predict_uncertainty = params.model['model_args']['predict_uncertainty']
        except KeyError as e:
            predict_uncertainty = False
            print('predict_wind_and_turbulence: predict_uncertainty key not available, setting default value: False')

        for i, data in tqdm(enumerate(loader_testset), total=len(loader_testset)):
            inputs = data[0]
            labels = data[1]

            inputs, labels = inputs.to(device), labels.to(device)

            if (inputs.shape[0] > 1):
                raise Exception('To compute the prediction error a batchsize of 1 is required')

            outputs = net(inputs)

            # move the tensors to the cpu
            outputs.squeeze_()
            labels.squeeze_()

            # scale the values to the actual velocities
            d3 = (len(list(outputs.size())) > 3)

            scale = 1.0
            if params.data['autoscale']:
                scale = data[2].item()

            if len(labels.shape) == 4:
                labels[0] *= scale * params.data['ux_scaling']
                labels[1] *= scale * params.data['uy_scaling']
                labels[2] *= scale * params.data['uz_scaling']
                outputs[0] *= scale * params.data['ux_scaling']
                outputs[1] *= scale * params.data['uy_scaling']
                outputs[2] *= scale * params.data['uz_scaling']

                if 'turb' in  params.data['label_channels']:
                    labels[3] *= scale * scale * params.data['turbulence_scaling']
                    outputs[3] *= scale * scale * params.data['turbulence_scaling']

            elif len(labels.shape) == 3:
                labels[0] *= scale * params.data['ux_scaling']
                labels[1] *= scale * params.data['uz_scaling']
                outputs[0] *= scale * params.data['ux_scaling']
                outputs[1] *= scale * params.data['uz_scaling']

                if 'turb' in  params.data['label_channels']:
                    labels[3] *= scale * scale * params.data['turbulence_scaling']
                    outputs[3] *= scale * scale * params.data['turbulence_scaling']

            else:
                print('dataset_prediction_error: unknown dimension of the data:', len(outputs.shape))
                raise ValueError

            # compute the overall loss
            if predict_uncertainty:
                num_channels = outputs.shape[1]
                dloss = loss_fn(outputs[:int(num_channels/2)], labels)
            else:
                dloss = loss_fn(outputs, labels)
            losses['loss_total'][i] = dloss

            # find the worst prediction
            if dloss > maxloss:
                maxloss = dloss
                worst_index = i

            # compute the losses of the individual channels
            if len(labels.shape) == 4:
                losses['loss_ux'][i] = loss_fn(outputs[0], labels[0])
                losses['loss_uy'][i] = loss_fn(outputs[1], labels[1])
                losses['loss_uz'][i] = loss_fn(outputs[2], labels[2])
                if 'turb' in  params.data['label_channels']:
                    losses['loss_turb'][i] = loss_fn(outputs[3], labels[3])

            elif len(labels.shape) == 3:
                losses['loss_ux'][i] = loss_fn(outputs[0], labels[0])
                losses['loss_uz'][i] = loss_fn(outputs[1], labels[1])
                if 'turb' in  params.data['label_channels']:
                    losses['loss_turb'][i] = loss_fn(outputs[2], labels[2])

            else:
                print('dataset_prediction_error: unknown dimension of the data:', len(outputs.shape))
                raise ValueError

            # compute the prediction errors and extract the data
            error_stats = utils.prediction_error.compute_prediction_error(labels,
                                                                          outputs,
                                                                          inputs[0,0] * params.data['terrain_scaling'],
                                                                          predict_uncertainty, device,
                                                                          'turb' in  params.data['label_channels'])
            for key in error_stats.keys():
                if not np.isnan(error_stats[key]):
                    prediction_errors[key].append(error_stats[key])

        for key in prediction_errors.keys():
            prediction_errors[key] = np.array(prediction_errors[key])

        # print the results over the full dataset
        print('INFO: Average loss on test set: %s' % (np.mean(losses['loss_total'])))
        print('INFO: Average loss on test set for ux: %s' % (np.mean(losses['loss_ux'])))
        print('INFO: Average loss on test set for uy: %s' % (np.mean(losses['loss_uy'])))
        print('INFO: Average loss on test set for uz: %s' % (np.mean(losses['loss_uz'])))
        print('INFO: Average loss on test set for turbulence: %s' % (np.mean(losses['loss_turb'])))

        print('INFO: Full domain errors, absolute:')
        print('\tmean total velocity error:        %s [m/s]' % (np.mean(prediction_errors['all_tot_mean'])))
        print('\tmean horizontal velocity error:   %s [m/s]' % (np.mean(prediction_errors['all_hor_mean'])))
        print('\tmean vertical velocity error:     %s [m/s]' % (np.mean(prediction_errors['all_ver_mean'])))
        print('\tmean turbulent kinetic energy error:    %s [J/kg]' % (np.mean(prediction_errors['all_turb_mean'])))
        print('\tmedian total velocity error:      %s [m/s]' % (np.mean(prediction_errors['all_tot_median'])))
        print('\tmedian horizontal velocity error: %s [m/s]' % (np.mean(prediction_errors['all_hor_median'])))
        print('\tmedian vertical velocity error:   %s [m/s]' % (np.mean(prediction_errors['all_ver_median'])))
        print('\tmedian turbulent kinetic energy error:  %s [J/kg]' % (np.mean(prediction_errors['all_turb_median'])))
        print('\tmax total velocity error:         %s [m/s]' % (np.mean(prediction_errors['all_tot_max'])))
        print('\tmax horizontal velocity error:    %s [m/s]' % (np.mean(prediction_errors['all_hor_max'])))
        print('\tmax vertical velocity error:      %s [m/s]' % (np.mean(prediction_errors['all_ver_max'])))
        print('\tmax tubulence velocity error:     %s [J/kg]' % (np.mean(prediction_errors['all_turb_max'])))
        print('')

        print('INFO: Full domain errors, relative:')
        print('\tmean total velocity error:        %s' % (np.mean(prediction_errors['all_tot_mean_rel'])))
        print('\tmean horizontal velocity error:   %s' % (np.mean(prediction_errors['all_hor_mean_rel'])))
        print('\tmean vertical velocity error:     %s' % (np.mean(prediction_errors['all_ver_mean_rel'])))
        print('\tmean turbulent kinetic energy error:    %s' % (np.mean(prediction_errors['all_turb_mean_rel'])))
        print('\tmedian total velocity error:      %s' % (np.mean(prediction_errors['all_tot_median_rel'])))
        print('\tmedian horizontal velocity error: %s' % (np.mean(prediction_errors['all_hor_median_rel'])))
        print('\tmedian vertical velocity error:   %s' % (np.mean(prediction_errors['all_ver_median_rel'])))
        print('\tmedian turbulent kinetic energy error:  %s' % (np.mean(prediction_errors['all_turb_median_rel'])))
        print('\tmax total velocity error:         %s' % (np.mean(prediction_errors['all_tot_max_rel'])))
        print('\tmax horizontal velocity error:    %s' % (np.mean(prediction_errors['all_hor_max_rel'])))
        print('\tmax vertical velocity error:      %s' % (np.mean(prediction_errors['all_ver_max_rel'])))
        print('\tmax tubulence velocity error:     %s' % (np.mean(prediction_errors['all_turb_max_rel'])))
        print('')

        print('INFO: High above terrain errors, absolute:')
        print('\tmean total velocity error:        %s [m/s]' % (np.mean(prediction_errors['high_tot_mean'])))
        print('\tmean horizontal velocity error:   %s [m/s]' % (np.mean(prediction_errors['high_hor_mean'])))
        print('\tmean vertical velocity error:     %s [m/s]' % (np.mean(prediction_errors['high_ver_mean'])))
        print('\tmean turbulent kinetic energy error:    %s [J/kg]' % (np.mean(prediction_errors['high_turb_mean'])))
        print('\tmedian total velocity error:      %s [m/s]' % (np.mean(prediction_errors['high_tot_median'])))
        print('\tmedian horizontal velocity error: %s [m/s]' % (np.mean(prediction_errors['high_hor_median'])))
        print('\tmedian vertical velocity error:   %s [m/s]' % (np.mean(prediction_errors['high_ver_median'])))
        print('\tmedian turbulent kinetic energy error:  %s [J/kg]' % (np.mean(prediction_errors['high_turb_median'])))
        print('\tmax total velocity error:         %s [m/s]' % (np.mean(prediction_errors['high_tot_max'])))
        print('\tmax horizontal velocity error:    %s [m/s]' % (np.mean(prediction_errors['high_hor_max'])))
        print('\tmax vertical velocity error:      %s [m/s]' % (np.mean(prediction_errors['high_ver_max'])))
        print('\tmax turbulent kinetic energy error:     %s [J/kg]' % (np.mean(prediction_errors['high_turb_max'])))
        print('')

        print('INFO: High above terrain errors, relative:')
        print('\tmean total velocity error:        %s' % (np.mean(prediction_errors['high_tot_mean_rel'])))
        print('\tmean horizontal velocity error:   %s' % (np.mean(prediction_errors['high_hor_mean_rel'])))
        print('\tmean vertical velocity error:     %s' % (np.mean(prediction_errors['high_ver_mean_rel'])))
        print('\tmean turbulent kinetic energy error:    %s' % (np.mean(prediction_errors['high_turb_mean_rel'])))
        print('\tmedian total velocity error:      %s' % (np.mean(prediction_errors['high_tot_median_rel'])))
        print('\tmedian horizontal velocity error: %s' % (np.mean(prediction_errors['high_hor_median_rel'])))
        print('\tmedian vertical velocity error:   %s' % (np.mean(prediction_errors['high_ver_median_rel'])))
        print('\tmedian turbulent kinetic energy error:  %s' % (np.mean(prediction_errors['high_turb_median_rel'])))
        print('\tmax total velocity error:         %s' % (np.mean(prediction_errors['high_tot_max_rel'])))
        print('\tmax horizontal velocity error:    %s' % (np.mean(prediction_errors['high_hor_max_rel'])))
        print('\tmax vertical velocity error:      %s' % (np.mean(prediction_errors['high_ver_max_rel'])))
        print('\tmax turbulent kinetic energy error:     %s' % (np.mean(prediction_errors['high_turb_max_rel'])))
        print('')

        print('INFO: Close to terrain errors, absolute:')
        print('\tmean total velocity error:        %s [m/s]' % (np.mean(prediction_errors['low_tot_mean'])))
        print('\tmean horizontal velocity error:   %s [m/s]' % (np.mean(prediction_errors['low_hor_mean'])))
        print('\tmean vertical velocity error:     %s [m/s]' % (np.mean(prediction_errors['low_ver_mean'])))
        print('\tmean turbulent kinetic energy error:    %s [J/kg]' % (np.mean(prediction_errors['low_turb_mean'])))
        print('\tmedian total velocity error:      %s [m/s]' % (np.mean(prediction_errors['low_tot_median'])))
        print('\tmedian horizontal velocity error: %s [m/s]' % (np.mean(prediction_errors['low_hor_median'])))
        print('\tmedian vertical velocity error:   %s [m/s]' % (np.mean(prediction_errors['low_ver_median'])))
        print('\tmedian turbulent kinetic energy error:  %s [J/kg]' % (np.mean(prediction_errors['low_turb_median'])))
        print('\tmax total velocity error:         %s [m/s]' % (np.mean(prediction_errors['low_tot_max'])))
        print('\tmax horizontal velocity error:    %s [m/s]' % (np.mean(prediction_errors['low_hor_max'])))
        print('\tmax vertical velocity error:      %s [m/s]' % (np.mean(prediction_errors['low_ver_max'])))
        print('\tmax turbulent kinetic energy error:     %s [J/kg]' % (np.mean(prediction_errors['low_turb_max'])))
        print('')

        print('INFO: Close to terrain errors, relative:')
        print('\tmean total velocity error:        %s' % (np.mean(prediction_errors['low_tot_mean_rel'])))
        print('\tmean horizontal velocity error:   %s' % (np.mean(prediction_errors['low_hor_mean_rel'])))
        print('\tmean vertical velocity error:     %s' % (np.mean(prediction_errors['low_ver_mean_rel'])))
        print('\tmean turbulent kinetic energy error:    %s' % (np.mean(prediction_errors['low_turb_mean_rel'])))
        print('\tmedian total velocity error:      %s' % (np.mean(prediction_errors['low_tot_median_rel'])))
        print('\tmedian horizontal velocity error: %s' % (np.mean(prediction_errors['low_hor_median_rel'])))
        print('\tmedian vertical velocity error:   %s' % (np.mean(prediction_errors['low_ver_median_rel'])))
        print('\tmedian turbulent kinetic energy error:  %s' % (np.mean(prediction_errors['low_turb_median_rel'])))
        print('\tmax total velocity error:         %s' % (np.mean(prediction_errors['low_tot_max_rel'])))
        print('\tmax horizontal velocity error:    %s' % (np.mean(prediction_errors['low_hor_max_rel'])))
        print('\tmax vertical velocity error:      %s' % (np.mean(prediction_errors['low_ver_max_rel'])))
        print('\tmax turbulent kinetic energy error:     %s' % (np.mean(prediction_errors['low_turb_max_rel'])))
        print('')

        return prediction_errors, losses, worst_index, maxloss

def predict_channels(input, label, scale, device, net, params, channels_to_plot, dataset,
                     plot_divergence = False, loss_fn = None, savename=None):
    with torch.no_grad():
        # predict and measure how long it takes
        input, label = input.to(device), label.to(device)
        start_time = time.time()
        output = net(input.unsqueeze(0))
        print('INFO: Inference time: ', (time.time() - start_time), 'seconds')
        input = input.squeeze()
        output = output.squeeze()

        channels_to_predict = params.data['label_channels']

        # make sure the channels to predict exist and are properly ordered
        default_channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut']
        for channel in channels_to_predict:
            if channel not in default_channels:
                raise ValueError('Incorrect label_channel detected: \'{}\', '
                                 'correct channels are {}'.format(channel, default_channels))
        channels_to_predict = [x for x in default_channels if x in channels_to_predict]

        # rescale the labels and predictions
        for i, channel in enumerate(channels_to_predict):
            if channel == 'terrain':
                output[i] *= params.data[channel + '_scaling']
                label[i] *= params.data[channel + '_scaling']
            elif channel.startswith('u') or channel == 'nut':
                output[i] *= scale * params.data[channel +'_scaling']
                label[i] *= scale * params.data[channel + '_scaling']
            elif channel == 'p' or channel == 'turb':
                output[i] *= scale * scale * params.data[channel + '_scaling']
                label[i] *= scale * scale * params.data[channel + '_scaling']
            elif channel == 'epsilon':
                output[i] *= scale * scale * scale * params.data[channel + '_scaling']
                label[i] *= scale * scale * scale * params.data[channel + '_scaling']

        try:
            predict_uncertainty = params.model['model_args']['predict_uncertainty']
        except KeyError as e:
            predict_uncertainty = False
            print('predict_channel: predict_uncertainty key not available, setting default value: False')

        if predict_uncertainty:
            num_channels = output.shape[0]
            if loss_fn:
                print('Loss: {}'.format(loss_fn(output[:int(num_channels/2)], label)))
        else:
            if loss_fn:
                for i, channel in enumerate(channels_to_predict):
                    print('Loss ' + channel + ': {}'.format(loss_fn(output[i], label[i])))
                print('Loss: {}'.format(loss_fn(output, label)))

        if savename is not None:
            np.save(savename, output.cpu().numpy())

        if channels_to_plot:
            utils.plot_prediction(channels_to_predict, channels_to_plot, output, label, input[0],predict_uncertainty,
                                  plot_divergence, ds=nn_data.get_grid_size(dataset))

def save_prediction_to_database(models_list, device, params, savename, testset):
    if len(models_list) == 0:
        print('ERROR: The given model list is empty')
        exit()

    interpolator = utils.interpolation.DataInterpolation(torch.device('cpu'), 3,
                                                         params.model['model_args']['n_x'],
                                                         params.model['model_args']['n_y'],
                                                         params.model['model_args']['n_z'])

    with torch.no_grad():
        with h5py.File(savename, 'w') as f:
            testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                                     shuffle=False, num_workers=0)

            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                # create the group name for this sample
                samplename = testset.get_name(i)
                grp = f.create_group(samplename)
                gridshape = None

                # get the prediction and scale it correctly
                inputs = data[0]
                labels = data[1]

                scale = 1.0
                if params.data['autoscale']:
                    scale = data[3].item()
                    ds = data[4]
                else:
                    ds = data[3]

                inputs, labels = inputs.to(device), labels.to(device)

                for model in models_list:
                    outputs = model['net'](inputs)
                    outputs = outputs.squeeze()

                    if gridshape == None:
                        gridshape = outputs.shape[1:4]
                    else:
                        if gridshape != outputs.shape[1:4]:
                            print('ERROR: Output shape of the models is not consistent, aborting')
                            exit()

                    if len(outputs.shape) == 4:
                        outputs[0] *= scale * params.data['ux_scaling']
                        outputs[1] *= scale * params.data['uy_scaling']
                        outputs[2] *= scale * params.data['uz_scaling']

                        if 'turb' in  model['params'].data['label_channels']:
                            outputs[3] *= scale * scale * params.data['turbulence_scaling']

                        outputs = outputs.cpu()

                        wind = outputs[:3].numpy()
                        if 'turb' in  model['params'].data['label_channels']:
                            turbulence = outputs[3].numpy()
                        else:
                            turbulence = np.zeros_like(outputs[0].numpy())

                    else:
                        print('ERROR: Unknown dimension of the output:', len(outputs.shape))
                        exit()

                    # save the prediction
                    grp.create_dataset('predictions/' + model['name'] + '/wind', data = wind, dtype='f')
                    grp.create_dataset('predictions/' + model['name'] + '/turbulence', data = turbulence, dtype='f')

                # prepare the inputs and labels
                labels = labels.squeeze()
                inputs = inputs.squeeze()

                labels[0] *= scale * params.data['ux_scaling']
                labels[1] *= scale * params.data['uy_scaling']
                labels[2] *= scale * params.data['uz_scaling']
                inputs[1] *= scale * params.data['ux_scaling']
                inputs[2] *= scale * params.data['uy_scaling']
                inputs[3] *= scale * params.data['uz_scaling']
                if 'turb' in  params.data['label_channels']:
                    labels[3] *= scale * scale * params.data['turbulence_scaling']

                inputs, labels = inputs.cpu(), labels.cpu()

                wind_label = labels[:3].numpy()
                if 'turb' in  model['params'].data['label_channels']:
                    turbulence_label = labels[3].numpy()
                else:
                    turbulence_label = np.zeros_like(labels[0].numpy())

                # save the reference
                grp.create_dataset('reference/wind', data = wind_label, dtype='f')
                grp.create_dataset('reference/turbulence', data = turbulence_label, dtype='f')

                # if the input and output have the same shape then also save the interpolated input as a prediction
                if ((outputs.shape[3] == inputs.shape[3]) and
                    (outputs.shape[2] == inputs.shape[2]) and
                    (outputs.shape[1] == inputs.shape[1])):
                    grp.create_dataset('predictions/interpolated/wind', data = interpolator.edge_interpolation(inputs[1:4]), dtype='f')
                    grp.create_dataset('predictions/interpolated/turbulence', data = np.zeros_like(turbulence_label), dtype='f')

                # create the no wind prediction
                grp.create_dataset('predictions/zerowind/wind', data = np.zeros_like(wind_label), dtype='f')
                grp.create_dataset('predictions/zerowind/turbulence', data = np.zeros_like(turbulence_label), dtype='f')

                # save the grid information
                terrain = (outputs.shape[1] - np.count_nonzero(inputs[0].numpy(), 0)) * ds[0,2].numpy()
                dset_terr = grp.create_dataset('terrain', data = terrain, dtype='f')

                grp.create_dataset('grid_info/nx', data = gridshape[2], dtype='i')
                grp.create_dataset('grid_info/ny', data = gridshape[1], dtype='i')
                grp.create_dataset('grid_info/nz', data = gridshape[0], dtype='i')

                grp.create_dataset('grid_info/resolution_horizontal', data = ds[0,0].item(), dtype='f')
                grp.create_dataset('grid_info/resolution_vertical', data = ds[0,2].item(), dtype='f')
