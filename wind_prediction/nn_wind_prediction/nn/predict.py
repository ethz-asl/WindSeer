#!/usr/bin/env python

import h5py
from math import trunc
import nn_wind_prediction.utils as utils
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

        loss = 0.0
        loss_ux = 0.0
        loss_uy = 0.0
        loss_uz = 0.0
        loss_nut = 0.0
        velocity_errors = np.zeros((16, len(loader_testset)))

        try:
            predict_uncertainty = params.model['model_args']['predict_uncertainty']
        except KeyError as e:
            predict_uncertainty = False
            print('predict_wind_and_turbulence: predict_uncertainty key not available, setting default value: False')

        for i, data in tqdm(enumerate(loader_testset), total=len(loader_testset)):
            inputs = data[0]
            labels = data[1]

            outputs = net(inputs)

            if predict_uncertainty:
                num_channels = outputs.shape[1]
                dloss = loss_fn(outputs[:,:int(num_channels/2),:], labels)
            else:
                dloss = loss_fn(outputs, labels)
            if dloss > maxloss:
                maxloss = dloss
                worst_index = i

            loss += dloss
            if len(outputs.shape) == 5:
                loss_ux += loss_fn(outputs[:,0,:,:,:], labels[:,0,:,:,:])
                loss_uy += loss_fn(outputs[:,1,:,:,:], labels[:,1,:,:,:])
                loss_uz += loss_fn(outputs[:,2,:,:,:], labels[:,2,:,:,:])
                if params.data['use_turbulence']:
                    loss_nut += loss_fn(outputs[:,3,:,:,:], labels[:,3,:,:,:])

            elif len(outputs.shape) == 4:
                loss_ux += loss_fn(outputs[:,0,:,:], labels[:,0,:,:])
                loss_uz += loss_fn(outputs[:,1,:,:], labels[:,1,:,:])
                if params.data['use_turbulence']:
                    loss_nut += loss_fn(outputs[:,2,:,:], labels[:,2,:,:])

            else:
                print('dataset_prediction_error: unknown dimension of the data:', len(output.shape))
                raise ValueError

            error_stats = utils.prediction_error.compute_prediction_error(labels, outputs, params.data['uhor_scaling'], params.data['uz_scaling'], predict_uncertainty)
            velocity_errors[0, i] = error_stats['avg_abs_error']
            velocity_errors[1, i] = error_stats['avg_abs_error_x']
            velocity_errors[2, i] = error_stats['avg_abs_error_y']
            velocity_errors[3, i] = error_stats['avg_abs_error_z']
            velocity_errors[4, i] = error_stats['low_error_hor']
            velocity_errors[5, i] = error_stats['low_error_vert']
            velocity_errors[6, i] = error_stats['low_error_tot']
            velocity_errors[7, i] = error_stats['high_error_hor']
            velocity_errors[8, i] = error_stats['high_error_vert']
            velocity_errors[9, i] = error_stats['high_error_tot']
            velocity_errors[10, i] = error_stats['max_low_hor']
            velocity_errors[11, i] = error_stats['max_low_vert']
            velocity_errors[12, i] = error_stats['max_low_tot']
            velocity_errors[13, i] = error_stats['max_high_hor']
            velocity_errors[14, i] = error_stats['max_high_vert']
            velocity_errors[15, i] = error_stats['max_high_tot']

            if ((i % np.ceil(len(loader_testset)/20.0)) == 0.0):
                print(trunc((i+1)/len(loader_testset)*100), '%')

        print('INFO: Average loss on test set: %s' % (loss.item()/len(loader_testset)))
        print('INFO: Average loss on test set for ux: %s' % (loss_ux.item()/len(loader_testset)))
        if len(outputs.shape) == 5:
            print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(loader_testset)))
        print('INFO: Average loss on test set for uz: %s' % (loss_uz.item()/len(loader_testset)))
        if params.data['use_turbulence']:
            print('INFO: Average loss on test set for turbulence: %s' % (loss_nut.item()/len(loader_testset)))

        print('INFO: Average absolute error total:   %s [m/s]' % (np.mean(velocity_errors[0, :])))
        print('INFO: Average absolute error x:       %s [m/s]' % (np.mean(velocity_errors[1, :])))
        if len(outputs.shape) == 5:
            print('INFO: Average absolute error y:       %s [m/s]' % (np.mean(velocity_errors[2, :])))
        print('INFO: Average absolute error z:       %s [m/s]' % (np.mean(velocity_errors[3, :])))
        print('INFO: Close terrain error hor:        %s [m/s]' % (np.mean(velocity_errors[4, :])))
        print('INFO: Close terrain error vert:       %s [m/s]' % (np.mean(velocity_errors[5, :])))
        print('INFO: Close terrain error tot:        %s [m/s]' % (np.mean(velocity_errors[6, :])))
        print('INFO: High above terrain error hor:   %s [m/s]' % (np.mean(velocity_errors[7, :])))
        print('INFO: High above terrain error vert:  %s [m/s]' % (np.mean(velocity_errors[8, :])))
        print('INFO: High above terrain error tot:   %s [m/s]' % (np.mean(velocity_errors[9, :])))
        print('INFO: Close terrain max error hor:    %s [m/s]' % (np.mean(velocity_errors[10, :])))
        print('INFO: Close terrain max error vert:   %s [m/s]' % (np.mean(velocity_errors[11, :])))
        print('INFO: Close terrain max error tot:    %s [m/s]' % (np.mean(velocity_errors[12, :])))
        print('INFO: High above terrain maxerr hor:  %s [m/s]' % (np.mean(velocity_errors[13, :])))
        print('INFO: High above terrain maxerr vert: %s [m/s]' % (np.mean(velocity_errors[14, :])))
        print('INFO: High above terrain maxerr tot:  %s [m/s]' % (np.mean(velocity_errors[15, :])))
        
        return velocity_errors, worst_index, maxloss


def predict_wind_and_turbulence(input, label, device, net, params, plotting_prediction, loss_fn = None):
    with torch.no_grad():
        input, label = input.to(device), label.to(device)
        start_time = time.time()
        output = net(input.unsqueeze(0))
        print('INFO: Inference time: ', (time.time() - start_time), 'seconds')
        input = input.squeeze()
        output = output.squeeze()

        try:
            predict_uncertainty = params.model['model_args']['predict_uncertainty']
        except KeyError as e:
            predict_uncertainty = False
            print('predict_wind_and_turbulence: predict_uncertainty key not available, setting default value: False')

        if predict_uncertainty:
            num_channels = output.shape[0]
            if loss_fn:
                print('Loss: {}'.format(loss_fn(output[:int(num_channels/2),:], label)))
        else:
            if loss_fn:
                print('Loss: {}'.format(loss_fn(output, label)))
    
        if plotting_prediction:
            if len(output.shape) == 4:
                output[0,:,:,:] *= params.data['uhor_scaling']
                output[1,:,:,:] *= params.data['uhor_scaling']
                output[2,:,:,:] *= params.data['uz_scaling']
                label[0,:,:,:] *= params.data['uhor_scaling']
                label[1,:,:,:] *= params.data['uhor_scaling']
                label[2,:,:,:] *= params.data['uz_scaling']
                if params.data['use_turbulence']:
                    output[3,:,:,:] *= params.data['turbulence_scaling']
                    label[3,:,:,:] *= params.data['turbulence_scaling']

            elif len(output.shape) == 3:
                output[0,:,:] *= params.data['uhor_scaling']
                output[1,:,:] *= params.data['uz_scaling']
                label[0,:,:] *= params.data['uhor_scaling']
                label[1,:,:] *= params.data['uz_scaling']

            else:
                print('predict_wind_and_turbulence: Unknown dimension of the output:', len(output.shape))

            utils.plot_prediction(output, label, input[0], predict_uncertainty)


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
                        outputs[0,:,:,:] *= params.data['uhor_scaling']
                        outputs[1,:,:,:] *= params.data['uhor_scaling']
                        outputs[2,:,:,:] *= params.data['uz_scaling']

                        if model['params'].data['use_turbulence']:
                            outputs[3,:,:,:] *= params.data['turbulence_scaling']

                        outputs = outputs.cpu()

                        wind = outputs[:3,:,:,:].numpy()
                        if model['params'].data['use_turbulence']:
                            turbulence = outputs[3,:,:,:].numpy()
                        else:
                            turbulence = np.zeros_like(outputs[0,:,:,:].numpy())

                    else:
                        print('ERROR: Unknown dimension of the output:', len(outputs.shape))
                        exit()

                    # save the prediction
                    grp.create_dataset('predictions/' + model['name'] + '/wind', data = wind, dtype='f')
                    grp.create_dataset('predictions/' + model['name'] + '/turbulence', data = turbulence, dtype='f')

                # prepare the inputs and labels
                labels = labels.squeeze()
                inputs = inputs.squeeze()

                labels[0,:,:,:] *= params.data['uhor_scaling']
                labels[1,:,:,:] *= params.data['uhor_scaling']
                labels[2,:,:,:] *= params.data['uz_scaling']
                inputs[1,:,:,:] *= params.data['uhor_scaling']
                inputs[2,:,:,:] *= params.data['uhor_scaling']
                inputs[3,:,:,:] *= params.data['uz_scaling']
                if model['params'].data['use_turbulence']:
                    labels[3,:,:,:] *= params.data['turbulence_scaling']

                inputs, labels = inputs.cpu(), labels.cpu()

                wind_label = labels[:3,:,:,:].numpy()
                if model['params'].data['use_turbulence']:
                    turbulence_label = labels[3,:,:,:].numpy()
                else:
                    turbulence_label = np.zeros_like(labels[0,:,:,:].numpy())

                # save the reference
                grp.create_dataset('reference/wind', data = wind_label, dtype='f')
                grp.create_dataset('reference/turbulence', data = turbulence_label, dtype='f')

                # if the input and output have the same shape then also save the interpolated input as a prediction
                if ((outputs.shape[3] == inputs.shape[3]) and
                    (outputs.shape[2] == inputs.shape[2]) and
                    (outputs.shape[1] == inputs.shape[1])):
                    grp.create_dataset('predictions/interpolated/wind', data = interpolator.edge_interpolation(inputs[1:4,:,:,:]), dtype='f')
                    grp.create_dataset('predictions/interpolated/turbulence', data = np.zeros_like(turbulence_label), dtype='f')

                # create the no wind prediction
                grp.create_dataset('predictions/zerowind/wind', data = np.zeros_like(wind_label), dtype='f')
                grp.create_dataset('predictions/zerowind/turbulence', data = np.zeros_like(turbulence_label), dtype='f')

                # save the grid information
                terrain = (outputs.shape[1] - np.count_nonzero(inputs[0,:,:,:].numpy(), 0)) * ds[2]
                dset_terr = grp.create_dataset('terrain', data = terrain.numpy(), dtype='f')

                grp.create_dataset('grid_info/nx', data = gridshape[2], dtype='i')
                grp.create_dataset('grid_info/ny', data = gridshape[1], dtype='i')
                grp.create_dataset('grid_info/nz', data = gridshape[0], dtype='i')

                grp.create_dataset('grid_info/resolution_horizontal', data = ds[0].item(), dtype='f')
                grp.create_dataset('grid_info/resolution_vertical', data = ds[2].item(), dtype='f')
