#!/usr/bin/env python

import h5py
import windseer.evaluation as eval
import windseer.utils as utils
import windseer.data as nn_data
import windseer.plotting as plotting
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import pandas as pd


def get_prediction(
        input, label, scale, device, net, params, scale_input=False, verbose=False
    ):
    '''
    Get a prediction from the neural network and rescale all tensors.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor
    label : torch.Tensor or None
        Label tensor
    scale : torch.Tensor
        Scale of the sample
    device : torch.Device
        Device where the computations are executed
    net : torch.Module
        Fully trained neural network that is evaluated
    params : WindseerParams
        Parameter dictionary
    scale_input : bool, default : False
        Flag indicating if the input first needs to be scaled according to the config
    verbose : bool, default : True
        Flag indicating if the prediction times are printed to the console

    Returns
    -------
    prediction : dict
        Dictionary with the network predictions
    input_rescaled : torch.Tensor
        Rescaled input tensor
    label_rescaled : torch.Tensor
        Rescaled label tensor
    '''
    with torch.no_grad():
        if scale_input:
            input = utils.scale_tensor(
                input, params.data['input_channels'], scale, params
                )
        input = input.to(device)
        if not label is None:
            label = label.to(device)

            if not len(input.shape) == len(label.shape):
                raise ValueError(
                    'The input and label tensor are expected to have the same number of dimensions'
                    )

        if verbose:
            torch.cuda.synchronize()
            start_time = time.time()

        if len(input.shape) == 4:
            input = input.unsqueeze(0)
            if not label is None:
                label = label.unsqueeze(0)

        elif len(input.shape) < 4 or len(input.shape) > 5:
            raise ValueError('Expected a 4D or 5D tensor')

        prediction = net(input)

        if verbose:
            torch.cuda.synchronize()
            print('INFO: Inference time: ', (time.time() - start_time), 'seconds')

        prediction['pred'] = utils.rescale_tensor(
            prediction['pred'], params.data['label_channels'], scale, params
            )
        if label is None:
            label_rescaled = None
        else:
            label_rescaled = utils.rescale_tensor(
                label, params.data['label_channels'], scale, params
                )

        input_rescaled = utils.rescale_tensor(
            input, params.data['input_channels'], scale, params
            )

        return prediction, input_rescaled, label_rescaled


def compute_prediction_error(
        net,
        device,
        params,
        loss_fn,
        testset,
        single_sample=False,
        sample_index=0,
        num_predictions=100,
        print_output=True
    ):
    '''
    Compute the aggregated prediction errors of a dataset or a single sample.
    Assumes that ux, uy, uz, and optionally the TKE are predicted.

    Parameters
    ----------
    net : torch.Module
        Fully trained neural network that is evaluated
    device : torch.Device
        Device where the computations are executed
    params : WindseerParams
        Parameter dictionary
    loss_fn : CombinedLoss
        Loss function
    testset : HDF5Dataset
        Test dataset
    single_sample : bool, default : False
        Flag indicating if the dataset error is computed or multiple rounds for a single sample
    sample_index : int, default : 0
        In case of a single sample this indicates the sample that is used
    num_predictions : int, default : 100
        In case of a single sample this indicates how many predictions are executed
    print_output : bool, defualt : True
        Flag indicating if the errors are printed to the console

    Returns
    -------
    prediction_errors : dict
        Dictionary with the prediction errors
    losses : dict
        Dictionary with the losses
    worst_index : int
        Index of the sample with the highest loss
    maxloss : torch.Tensor
        Highest loss value
    '''
    for channel in ['ux', 'uy', 'uz']:
        if not channel in params.data['label_channels']:
            raise ValueError('This script assumes that ' + channel + ' is predicted')

    for channel in params.data['label_channels']:
        if not channel in ['ux', 'uy', 'uz', 'turb']:
            raise ValueError('Encountered unexpeted label channel: ' + channel)

    with torch.no_grad():
        worst_index = -1
        maxloss = -np.inf

        if single_sample:
            num_iterations = num_predictions
        else:
            num_iterations = len(testset)

        losses = {
            'loss_total': np.zeros(num_iterations),
            'loss_ux': np.zeros(num_iterations),
            'loss_uy': np.zeros(num_iterations),
            'loss_uz': np.zeros(num_iterations),
            'loss_turb': np.zeros(num_iterations),
            }

        metrics_dataset = {
            'mse': np.zeros(num_iterations),
            'mae': np.zeros(num_iterations),
            'max_error': np.zeros(num_iterations),
            'median_absolute_error': np.zeros(num_iterations),
            'explained_variance_score': np.zeros(num_iterations),
            'r2_score': np.zeros(num_iterations),
            'trajectory_length': np.zeros(num_iterations),
            }

        prediction_errors = {}
        for domain in ['all_', 'low_', 'high_']:
            for property in ['tot_', 'hor_', 'ver_', 'turb_']:
                for metric in ['mean', 'max', 'median']:
                    for relative in ['', '_rel']:
                        prediction_errors[domain + property + metric + relative] = []

        try:
            predict_uncertainty = params.model['model_args']['predict_uncertainty']
        except KeyError as e:
            predict_uncertainty = False
            print(
                'predict_wind_and_turbulence: predict_uncertainty key not available, setting default value: False'
                )

        for i in tqdm(range(num_iterations), total=num_iterations):
            if single_sample:
                data = testset[sample_index]
            else:
                data = testset[i]

            scale = 1.0
            if params.data['autoscale']:
                scale = data[3].item()

            prediction, inputs, labels = get_prediction(
                data[0], data[1], scale, device, net, params
                )

            # compute the overall loss
            dloss = loss_fn(prediction, labels, inputs)
            losses['loss_total'][i] = dloss

            outputs = prediction['pred']
            outputs.squeeze_()
            labels.squeeze_()

            # find the worst prediction
            if dloss > maxloss:
                maxloss = dloss
                worst_index = i

            # compute the losses of the individual channels
            mse_loss = torch.nn.MSELoss()
            index = 0
            for channel in ['ux', 'uy', 'uz', 'turb']:
                losses['loss_' + channel][i] = mse_loss(outputs[index], labels[index])
                index += 1

            # compute the error metrics
            outputs_metrics = outputs.view(outputs.shape[0],
                                           -1).permute(1, 0).cpu().detach()
            labels_metrics = labels.view(labels.shape[0], -1).permute(1,
                                                                      0).cpu().detach()

            metrics_dataset['mse'][i] = metrics.mean_squared_error(
                labels_metrics, outputs_metrics
                )
            metrics_dataset['mae'][i] = metrics.mean_absolute_error(
                labels_metrics, outputs_metrics
                )
            metrics_dataset['max_error'][i] = (labels_metrics -
                                               outputs_metrics).abs().max().item()
            metrics_dataset['median_absolute_error'][i] = metrics.median_absolute_error(
                labels_metrics, outputs_metrics
                )
            metrics_dataset['explained_variance_score'][
                i] = metrics.explained_variance_score(labels_metrics, outputs_metrics)
            metrics_dataset['r2_score'][i] = metrics.r2_score(
                labels_metrics, outputs_metrics
                )

            if params.data['input_mode'] > 2:
                metrics_dataset['trajectory_length'][i] = inputs[0, -1].sum()
            else:
                metrics_dataset['trajectory_length'][i] = 0

            # compute the prediction errors and extract the data
            error_stats = eval.compute_prediction_error_sample(
                labels, outputs, inputs[0, 0], device, 'turb'
                in params.data['label_channels']
                )
            for key in error_stats.keys():
                if not np.isnan(error_stats[key]):
                    prediction_errors[key].append(error_stats[key])

        for key in prediction_errors.keys():
            prediction_errors[key] = np.array(prediction_errors[key])

        if print_output:
            print(
                'INFO: Average loss on test set: %s' % (np.mean(losses['loss_total']))
                )
            print(
                'INFO: Average loss on test set for ux: %s' %
                (np.mean(losses['loss_ux']))
                )
            print(
                'INFO: Average loss on test set for uy: %s' %
                (np.mean(losses['loss_uy']))
                )
            print(
                'INFO: Average loss on test set for uz: %s' %
                (np.mean(losses['loss_uz']))
                )
            print(
                'INFO: Average loss on test set for turbulence: %s' %
                (np.mean(losses['loss_turb']))
                )
            print('')

            for key in metrics_dataset.keys():
                print(
                    'INFO: ' + key + ': ' + '{}'.format(np.mean(metrics_dataset[key]))
                    )
            print('')

            for domain, txt_domain in zip(['all_', 'high_', 'low_'], [
                'INFO: Full domain errors', 'INFO: High above terrain errors',
                'INFO: Close to terrain errors'
                ]):
                for rel, txt_relative in zip(['', '_rel'], ['absolute', 'relative']):
                    print(txt_domain + ', ' + txt_relative + ':')

                    for metric in ['mean', 'median', 'max']:
                        for property, txt_property in zip([
                            'tot_', 'hor_', 'ver_', 'turb_'
                            ], [' total ', ' horizontal ', ' vertical ', ' TKE ']):
                            print(
                                '\t' + metric + txt_property + 'error: {}'.format(
                                    np.mean(
                                        prediction_errors[domain + property + metric +
                                                          rel]
                                        )
                                    )
                                )
                    print('')

        return prediction_errors, losses, metrics_dataset, worst_index, maxloss


def predict_and_visualize(
        dataset,
        index,
        device,
        net,
        params,
        channels_to_plot,
        plot_divergence=False,
        loss_fn=None,
        savename=None,
        plottools=False,
        mayavi=False,
        blocking=False,
        mayavi_configs={}
    ):
    '''
    Predict the flow with the network and visualize the results.

    Parameters
    ----------
    dataset : HDF5Dataset
        HDF5Dataset class
    index : int
        Index of the sample in the dataset that is used
    device : torch.Device
        Device where the computations are executed
    net : torch.Module
        Fully trained neural network that is evaluated
    params : WindseerParams
        Parameter dictionary
    channels_to_plot : str or list of str
        Indicates which channels should be plotted, either 'all' or a list of the channels
    plot_divergence : bool, default: False
        Indicates if the divergence of the prediction should be plotted
    loss_fn : CombinedLoss
        Loss function
    savename : None or str, default: None
        Savename for the prediction tensor, if not None the prediction is saved
    plottools : bool, default : False
        Indicates if the prediction should be visualized with the plottools methods
    mayavi : bool, default : False
        Indicates if the prediction should be visualized with the mayavi methods
    blocking : bool, default : False
        Indicates if at the end the blocking plt.show() is called
    mayavi_configs : dict, default : empty dict
        Configuration parameter for the mayavi plots
    '''
    input_channels = dataset.get_input_channels()

    with torch.no_grad():
        data = dataset[index]
        input = data[0]
        label = data[1]
        scale = 1.0
        if params.data['autoscale']:
            scale = data[3].item()

        prediction, inputs, labels = get_prediction(
            input, label, scale, device, net, params, verbose=True
            )

        pred = prediction['pred'].squeeze()
        input = inputs.squeeze()
        label = labels.squeeze()
        terrain = input[0].squeeze()
        try:
            uncertainty = prediction['logvar'].squeeze()
        except KeyError as e:
            uncertainty = None

        channels_to_predict = params.data['label_channels']

        # make sure the channels to predict exist and are properly ordered
        default_channels = ['terrain', 'ux', 'uy', 'uz', 'turb', 'p', 'epsilon', 'nut']
        for channel in channels_to_predict:
            if channel not in default_channels:
                raise ValueError(
                    'Incorrect label_channel detected: \'{}\', '
                    'correct channels are {}'.format(channel, default_channels)
                    )
        channels_to_predict = [x for x in default_channels if x in channels_to_predict]

        if loss_fn:
            for i, channel in enumerate(channels_to_predict):
                print('Loss ' + channel + ': {}'.format(loss_fn(pred[i], label[i])))
            print('Loss: {}'.format(loss_fn(pred, label)))

        if savename is not None:
            np.save(savename, pred.cpu().numpy())

        if mayavi:
            default_mayavi_configs = {
                'view_settings': None,
                'animate': False,
                'save_animation': False
                }
            mayavi_configs = utils.dict_update(default_mayavi_configs, mayavi_configs)

            ui = []

            if input is not None and input_channels is not None:
                if 'mask' in input_channels:
                    plotting.mlab_plot_measurements(
                        input[1:-1],
                        input[-1],
                        terrain,
                        terrain_mode='blocks',
                        terrain_uniform_color=True,
                        blocking=False,
                        view_settings=mayavi_configs['view_settings'],
                        animate=mayavi_configs['animate'] == 2,
                        save_animation=mayavi_configs['save_animation']
                        )

            if uncertainty is not None:
                ui.append(
                    plotting.mlab_plot_uncertainty(
                        uncertainty,
                        terrain,
                        terrain_uniform_color=True,
                        terrain_mode='blocks',
                        prediction_channels=channels_to_predict,
                        blocking=False,
                        uncertainty_mode='norm',
                        view_settings=mayavi_configs['view_settings'],
                        animate=mayavi_configs['animate'] == 3,
                        save_animation=mayavi_configs['save_animation']
                        )
                    )

            ui.append(
                plotting.mlab_plot_prediction(
                    pred,
                    terrain,
                    terrain_mode='blocks',
                    terrain_uniform_color=True,
                    prediction_channels=channels_to_predict,
                    blocking=False,
                    view_settings=mayavi_configs['view_settings'],
                    animate=mayavi_configs['animate'] == 0,
                    save_animation=mayavi_configs['save_animation']
                    )
                )

            plotting.mlab_plot_streamlines(
                pred,
                terrain,
                terrain_mode='blocks',
                terrain_uniform_color=True,
                blocking=False,
                view_settings=mayavi_configs['view_settings'],
                animate=mayavi_configs['animate'] == 4,
                save_animation=mayavi_configs['save_animation'],
                title='Predicted Flow'
                )

            plotting.mlab_plot_streamlines(
                label,
                terrain,
                terrain_mode='blocks',
                terrain_uniform_color=True,
                blocking=False,
                view_settings=mayavi_configs['view_settings'],
                animate=mayavi_configs['animate'] == 5,
                save_animation=mayavi_configs['save_animation'],
                title='Label Flow'
                )

            ui.append(
                plotting.mlab_plot_error(
                    label - pred,
                    terrain,
                    terrain_uniform_color=True,
                    terrain_mode='blocks',
                    prediction_channels=channels_to_predict,
                    blocking=blocking and not plottools,
                    error_mode='norm',
                    view_settings=mayavi_configs['view_settings'],
                    animate=mayavi_configs['animate'] == 1,
                    save_animation=mayavi_configs['save_animation']
                    )
                )

        if channels_to_plot and plottools:
            if plot_divergence:
                ds = nn_data.get_grid_size(dataset)
            else:
                ds = None

            plotting.plot_prediction(
                provided_prediction_channels=channels_to_predict,
                prediction=pred,
                label=label,
                uncertainty=uncertainty,
                provided_input_channels=input_channels,
                input=input,
                terrain=input[0].squeeze(),
                ds=ds,
                blocking=blocking
                )


def save_prediction_to_database(models_list, device, params, savename, testset):
    '''
    Generate a database of predictions for the planning benchmark

    Parameters
    ----------
    models_list : list of dict
        List of the models, each entry in the list contains the model, the params, and a name
    device : torch.Device
        Device where the computations are executed
    params : WindseerParams
        Parameter dictionary
    savename : None or str, default: None
        Savename for the prediction tensor, if not None the prediction is saved
    testset : HDF5Dataset
        Test dataset
    '''
    if len(models_list) == 0:
        print('ERROR: The given model list is empty')
        exit()

    interpolator = nn_data.interpolation.DataInterpolation(
        torch.device('cpu'), 3, params.model['model_args']['n_x'],
        params.model['model_args']['n_y'], params.model['model_args']['n_z']
        )

    with torch.no_grad():
        with h5py.File(savename, 'w') as f:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=1, shuffle=False, num_workers=0
                )

            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                # create the group name for this sample
                samplename = testset.get_name(i)
                grp = f.create_group(samplename)
                gridshape = None

                scale = 1.0
                if params.data['autoscale']:
                    scale = data[3].item()
                    ds = data[4]
                else:
                    ds = data[3]

                for model in models_list:
                    prediction, inputs, labels = get_prediction(
                        data[0], data[1], scale, device, model['net'], model['params']
                        )

                    outputs = prediction['pred'].squeeze().cpu()

                    if gridshape == None:
                        gridshape = outputs.shape[1:4]
                    else:
                        if gridshape != outputs.shape[1:4]:
                            print(
                                'ERROR: Output shape of the models is not consistent, aborting'
                                )
                            exit()

                    wind = outputs[:3].numpy()

                    if 'turb' in model['params'].data['label_channels']:
                        turbulence = outputs[3].numpy()
                    else:
                        turbulence = np.zeros_like(outputs[0].numpy())

                    # save the prediction
                    grp.create_dataset(
                        'predictions/' + model['name'] + '/wind', data=wind, dtype='f'
                        )
                    grp.create_dataset(
                        'predictions/' + model['name'] + '/turbulence',
                        data=turbulence,
                        dtype='f'
                        )

                # prepare the inputs and labels
                labels = labels.squeeze().cpu()
                inputs = inputs.squeeze().cpu()

                wind_label = labels[:3].numpy()

                if 'turb' in model['params'].data['label_channels']:
                    turbulence_label = labels[3].numpy()
                else:
                    turbulence_label = np.zeros_like(labels[0].numpy())

                # save the reference
                grp.create_dataset('reference/wind', data=wind_label, dtype='f')
                grp.create_dataset(
                    'reference/turbulence', data=turbulence_label, dtype='f'
                    )

                # if the input and output have the same shape then also save the interpolated input as a prediction
                if ((outputs.shape[3] == inputs.shape[3]) and
                    (outputs.shape[2] == inputs.shape[2]) and
                    (outputs.shape[1] == inputs.shape[1])):
                    grp.create_dataset(
                        'predictions/interpolated/wind',
                        data=interpolator.edge_interpolation(inputs[1:4]),
                        dtype='f'
                        )
                    grp.create_dataset(
                        'predictions/interpolated/turbulence',
                        data=np.zeros_like(turbulence_label),
                        dtype='f'
                        )

                # create the no wind prediction
                grp.create_dataset(
                    'predictions/zerowind/wind',
                    data=np.zeros_like(wind_label),
                    dtype='f'
                    )
                grp.create_dataset(
                    'predictions/zerowind/turbulence',
                    data=np.zeros_like(turbulence_label),
                    dtype='f'
                    )

                # save the grid information
                terrain = (outputs.shape[1] -
                           np.count_nonzero(inputs[0].numpy(), 0)) * ds[0][2].item()
                dset_terr = grp.create_dataset('terrain', data=terrain, dtype='f')

                grp.create_dataset('grid_info/nx', data=gridshape[2], dtype='i')
                grp.create_dataset('grid_info/ny', data=gridshape[1], dtype='i')
                grp.create_dataset('grid_info/nz', data=gridshape[0], dtype='i')

                grp.create_dataset(
                    'grid_info/resolution_horizontal', data=ds[0][0].item(), dtype='f'
                    )
                grp.create_dataset(
                    'grid_info/resolution_vertical', data=ds[0][2].item(), dtype='f'
                    )
