import copy
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import pearsonr
import torch

import nn_wind_prediction.utils as nn_utils

from .bin_log_data import bin_log_data, extract_window_wind_data
from .interpolation import interpolate_flight_path
from .utils import predict, get_smooth_data

def prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config, use_gps_time = False, return_input = False):
    with torch.no_grad():
        measurement, _, mask, _ = bin_log_data(wind_data,
                                               grid_dimensions,
                                               method = 'binning',
                                               t_start = t_start,
                                               t_end = t_end,
                                               use_gps_time = use_gps_time)

        input_idx = []
        if 'ux'  in config.params['model']['input_channels']:
            input_idx.append(0)
        if 'uy'  in config.params['model']['input_channels']:
            input_idx.append(1)
        if 'uz'  in config.params['model']['input_channels']:
            input_idx.append(2)

        if config.params['model']['input_smoothing']:
            measurement = get_smooth_data(measurement,
                                          mask.bool(),
                                          config.params['model']['grid_size'],
                                          config.params['model']['input_smoothing_interpolation'],
                                          config.params['model']['input_smoothing_interpolation_linear'])

        measurement = measurement.unsqueeze(0).float()
        mask = mask.unsqueeze(0).float()

        input = torch.cat([terrain, measurement[:, input_idx], mask.unsqueeze(0)], dim = 1)

        input = input.to(device)

        if return_input:
            return predict(net, input, scale, config.params['model']), input
        else:
            return predict(net, input, scale, config.params['model'])

def get_baseline(wind_data, t_start, t_end, method, time_pred = None, binning = False, n_cells = 64, use_gps_time = False):
    if binning:
        if method == 'averaging':
            wind_data_in = extract_window_wind_data(wind_data, t_start, t_end, use_gps_time)
            prediction = torch.ones((3, n_cells, n_cells, n_cells))
            prediction[0] *= np.mean(wind_data_in['we'])
            prediction[1] *= np.mean(wind_data_in['wn'])
            prediction[2] *= -np.mean(wind_data_in['wd'])

        elif method == 'zero':
            prediction = torch.zeros((3, n_cells, n_cells, n_cells))

        else:
            raise ValueError('Unknown baseline method: ', config.params['evaluation']['baseline_method'])

    else:
        prediction = {}
        if method == 'averaging':
            wind_data_in = extract_window_wind_data(wind_data, t_start, t_end, use_gps_time)
            prediction['we_pred'] = np.mean(wind_data_in['we']) * np.ones_like(time_pred)
            prediction['wn_pred'] = np.mean(wind_data_in['wn']) * np.ones_like(time_pred)
            prediction['wu_pred'] = np.mean(-wind_data_in['wd']) * np.ones_like(time_pred)

        elif method == 'zero':
            prediction['we_pred'] = np.zeros_like(time_pred)
            prediction['wn_pred'] = np.zeros_like(time_pred)
            prediction['wu_pred'] = np.zeros_like(time_pred)

        else:
            raise ValueError('Unknown baseline method: ', config.params['evaluation']['baseline_method'])

    return prediction

def evaluate_flight_log(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation = []):
    # input check
    if config.params['evaluation']['t_start'] < 0 or config.params['evaluation']['t_start'] is None:
        print('Setting the start time to 0')
        config.params['evaluation']['t_start'] = 0

    if config.params['evaluation']['dt_input'] < 0 or config.params['evaluation']['dt_input'] is None:
        raise ValueError('negative window size dt_input is not allowed')

    if config.params['evaluation']['dt_pred'] < 0 or config.params['evaluation']['dt_pred'] is None:
        raise ValueError('negative window size dt_pred is not allowed')

    if config.params['evaluation']['mode'] == 0:
        evaluate_project_forward(wind_data, scale, terrain, grid_dimensions, net, config, device)

    elif config.params['evaluation']['mode'] == 1:
        evaluate_sliding_window_path(wind_data, scale, terrain, grid_dimensions, net, config, device)

    elif config.params['evaluation']['mode'] == 2:
        evaluate_sliding_window_blocks(wind_data, scale, terrain, grid_dimensions, net, config, device)

    elif config.params['evaluation']['mode'] == 3:
        if len(wind_data_validation) < 1:
            raise ValueError('Cross flight evaluation needs to have wind_data from a validation flight')
        evaluate_prediction_cross_flight_forward(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation)

    elif config.params['evaluation']['mode'] == 4:
        if len(wind_data_validation) < 1:
            raise ValueError('Cross flight evaluation needs to have wind_data from a validation flight')
        evaluate_prediction_cross_flight_sliding_window(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation)

    else:
        raise ValueError('Unknown evaluation mode:', config.params['evaluation']['mode'])

def evaluate_project_forward(wind_data, scale, terrain, grid_dimensions, net, config, device):
    t_start = wind_data['time'][0] * 1e-6 + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    t_end = t_start + dt

    if config.params['evaluation']['compute_baseline']:
        prediction_path = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'], wind_data['time'], binning=False)

    else:
        prediction = prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config)

        flight_path = {'x': wind_data['x'],
                       'y': wind_data['y'],
                       'z': wind_data['alt']}

        prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)

    for key in prediction_path.keys():
        wind_data[key] = prediction_path[key]

    # compute the metrics
    wind_data_prediction = extract_window_wind_data(wind_data, t_end, None)

    prediction_errors = {}
    compute_error(wind_data_prediction, prediction_errors, "")
    print('Prediction metric:')
    print('\tAverage absolute error ux: ', prediction_errors['we'][0])
    print('\tAverage absolute error uy: ', prediction_errors['wn'][0])
    print('\tAverage absolute error uz: ', prediction_errors['wu'][0])
    print('\tMaximum absolute error ux: ', prediction_errors['we_max'][0])
    print('\tMaximum absolute error uy: ', prediction_errors['wn_max'][0])
    print('\tMaximum absolute error uz: ', prediction_errors['wu_max'][0])
    print('\tAverage unbiased absolute error ux: ', prediction_errors['we_unbiased'][0])
    print('\tAverage unbiased absolute error uy: ', prediction_errors['wn_unbiased'][0])
    print('\tAverage unbiased absolute error uz: ', prediction_errors['wu_unbiased'][0])
    print('\tError std ux: ', prediction_errors['we_std'][0])
    print('\tError std uy: ', prediction_errors['wn_std'][0])
    print('\tError std uz: ', prediction_errors['wu_std'][0])

    plot_correlation([wind_data_prediction])

    # plot the prediction
    num_plots = 3
    if 'turb_pred' in wind_data_prediction.keys():
        num_plots = 4

    time = wind_data['time'] * 1e-6
    plt.figure()
    ax = plt.subplot(num_plots,1,1)
    ax.plot(time, wind_data['we_pred'], label='prediction')
    ax.plot(time, wind_data['we'], label='measurements')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.legend()
    ax.set_ylabel('ux | we [m/s]')

    ax = plt.subplot(num_plots,1,2)
    ax.plot(time, wind_data['wn_pred'], label='prediction')
    ax.plot(time, wind_data['wn'], label='measurements')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uy | wn [m/s]')

    ax = plt.subplot(num_plots,1,3)
    ax.plot(time, wind_data['wu_pred'], label='prediction')
    ax.plot(time, -wind_data['wd'], label='measurements')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uz | wu [m/s]')
    ax.set_xlabel('time [s]')

    if 'turb_pred' in wind_data_prediction.keys():
        measured_tke = 0.5 * ((wind_data['wn_raw']-wind_data['wn'])**2 +
                              (wind_data['we_raw']-wind_data['we'])**2 +
                              (wind_data['wd_raw']-wind_data['wd'])**2)
        ax = plt.subplot(num_plots,1,4)
        ax.plot(time, wind_data['turb_pred'], label='prediction')
        ax.plot(time, measured_tke, label='measurement')
        y_lim = ax.get_ylim()
        ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                               t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                               alpha = 0.2, color = 'grey'))
        ax.set_ylabel('TKE [m^2/s^2] | noise [m/s]')
        ax.set_xlabel('time [s]')

    plt.show()

def evaluate_sliding_window_path(wind_data, scale, terrain, grid_dimensions, net, config, device):
    t_start = wind_data['time'][0] * 1e-6 + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    dt_pred = config.params['evaluation']['dt_pred']
    t_end = t_start + dt
    t_max = wind_data['time'][-1] * 1e-6

    prediction_errors = {}
    predictions = []
    vlines = [t_start]

    while (t_end < t_max):
        if config.params['evaluation']['full_flight']:
            t_start_pred = None
            t_end_pred = None
        else:
            t_start_pred = t_end
            t_end_pred = t_end + dt_pred

        wind_data_prediction = extract_window_wind_data(wind_data, t_start_pred, t_end_pred)

        if config.params['evaluation']['compute_baseline']:
            prediction_path = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'], wind_data_prediction['time'], False)

        else:
            prediction = prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config)

            flight_path = {'x': wind_data_prediction['x'],
                           'y': wind_data_prediction['y'],
                           'z': wind_data_prediction['alt']}

            prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)

        for key in prediction_path.keys():
            wind_data_prediction[key] = prediction_path[key]

        predictions.append(copy.deepcopy(wind_data_prediction))
        vlines.append(t_end)

        compute_error(wind_data_prediction, prediction_errors, "")

        if not config.params['evaluation']['cumulative']:
            t_start = t_end
            t_end = t_start + dt

        else:
            t_end += dt

    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.set_title("Prediction errors")
    ax.plot(prediction_errors['we'], label='we')
    ax.plot(prediction_errors['wn'], label='wn')
    ax.plot(prediction_errors['wu'], label='wu')
    ax.set_ylabel('average prediction error [m/s]')

    ax = plt.subplot(2,1,2)
    ax.plot(prediction_errors['we_max'], label='we')
    ax.plot(prediction_errors['wn_max'], label='wn')
    ax.plot(prediction_errors['wu_max'], label='wu')
    ax.legend()
    ax.set_ylabel('maximum prediction error [m/s]')
    ax.set_xlabel('window number [-]')

    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.set_title("Unbiased prediction errors")
    ax.plot(prediction_errors['we_unbiased'], label='we')
    ax.plot(prediction_errors['wn_unbiased'], label='wn')
    ax.plot(prediction_errors['wu_unbiased'], label='wu')
    ax.legend()
    ax.set_ylabel('unbiased prediction error [m/s]')

    ax = plt.subplot(2,1,2)
    ax.set_title("Error standard deviation")
    ax.plot(prediction_errors['we_std'], label='we std')
    ax.plot(prediction_errors['wn_std'], label='wn std')
    ax.plot(prediction_errors['wu_std'], label='wu std')
    ax.legend()
    ax.set_ylabel('error standard deviation [m^2/s^2]')
    ax.set_xlabel('window number [-]')

    #plot_correlation(predictions)
    plot_predictions([wind_data], [predictions], vlines, title="Wind Prediction", should_plot_turb=config.params['evaluation']['plot_turbulence'])

    plt.show()

def evaluate_sliding_window_blocks(wind_data, scale, terrain, grid_dimensions, net, config, device):
    t_start = wind_data['time'][0] * 1e-6 + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    dt_pred = config.params['evaluation']['dt_pred']
    t_end = t_start + dt
    t_max = wind_data['time'][-1]* 1e-6

    prediction_errors = {'ux': [],
                         'uy': [],
                         'uz': []}

    while (t_end < t_max):
        measurement, _, mask, _ = bin_log_data(wind_data,
                                               grid_dimensions,
                                               method = 'binning',
                                               t_start = t_end,
                                               t_end = t_end + dt_pred)

        if config.params['evaluation']['compute_baseline']:
            prediction = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                      t_end, t_end + dt_pred, True, grid_dimensions['n_cells'])

        else:
            prediction = prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config)['pred'].cpu().squeeze()

        masked_error_abs = (measurement - prediction[:3]).abs() * mask

        prediction_errors['ux'].append(
             masked_error_abs[0].sum() / mask.sum())
        prediction_errors['uy'].append(
             masked_error_abs[1].sum() / mask.sum())
        prediction_errors['uz'].append(
             masked_error_abs[2].sum() / mask.sum())

        t_start = t_end
        t_end = t_start + dt

    plt.figure()
    plt.plot(prediction_errors['ux'], label='ux')
    plt.plot(prediction_errors['uy'], label='uy')
    plt.plot(prediction_errors['uz'], label='uz')
    plt.legend()
    plt.ylabel('average prediction error [m/s]')
    plt.xlabel('window number [-]')

    plt.show()

def evaluate_prediction_cross_flight_forward(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation):
    if not 'time_gps' in wind_data.keys() or not 'time_gps' in wind_data_validation[0].keys():
        raise ValueError('GPS time is required for cross flight evaluation')

    t_start = wind_data['time_gps'][0] + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    t_end = t_start + dt

    if config.params['evaluation']['compute_baseline']:
        prediction_path = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                       wind_data['time_gps'], binning = False, use_gps_time = True)

        prediction_path_validation = []
        for wd_validation in wind_data_validation:
            prediction_path_validation.append(get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                                           wd_validation['time_gps'], binning = False, use_gps_time = True))

    else:
        prediction = prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config, True)

        flight_path = {'x': wind_data['x'],
                       'y': wind_data['y'],
                       'z': wind_data['alt']}

        prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)

        prediction_path_validation = []
        for wd_validation in wind_data_validation:
            flight_path_validation = {'x': wd_validation['x'],
                                      'y': wd_validation['y'],
                                      'z': wd_validation['alt']}
            prediction_path_validation.append(interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path_validation))

    for key in prediction_path.keys():
        wind_data[key] = prediction_path[key]

    for wd_val, pp_val in zip(wind_data_validation, prediction_path_validation):
        for key in pp_val.keys():
            wd_val[key] = pp_val[key]

    # compute the metrics
    wind_data_prediction = extract_window_wind_data(wind_data, t_end, None, True)
    prediction_errors = {}
    compute_error(wind_data_prediction, prediction_errors, "")
    print('Prediction Metric Input Log:')
    print('\tAverage absolute error ux: ', prediction_errors['we'][0])
    print('\tAverage absolute error uy: ', prediction_errors['wn'][0])
    print('\tAverage absolute error uz: ', prediction_errors['wu'][0])
    print('\tMaximum absolute error ux: ', prediction_errors['we_max'][0])
    print('\tMaximum absolute error uy: ', prediction_errors['wn_max'][0])
    print('\tMaximum absolute error uz: ', prediction_errors['wu_max'][0])
    print('\tAverage unbiased absolute error ux: ', prediction_errors['we_unbiased'][0])
    print('\tAverage unbiased absolute error uy: ', prediction_errors['wn_unbiased'][0])
    print('\tAverage unbiased absolute error uz: ', prediction_errors['wu_unbiased'][0])
    print('\tError std ux: ', prediction_errors['we_std'][0])
    print('\tError std uy: ', prediction_errors['wn_std'][0])
    print('\tError std uz: ', prediction_errors['wu_std'][0])

    plot_correlation([wind_data_prediction], "Correlation plots input flight")

    for i, wd_val in enumerate(wind_data_validation):
        compute_error(wd_val, prediction_errors, "_val" + str(i))

        print('Prediction Metric Validation Flight ' + str(i) + ' Log:')
        print('\tAverage absolute error ux: ', prediction_errors['we_val' + str(i)][0])
        print('\tAverage absolute error uy: ', prediction_errors['wn_val' + str(i)][0])
        print('\tAverage absolute error uz: ', prediction_errors['wu_val' + str(i)][0])
        print('\tMaximum absolute error ux: ', prediction_errors['we_max_val' + str(i)][0])
        print('\tMaximum absolute error uy: ', prediction_errors['wn_max_val' + str(i)][0])
        print('\tMaximum absolute error uz: ', prediction_errors['wu_max_val' + str(i)][0])
        print('\tAverage unbiased absolute error ux: ', prediction_errors['we_unbiased_val' + str(i)][0])
        print('\tAverage unbiased absolute error uy: ', prediction_errors['wn_unbiased_val' + str(i)][0])
        print('\tAverage unbiased absolute error uz: ', prediction_errors['wu_unbiased_val' + str(i)][0])
        print('\tError std ux: ', prediction_errors['we_std_val' + str(i)][0])
        print('\tError std uy: ', prediction_errors['wn_std_val' + str(i)][0])
        print('\tError std uz: ', prediction_errors['wu_std_val' + str(i)][0])

        plot_correlation([wd_val], "Correlation plots validation flight " + str(i))

    # plot the prediction
    num_plots = 3
    plot_turbulence = 'turb_pred' in wind_data_prediction.keys() and config.params['evaluation']['plot_turbulence']
    if plot_turbulence:
        num_plots = 4

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_mpl = prop_cycle.by_key()['color']

    plt.figure()
    ax = plt.subplot(num_plots,1,1)
    ax.plot(wind_data['time_gps'], wind_data['we_pred'], '-', color=colors_mpl[0], label='pred input')
    ax.plot(wind_data['time_gps'], wind_data['we'], '--', color=colors_mpl[0], label='measurements input')
    for i, wd_val in enumerate(wind_data_validation):
        ax.plot(wd_val['time_gps'], wd_val['we_pred'], '-', color=colors_mpl[i+1], label='pred validation ' + str(i))
        ax.plot(wd_val['time_gps'], wd_val['we'], '--', color=colors_mpl[i+1], label='measurements validation ' + str(i))
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.legend()
    ax.set_ylabel('ux | we [m/s]')

    ax = plt.subplot(num_plots,1,2)
    ax.plot(wind_data['time_gps'], wind_data['wn_pred'], '-', color=colors_mpl[0], label='pred input')
    ax.plot(wind_data['time_gps'], wind_data['wn'], '--', color=colors_mpl[0], label='measurements input')
    for i, wd_val in enumerate(wind_data_validation):
        ax.plot(wd_val['time_gps'], wd_val['wn_pred'], '-', color=colors_mpl[i+1], label='pred validation ' + str(i))
        ax.plot(wd_val['time_gps'], wd_val['wn'], '--', color=colors_mpl[i+1], label='measurements validation ' + str(i))
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uy | wn [m/s]')

    ax = plt.subplot(num_plots,1,3)
    ax.plot(wind_data['time_gps'], wind_data['wu_pred'], '-', color=colors_mpl[0], label='pred input')
    ax.plot(wind_data['time_gps'], -wind_data['wd'], '--', color=colors_mpl[0], label='measurements input')
    for i, wd_val in enumerate(wind_data_validation):
        ax.plot(wd_val['time_gps'], wd_val['wu_pred'], '-', color=colors_mpl[i+1], label='pred validation ' + str(i))
        ax.plot(wd_val['time_gps'], -wd_val['wd'], '--', color=colors_mpl[i+1], label='measurements validation ' + str(i))
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uz | wu [m/s]')
    ax.set_xlabel('time [s]')

    if plot_turbulence:
        measured_tke = 0.5 * ((wind_data['wn_raw']-wind_data['wn'])**2 +
                              (wind_data['we_raw']-wind_data['we'])**2 +
                              (wind_data['wd_raw']-wind_data['wd'])**2)

        ax = plt.subplot(num_plots,1,4)
        ax.plot(wind_data['time_gps'], wind_data['turb_pred'], label='pred input')
        ax.plot(wind_data['time_gps'], measured_tke, label='measurement input')
        for i, wd_val in enumerate(wind_data_validation):
            measured_tke_val = 0.5 * ((wd_val['wn_raw']-wd_val['wn'])**2 +
                                      (wd_val['we_raw']-wd_val['we'])**2 +
                                      (wd_val['wd_raw']-wd_val['wd'])**2)
            ax.plot(wd_val['time_gps'], wd_val['turb_pred'], label='pred validation ' + str(i))
            ax.plot(wd_val['time_gps'], measured_tke_val, label='measurement validation ' + str(i))
        y_lim = ax.get_ylim()
        ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                               t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                               alpha = 0.2, color = 'grey'))
        ax.set_ylabel('TKE [m^2/s^2] | noise [m/s]')
        ax.set_xlabel('time [s]')

    plt.show()

def evaluate_prediction_cross_flight_sliding_window(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation):
    if not 'time_gps' in wind_data.keys() or not 'time_gps' in wind_data_validation[0].keys():
        raise ValueError('GPS time is required for cross flight evaluation')

    t_start = wind_data['time_gps'][0] + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    dt_pred = config.params['evaluation']['dt_pred']
    t_end = t_start + dt
    t_max = wind_data['time_gps'][-1]

    prediction_errors = {}
    predictions = []
    predictions_validation = []
    for wd_val in wind_data_validation:
        predictions_validation.append([])
    vlines = [t_start]

    while (t_end < t_max):
        if config.params['evaluation']['full_flight']:
            t_start_pred = None
            t_end_pred = None
        else:
            t_start_pred = t_end
            t_end_pred = t_end + dt_pred

        wind_data_prediction = extract_window_wind_data(wind_data, t_start_pred, t_end_pred, use_gps_time = True)

        wind_data_prediction_val = []
        for wd_val in wind_data_validation:
            wind_data_prediction_val.append(extract_window_wind_data(wd_val, t_start_pred, t_end_pred, use_gps_time = True))

        if config.params['evaluation']['compute_baseline']:
            prediction_path = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                           wind_data_prediction['time_gps'], binning = False, use_gps_time = True)

            prediction_path_validation = []
            for wd_pred_val in wind_data_prediction_val:
                prediction_path_validation.append(get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                                               wd_pred_val['time_gps'], binning = False, use_gps_time = True))

        else:
            prediction = prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config, use_gps_time = True)

            flight_path = {'x': wind_data_prediction['x'],
                           'y': wind_data_prediction['y'],
                           'z': wind_data_prediction['alt']}

            prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)

            prediction_path_validation = []
            for wd_pred_val in wind_data_prediction_val:
                flight_path_validation = {'x': wd_pred_val['x'],
                                          'y': wd_pred_val['y'],
                                          'z': wd_pred_val['alt']}

                prediction_path_validation.append(interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path_validation))

        for key in prediction_path.keys():
            wind_data_prediction[key] = prediction_path[key]

        for wd_pred_val, pp_val in zip(wind_data_prediction_val, prediction_path_validation):
            for key in pp_val.keys():
                wd_pred_val[key] = pp_val[key]

        predictions.append(copy.deepcopy(wind_data_prediction))
        for i, wd_pred_val in enumerate(wind_data_prediction_val):
            predictions_validation[i].append(copy.deepcopy(wd_pred_val))
        vlines.append(t_end)

        compute_error(wind_data_prediction, prediction_errors, "")

        for i, wd_val in enumerate(wind_data_prediction_val):
            compute_error(wd_val, prediction_errors, "_val" + str(i))

        if not config.params['evaluation']['cumulative']:
            t_start = t_end
            t_end = t_start + dt

        else:
            t_end += dt

    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.set_title("Prediction errors")
    ax.plot(prediction_errors['we'], label='we')
    ax.plot(prediction_errors['wn'], label='wn')
    ax.plot(prediction_errors['wu'], label='wu')
    for i in range(len(predictions_validation)):
        ax.plot(prediction_errors['we_val' + str(i)], label='we validation' + str(i))
        ax.plot(prediction_errors['wn_val' + str(i)], label='wn validation' + str(i))
        ax.plot(prediction_errors['wu_val' + str(i)], label='wu validation' + str(i))
    ax.set_ylabel('average prediction error [m/s]')

    ax = plt.subplot(2,1,2)
    ax.plot(prediction_errors['we_max'], label='we')
    ax.plot(prediction_errors['wn_max'], label='wn')
    ax.plot(prediction_errors['wu_max'], label='wu')
    for i in range(len(predictions_validation)):
        ax.plot(prediction_errors['we_max_val' + str(i)], label='we validation' + str(i))
        ax.plot(prediction_errors['wn_max_val' + str(i)], label='wn validation' + str(i))
        ax.plot(prediction_errors['wu_max_val' + str(i)], label='wu validation' + str(i))
    ax.legend()
    ax.set_ylabel('maximum prediction error [m/s]')
    ax.set_xlabel('window number [-]')

    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.set_title("Unbiased prediction errors")
    ax.plot(prediction_errors['we_unbiased'], label='we')
    ax.plot(prediction_errors['wn_unbiased'], label='wn')
    ax.plot(prediction_errors['wu_unbiased'], label='wu')
    for i in range(len(predictions_validation)):
        ax.plot(prediction_errors['we_unbiased_val' + str(i)], label='we validation' + str(i))
        ax.plot(prediction_errors['wn_unbiased_val' + str(i)], label='wn validation' + str(i))
        ax.plot(prediction_errors['wu_unbiased_val' + str(i)], label='wu validation' + str(i))
    ax.legend()
    ax.set_ylabel('unbiased prediction error [m/s]')

    ax = plt.subplot(2,1,2)
    ax.set_title("Error standard deviation")
    ax.plot(prediction_errors['wn_std'], label='we std')
    ax.plot(prediction_errors['we_std'], label='wn std')
    ax.plot(prediction_errors['wu_std'], label='wu std')
    for i in range(len(predictions_validation)):
        ax.plot(prediction_errors['we_std_val' + str(i)], label='we std validation' + str(i))
        ax.plot(prediction_errors['wn_std_val' + str(i)], label='wn std validation' + str(i))
        ax.plot(prediction_errors['wu_std_val' + str(i)], label='wu std validation' + str(i))
    ax.legend()
    ax.set_ylabel('std [m^2/s^2]')
    ax.set_xlabel('window number [-]')

    if config.params['evaluation']['single_figure']:
        plot_predictions([wind_data] + wind_data_validation, [predictions] + predictions_validation, vlines,
                         use_gps_time = True, should_plot_turb=config.params['evaluation']['plot_turbulence'],
                         plot_magnitude=config.params['evaluation']['plot_magnitude'])

    else:
        plot_predictions([wind_data], [predictions], vlines, use_gps_time = True, title="Input flight",
                         should_plot_turb=config.params['evaluation']['plot_turbulence'],
                         plot_magnitude=config.params['evaluation']['plot_magnitude'])

        for i, (wd_val, pred_val) in enumerate(zip(wind_data_validation, predictions_validation)):
            plot_predictions([wd_val], [pred_val], vlines, use_gps_time = True, title="Validation flight " + str(i),
                             should_plot_turb=config.params['evaluation']['plot_turbulence'],
                             plot_magnitude=config.params['evaluation']['plot_magnitude'])

    plt.show()

def compute_error(prediction_data, errors, field_appendix = ""):
    fields = ['we',
              'wn',
              'wu',
              'we_max',
              'wn_max',
              'wu_max',
              'absolute_errors',
              'we_unbiased',
              'wn_unbiased',
              'wu_unbiased',
              'we_std',
              'wn_std',
              'wu_std',]

    for f in fields:
        if not (f  + field_appendix) in errors.keys():
            errors[(f  + field_appendix)] = []

    if len(prediction_data['time']) > 0:
        errors['we' + field_appendix].append(
            np.mean(np.abs(prediction_data['we'] - prediction_data['we_pred'])))
        errors['wn' + field_appendix].append(
            np.mean(np.abs(prediction_data['wn'] - prediction_data['wn_pred'])))
        errors['wu' + field_appendix].append(
            np.mean(np.abs(-prediction_data['wd'] - prediction_data['wu_pred'])))

        errors['we_max' + field_appendix].append(
            np.max(np.abs(prediction_data['we'] - prediction_data['we_pred'])))
        errors['wn_max' + field_appendix].append(
            np.max(np.abs(prediction_data['wn'] - prediction_data['wn_pred'])))
        errors['wu_max' + field_appendix].append(
            np.max(np.abs(-prediction_data['wd'] - prediction_data['wu_pred'])))

        errors['absolute_errors' + field_appendix].append(np.sqrt(
            np.power(np.abs(prediction_data['wn'] - prediction_data['wn_pred']), 2) +
            np.power(np.abs(prediction_data['we'] - prediction_data['we_pred']), 2) +
            np.power(np.abs(-prediction_data['wd'] - prediction_data['wu_pred']), 2)))

        errors['we_unbiased' + field_appendix].append(
            np.mean(np.abs((prediction_data['we'] - np.mean(prediction_data['we'])) - (prediction_data['we_pred'] - np.mean(prediction_data['we_pred'])))))
        errors['wn_unbiased' + field_appendix].append(
            np.mean(np.abs((prediction_data['wn'] - np.mean(prediction_data['wn'])) - (prediction_data['wn_pred'] - np.mean(prediction_data['wn_pred'])))))
        errors['wu_unbiased' + field_appendix].append(
            np.mean(np.abs(-(prediction_data['wd'] - np.mean(prediction_data['wd'])) - (prediction_data['wu_pred'] - np.mean(prediction_data['wu_pred'])))))

        errors['we_std' + field_appendix].append(
            np.std(prediction_data['we'] - prediction_data['we_pred']))
        errors['wn_std' + field_appendix].append(
            np.std(prediction_data['wn'] - prediction_data['wn_pred']))
        errors['wu_std' + field_appendix].append(
            np.std(-prediction_data['wd'] - prediction_data['wu_pred']))

    else:
        for f in fields:
            errors[f + field_appendix].append(np.NaN)

def plot_correlation(prediction_data, title="Correlation plots"):
    num_rows = len(prediction_data)
    fig, axs = plt.subplots(num_rows, 3, squeeze=False)
    fig.suptitle(title)
    for i, pred_data in enumerate(prediction_data):
        axs[i, 0].scatter(pred_data['we'], pred_data['we_pred'])
        axs[i, 1].scatter(pred_data['wn'], pred_data['wn_pred'])
        axs[i, 2].scatter(-pred_data['wd'], pred_data['wu_pred'])
        axs[i, 0].set_ylabel('prediction window ' + str(i) + ' [m/s]')
        axs[i, 0].set_aspect('equal')
        axs[i, 1].set_aspect('equal')
        axs[i, 2].set_aspect('equal')

    axs[-1, 0].set_xlabel('measured we [m/s]')
    axs[-1, 1].set_xlabel('measured wn [m/s]')
    axs[-1, 2].set_xlabel('measured wu [m/s]')

def plot_predictions(wind_data, predictions, vlines = None, use_gps_time = False, title=None, should_plot_turb=True, plot_magnitude=False):
    if not len(wind_data) == len(predictions):
        raise ValueError('The wind data and predictions list must have equal length')

    plot_turbulence = 'turb_pred' in predictions[0][0].keys() and should_plot_turb
    num_plots = 3
    if plot_turbulence:
        num_plots = 4

    plt.figure(figsize=(7.3,2.2))
    ax1 = plt.subplot(num_plots,1,1)
    ax2 = plt.subplot(num_plots,1,2)
    ax3 = plt.subplot(num_plots,1,3)
    if plot_turbulence:
        ax4 = plt.subplot(num_plots,1,4)

    if title:
        ax1.set_title(title)

    if len(wind_data) == 1:
        color_maps = ['jet']
        line_style = ['solid']
        color_raw = ['grey']
        color_meas = ['black']
    else:
        color_maps = ['autumn', 'winter', 'summer']
        line_style = ['dashed', 'dashed', 'dashed']
        color_raw = ['salmon', 'royalblue', 'seagreen']
        color_meas = ['firebrick', 'cornflowerblue', 'mediumseagreen']

    if use_gps_time:
        time_offset = wind_data[0]['time_gps'][0]
    else:
        time_offset = wind_data[0]['time'][0] * 1e-6

    lw = 1.0
    for i, (wd, pred) in enumerate(zip(wind_data, predictions)):
        if use_gps_time:
            time = wd['time_gps'] - time_offset
        else:
            time = wd['time'] * 1e-6 - time_offset

        if plot_magnitude:
            magnitude = np.sqrt(wd['we']**2 + wd['wn']**2)
            direction = np.degrees(np.arctan2(-wd['wn'], -wd['we']))
            ax1.plot(time, magnitude, color=color_meas[i], linestyle=line_style[i], linewidth=lw, label='measurements')
            ax2.plot(time, direction, color=color_meas[i], linestyle=line_style[i], linewidth=lw, label='measurements')
        else:
            ax1.plot(time, wd['we'], color=color_meas[i], linestyle=line_style[i], linewidth=lw, label='measurements')
            ax2.plot(time, wd['wn'], color=color_meas[i], linestyle=line_style[i], linewidth=lw, label='measurements')
        ax3.plot(time, -wd['wd'], color=color_meas[i], linestyle=line_style[i], linewidth=lw, label='measurements')
        if plot_turbulence:
            measured_tke = 0.5 * ((wd['wn_raw']-wd['wn'])**2 +
                                  (wd['we_raw']-wd['we'])**2 +
                                  (wd['wd_raw']-wd['wd'])**2)
            ax4.plot(time, measured_tke, color=colors_meas[i], linestyle=line_style[i], linewidth=lw, label='measurements')

        cm = plt.get_cmap(color_maps[i])
        num_colors = len([True for p in pred if len(p['time'])>0])

        for k, p in enumerate(pred):
            if use_gps_time:
                time_pred = p['time_gps'] - time_offset
            else:
                time_pred = p['time'] * 1e-6 - time_offset

            color_pred = cm(1.0*k/num_colors)

            if plot_magnitude:
                magnitude = np.sqrt(p['we_pred']**2 + p['wn_pred']**2)
                direction = np.degrees(np.arctan2(-p['wn_pred'], -p['we_pred']))
                ax1.plot(time_pred, magnitude, label='pred flight ' + str(i) + ' window ' + str(k), linewidth=lw, color=color_pred)
                ax2.plot(time_pred, direction, label='pred flight ' + str(i) + ' window ' + str(k), linewidth=lw, color=color_pred)
            else:
                ax1.plot(time_pred, p['we_pred'], label='pred flight ' + str(i) + ' window ' + str(k), linewidth=lw, color=color_pred)
                ax2.plot(time_pred, p['wn_pred'], label='pred flight ' + str(i) + ' window ' + str(k), linewidth=lw, color=color_pred)
            ax3.plot(time_pred, p['wu_pred'], label='pred flight ' + str(i) + ' window ' + str(k), linewidth=lw, color=color_pred)
            if plot_turbulence:
                ax4.plot(time_pred, p['turb_pred'], label='pred flight ' + str(i) + ' window ' + str(k), linewidth=lw, color=color_pred)

    ylim = np.array(ax1.get_ylim())
    text_y = ylim[1]
    ylim[1] += 0.1 * np.diff(ylim)
    ax1.set_ylim(ylim)

    alpha = 0.3
    lw_raw = 0.5
    for i, (wd, pred) in enumerate(zip(wind_data, predictions)):
        if use_gps_time:
            time = wd['time_gps'] - time_offset
        else:
            time = wd['time'] * 1e-6 - time_offset

        if plot_magnitude:
            if 'we_raw' in wd.keys() and 'wn_raw' in wd.keys():
                magnitude = np.sqrt(wd['we_raw']**2 + wd['wn_raw']**2)
                direction = np.degrees(np.arctan2(-wd['wn_raw'], -wd['we_raw']))
                ylim = np.array(ax1.get_ylim())
                ax1.plot(time, magnitude, color=color_raw[i], linewidth=lw_raw, alpha=alpha, label='raw measurements', zorder=0)
                ax1.set_ylim(ylim)
                ylim = np.array(ax2.get_ylim())
                ax2.plot(time, direction, color=color_raw[i], linewidth=lw_raw, alpha=alpha, label='raw measurements', zorder=0)
                ax2.set_ylim(ylim)
        else:
            if 'we_raw' in wd.keys():
                ylim = np.array(ax1.get_ylim())
                ax1.plot(time, wd['we_raw'], color=color_raw[i], linewidth=lw_raw, alpha=alpha, label='raw measurements', zorder=0)
                ax1.set_ylim(ylim)
            if 'wn_raw' in wd.keys():
                ylim = np.array(ax2.get_ylim())
                ax2.plot(time, wd['wn_raw'], color=color_raw[i], linewidth=lw_raw, alpha=alpha, label='raw measurements', zorder=0)
                ax2.set_ylim(ylim)
        if 'wd_raw' in wd.keys():
            ylim = np.array(ax3.get_ylim())
            ax3.plot(time, -wd['wd_raw'], color=color_raw[i], linewidth=lw_raw, alpha=alpha, label='raw measurements', zorder=0)
            ax3.set_ylim(ylim)

    if vlines:
        for i, vl in enumerate(vlines):
            ax1.axvline(x=vl-time_offset, color='grey', alpha=0.5)
            ax2.axvline(x=vl-time_offset, color='grey', alpha=0.5)
            ax3.axvline(x=vl-time_offset, color='grey', alpha=0.5)
            if plot_turbulence:
                ax4.axvline(x=vl-time_offset, color='grey', alpha=0.5)

            if i > 0:
                ax1.text((vl - vl_old) * 0.5 + vl_old - time_offset, text_y, str(i - 1),
                         horizontalalignment='center',
                         verticalalignment='center')
            vl_old = vl

    if plot_magnitude:
        ax1.set_ylabel('magnitude [m/s]')
        ax2.set_ylabel('direction [deg]')
    else:
        ax1.set_ylabel('we [m/s]')
        ax2.set_ylabel('wn [m/s]')
    ax3.set_ylabel('wu [m/s]')
    ax1.axes.xaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    if plot_turbulence:
        ax3.axes.xaxis.set_visible(False)
        ax4.set_ylabel('tke [m^2/s^2]')
        ax4.set_xlabel('time [s]')
    else:
        ax3.set_xlabel('time [s]')

    plt.subplots_adjust(top=0.99, bottom=0.12, left=0.05, right=0.81, hspace=0.03)
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

def evaluate_flight_loiters(flight_data, scale, terrain, grid_dimensions, net, config, device):
    # get the average wind for each loiter
    if config.params['evaluation']['early_averaging']:
        for data in flight_data:
            for ld in data['loiter_data']:
                we = data['wind_data']['we_raw'][ld['idx_start']:ld['idx_stop']].mean()
                data['wind_data']['we_raw'][ld['idx_start']:ld['idx_stop']] = we
                data['wind_data']['we'][ld['idx_start']:ld['idx_stop']] = we

                wn = data['wind_data']['wn_raw'][ld['idx_start']:ld['idx_stop']].mean()
                data['wind_data']['wn_raw'][ld['idx_start']:ld['idx_stop']] = wn
                data['wind_data']['wn'][ld['idx_start']:ld['idx_stop']] = wn

                wd = data['wind_data']['wd_raw'][ld['idx_start']:ld['idx_stop']].mean()
                data['wind_data']['wd_raw'][ld['idx_start']:ld['idx_stop']] = wd
                data['wind_data']['wd'][ld['idx_start']:ld['idx_stop']] = wd

    if config.params['evaluation']['benchmark']:
        all_results = []
        for k, data in enumerate(flight_data):
            print('Processing flight ' + str(k))
            for i, ld in enumerate(data['loiter_data']):
                print('\tProcessing loiter ' + str(i))

                t_start = data['wind_data']['time_gps'][ld['idx_start']]
                t_end = data['wind_data']['time_gps'][ld['idx_stop']-1]

                if config.params['evaluation']['compute_baseline']:
                    prediction = {}
                    prediction['pred'] = get_baseline(data['wind_data'], t_start, t_end, config.params['evaluation']['baseline_method'], 
                                                      n_cells=grid_dimensions['n_cells'] ,binning=True, use_gps_time=True)
                else:
                    prediction = prediction_sparse(data['wind_data'], grid_dimensions, t_start, t_end, terrain, net, scale, device, config, use_gps_time = True)

                results = {'we_meas': [],
                           'wn_meas': [],
                           'wu_meas': [],
                           'dir_meas': [],
                           'mag_meas': [],
                           'we_pred': [],
                           'wn_pred': [],
                           'wu_pred': [],
                           'dir_pred': [],
                           'mag_pred': [],
                           'label': [],
                           'name': 'flight' + str(k) + '_loiter' + str(i),
                           'flight_idx': k}

                for data_val in flight_data:
                    for i_val, ld_val in enumerate(data_val['loiter_data']):
                        flight_path = {'x': data_val['wind_data']['x'][ld_val['idx_start']:ld_val['idx_stop']],
                                       'y': data_val['wind_data']['y'][ld_val['idx_start']:ld_val['idx_stop']],
                                       'z': data_val['wind_data']['alt'][ld_val['idx_start']:ld_val['idx_stop']]}

                        prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)

                        we_meas = data_val['wind_data']['we'][ld_val['idx_start']:ld_val['idx_stop']].mean()
                        wn_meas = data_val['wind_data']['wn'][ld_val['idx_start']:ld_val['idx_stop']].mean()
                        we_pred = prediction_path['we_pred'].mean()
                        wn_pred = prediction_path['wn_pred'].mean()
                        results['we_meas'].append(we_meas)
                        results['wn_meas'].append(wn_meas)
                        results['wu_meas'].append(-data_val['wind_data']['wd'][ld_val['idx_start']:ld_val['idx_stop']].mean())
                        results['dir_meas'].append(np.degrees(np.arctan2(-wn_meas, -we_meas)))
                        results['mag_meas'].append(np.sqrt(we_meas**2 + wn_meas**2))
                        results['we_pred'].append(we_pred)
                        results['wn_pred'].append(wn_pred)
                        results['wu_pred'].append(prediction_path['wu_pred'].mean())
                        results['dir_pred'].append(np.degrees(np.arctan2(-wn_pred, -we_pred)))
                        results['mag_pred'].append(np.sqrt(we_pred**2 + wn_pred**2))
                        results['label'].append(data['name'] + '_' + str(i))

                for key in results.keys():
                    if not key == 'label':
                        results[key] = np.array(results[key])

                results['we_error'] = np.abs(results['we_meas'] - results['we_pred']).mean()
                results['wn_error'] = np.abs(results['wn_meas'] - results['wn_pred']).mean()
                results['wu_error'] = np.abs(results['wu_meas'] - results['wu_pred']).mean()
                results['dir_error'] = np.abs(results['dir_meas'] - results['dir_pred']).mean()
                results['mag_error'] = np.abs(results['mag_meas'] - results['mag_pred']).mean()
                stats = pearsonr(results['we_meas'], results['we_pred'])
                results['we_r'] = stats[0]
                stats = pearsonr(results['wn_meas'], results['wn_pred'])
                results['wn_r'] = stats[0]
                stats = pearsonr(results['wu_meas'], results['wu_pred'])
                results['wu_r'] = stats[0]
                stats = pearsonr(results['dir_meas'], results['dir_pred'])
                results['dir_r'] = stats[0]
                stats = pearsonr(results['mag_meas'], results['mag_pred'])
                results['mag_r'] = stats[0]
                all_results.append(results)

        for results in all_results:
            print('-------------------')
            print('Input: ' + str(results['name']))
            print('Error we: ' + str(results['we_error']))
            print('Error wn: ' + str(results['wn_error']))
            print('Error wu: ' + str(results['wu_error']))
            print('Error dir: ' + str(results['dir_error']))
            print('Error mag: ' + str(results['mag_error']))
            print('Pearson Corr We: ' + str(results['we_r']))
            print('Pearson Corr Wn: ' + str(results['wn_r']))
            print('Pearson Corr Wu: ' + str(results['wu_r']))
            print('Pearson Corr dir: ' + str(results['dir_r']))
            print('Pearson Corr mag: ' + str(results['mag_r']))

        for i in range(len(flight_data)):
            print('-------------------')
            print('Averaged over flight ' + str(i))
            print('Error we: ' + str(np.mean([res['we_error'] for res in all_results if res['flight_idx'] == i])))
            print('Error wn: ' + str(np.mean([res['wn_error'] for res in all_results if res['flight_idx'] == i])))
            print('Error wu: ' + str(np.mean([res['wu_error'] for res in all_results if res['flight_idx'] == i])))
            print('Error dir: ' + str(np.mean([res['dir_error'] for res in all_results if res['flight_idx'] == i])))
            print('Error mag: ' + str(np.mean([res['mag_error'] for res in all_results if res['flight_idx'] == i])))
            print('Pearson Corr We: ' + str(np.mean([res['we_r'] for res in all_results if res['flight_idx'] == i])))
            print('Pearson Corr Wn: ' + str(np.mean([res['wn_r'] for res in all_results if res['flight_idx'] == i])))
            print('Pearson Corr Wu: ' + str(np.mean([res['wu_r'] for res in all_results if res['flight_idx'] == i])))
            print('Pearson Corr dir: ' + str(np.mean([res['dir_r'] for res in all_results if res['flight_idx'] == i])))
            print('Pearson Corr mag: ' + str(np.mean([res['mag_r'] for res in all_results if res['flight_idx'] == i])))

        print('-------------------')
        print('Total average:')
        print('Error we: ' + str(np.mean([res['we_error'] for res in all_results])))
        print('Error wn: ' + str(np.mean([res['wn_error'] for res in all_results])))
        print('Error wu: ' + str(np.mean([res['wu_error'] for res in all_results])))
        print('Error dir: ' + str(np.mean([res['dir_error'] for res in all_results])))
        print('Error mag: ' + str(np.mean([res['mag_error'] for res in all_results])))
        print('Pearson Corr We: ' + str(np.mean([res['we_r'] for res in all_results])))
        print('Pearson Corr Wn: ' + str(np.mean([res['wn_r'] for res in all_results])))
        print('Pearson Corr Wu: ' + str(np.mean([res['wu_r'] for res in all_results])))
        print('Pearson Corr dir: ' + str(np.mean([res['dir_r'] for res in all_results])))
        print('Pearson Corr mag: ' + str(np.mean([res['mag_r'] for res in all_results])))
        print('-------------------')

        if config.params['evaluation']['show_plots']:
            fig, ax = plt.subplots(1,3)
            for i in range(len(flight_data)):
                if config.params['evaluation']['plot_magnitude']:
                    ax[0].plot(np.concatenate([res['mag_meas'] for res in all_results if res['flight_idx'] == i]),
                               np.concatenate([res['mag_pred'] for res in all_results if res['flight_idx'] == i]), 'o', label='flight ' + str(i))
                    ax[1].plot(np.concatenate([res['dir_meas'] for res in all_results if res['flight_idx'] == i]),
                               np.concatenate([res['dir_pred'] for res in all_results if res['flight_idx'] == i]), 'o', label='flight ' + str(i))
                else:
                    ax[0].plot(np.concatenate([res['we_meas'] for res in all_results if res['flight_idx'] == i]),
                               np.concatenate([res['we_pred'] for res in all_results if res['flight_idx'] == i]), 'o', label='flight ' + str(i))
                    ax[1].plot(np.concatenate([res['wn_meas'] for res in all_results if res['flight_idx'] == i]),
                               np.concatenate([res['wn_pred'] for res in all_results if res['flight_idx'] == i]), 'o', label='flight ' + str(i))
                ax[2].plot(np.concatenate([res['wu_meas'] for res in all_results if res['flight_idx'] == i]),
                           np.concatenate([res['wu_pred'] for res in all_results if res['flight_idx'] == i]), 'o', label='flight ' + str(i))

            for i in range(len(ax)):
                limits = [np.min([ax[i].get_xlim()[0], ax[i].get_ylim()[0]]), np.max([ax[i].get_xlim()[1], ax[i].get_ylim()[1]])]
                ax[i].set_xlim(limits)
                ax[i].set_ylim(limits)
                ax[i].set_aspect('equal', adjustable='box')

            if config.params['evaluation']['plot_magnitude']:
                ax[0].set_xlabel('Mag meas')
                ax[1].set_xlabel('Dir meas')
                ax[0].set_ylabel('Mag pred')
                ax[1].set_ylabel('Dir pred')
            else:
                ax[0].set_xlabel('W_e meas')
                ax[1].set_xlabel('W_n meas')
                ax[0].set_ylabel('W_e pred')
                ax[1].set_ylabel('W_n pred')
            ax[2].set_xlabel('W_u meas')
            ax[2].set_ylabel('W_u pred')
            ax[2].legend()
            plt.show()

    else:
        loiter_data = flight_data[config.params['evaluation']['input_flight']]['loiter_data'][config.params['evaluation']['input_loiter']]
        t_start = flight_data[config.params['evaluation']['input_flight']]['wind_data']['time_gps'][loiter_data['idx_start']]
        t_end = flight_data[config.params['evaluation']['input_flight']]['wind_data']['time_gps'][loiter_data['idx_stop']-1]

        if config.params['evaluation']['compute_baseline']:
            prediction = {}
            prediction['pred'] = get_baseline(flight_data[config.params['evaluation']['input_flight']]['wind_data'],
                                              t_start, t_end, config.params['evaluation']['baseline_method'], 
                                              n_cells=grid_dimensions['n_cells'] ,binning=True, use_gps_time=True)
            input = None
        else:
            prediction, input = prediction_sparse(flight_data[config.params['evaluation']['input_flight']]['wind_data'],
                                                  grid_dimensions, t_start, t_end, terrain, net, scale, device, config,
                                                  use_gps_time = True, return_input=True)

        results = {'we_meas': [],
                   'wn_meas': [],
                   'wu_meas': [],
                   'dir_meas': [],
                   'mag_meas': [],
                   'we_pred': [],
                   'wn_pred': [],
                   'wu_pred': [],
                   'dir_pred': [],
                   'mag_pred': [],
                   'label': []}

        for data in flight_data:
            for i, ld in enumerate(data['loiter_data']):
                flight_path = {'x': data['wind_data']['x'][ld['idx_start']:ld['idx_stop']],
                               'y': data['wind_data']['y'][ld['idx_start']:ld['idx_stop']],
                               'z': data['wind_data']['alt'][ld['idx_start']:ld['idx_stop']]}

                prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)

                we_meas = data['wind_data']['we'][ld['idx_start']:ld['idx_stop']].mean()
                wn_meas = data['wind_data']['wn'][ld['idx_start']:ld['idx_stop']].mean()
                we_pred = prediction_path['we_pred'].mean()
                wn_pred = prediction_path['wn_pred'].mean()
                results['we_meas'].append(we_meas)
                results['wn_meas'].append(wn_meas)
                results['wu_meas'].append(-data['wind_data']['wd'][ld['idx_start']:ld['idx_stop']].mean())
                results['dir_meas'].append(np.degrees(np.arctan2(-wn_meas, -we_meas)))
                results['mag_meas'].append(np.sqrt(we_meas**2 + wn_meas**2))
                results['we_pred'].append(prediction_path['we_pred'].mean())
                results['wn_pred'].append(prediction_path['wn_pred'].mean())
                results['wu_pred'].append(prediction_path['wu_pred'].mean())
                results['dir_pred'].append(np.degrees(np.arctan2(-wn_pred, -we_pred)))
                results['mag_pred'].append(np.sqrt(we_pred**2 + wn_pred**2))

                results['label'].append(data['name'] + '_' + str(i))

        for key in results.keys():
            if not key == 'label':
                results[key] = np.array(results[key])

        print('Error we: ' + str(np.abs(results['we_meas'] - results['we_pred']).mean()))
        print('Error wn: ' + str(np.abs(results['wn_meas'] - results['wn_pred']).mean()))
        print('Error wu: ' + str(np.abs(results['wu_meas'] - results['wu_pred']).mean()))
        print('Error dir: ' + str(np.abs(results['dir_meas'] - results['dir_pred']).mean()))
        print('Error mag: ' + str(np.abs(results['mag_meas'] - results['mag_pred']).mean()))

        stats = pearsonr(results['we_meas'], results['we_pred'])
        print('Pearson Corr We: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        stats = pearsonr(results['wn_meas'], results['wn_pred'])
        print('Pearson Corr Wn: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        stats = pearsonr(results['wu_meas'], results['wu_pred'])
        print('Pearson Corr Wu: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        stats = pearsonr(results['dir_meas'], results['dir_pred'])
        print('Pearson Corr Dir: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        stats = pearsonr(results['mag_meas'], results['mag_pred'])
        print('Pearson Corr Mag: ' + str(stats[0]) + ', p: ' + str(stats[1]))

        if config.params['evaluation']['show_plots']:
            fig, ax = plt.subplots(3,1)
            if config.params['evaluation']['plot_magnitude']:
                ax[0].plot(results['mag_meas'], 'o', color = 'black')
                ax[0].plot(results['mag_pred'], 'x', color = 'blue')
                ax[1].plot(results['dir_meas'], 'o', color = 'black')
                ax[1].plot(results['dir_pred'], 'x', color = 'blue')
            else:
                ax[0].plot(results['we_meas'], 'o', color = 'black')
                ax[0].plot(results['we_pred'], 'x', color = 'blue')
                ax[1].plot(results['wn_meas'], 'o', color = 'black')
                ax[1].plot(results['wn_pred'], 'x', color = 'blue')
            ax[2].plot(results['wu_meas'], 'o', color = 'black', label='measurement')
            ax[2].plot(results['wu_pred'], 'x', color = 'blue', label='prediction')
            ax[2].legend()
            ax[0].axes.xaxis.set_visible(False)
            ax[1].axes.xaxis.set_visible(False)
            ax[2].set_xlabel('Loiter idx')
            if config.params['evaluation']['plot_magnitude']:
                ax[0].set_ylabel('Magnitude')
                ax[1].set_ylabel('Direction')
            else:
                ax[0].set_ylabel('We')
                ax[1].set_ylabel('Wn')
            ax[2].set_ylabel('Wu')

            if not input is None:
                input = input[0].cpu().detach()

            nn_utils.plot_prediction(config.params['model']['label_channels'],
                         prediction = prediction['pred'][0].cpu().detach(),
                         provided_input_channels = config.params['model']['input_channels'],
                         input = input,
                         terrain = terrain.cpu().squeeze())

