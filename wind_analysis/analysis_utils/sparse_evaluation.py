import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from .bin_log_data import bin_log_data, extract_window_wind_data
from .interpolation import interpolate_flight_path
from .utils import predict

def prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config, use_gps_time = False):
    with torch.no_grad():
        measurement, _, mask, _ = bin_log_data(wind_data,
                                               grid_dimensions,
                                               method = 'binning',
                                               t_start = t_start,
                                               t_end = t_end,
                                               use_gps_time = use_gps_time)

        measurement = measurement.unsqueeze(0).float()
        mask = mask.unsqueeze(0).float()

        input_idx = []
        if 'ux'  in config.params['model']['input_channels']:
            input_idx.append(0)
        if 'uy'  in config.params['model']['input_channels']:
            input_idx.append(1)
        if 'uz'  in config.params['model']['input_channels']:
            input_idx.append(2)

        input = torch.cat([terrain, measurement[:, input_idx], mask.unsqueeze(0)], dim = 1)

        input = input.to(device)

        return predict(net, input, scale, config.params['model'])

def get_baseline(wind_data, t_start, t_end, method, time_pred = None, binning = False, n_cells = 64, use_gps_time = False):
    prediction_path = {}
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

def evaluate_flight_log(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation = None):
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
        if wind_data_validation is None:
            raise ValueError('Cross flight evaluation needs to have wind_data from a validation flight')
        evaluate_prediction_cross_flight_forward(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation)

    elif config.params['evaluation']['mode'] == 4:
        if wind_data_validation is None:
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
    print('Prediction metric:')
    print('\tAverage absolute error ux: ', np.mean(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
    print('\tAverage absolute error uy: ', np.mean(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
    print('\tAverage absolute error uz: ', np.mean(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))
    print('\tMaximum absolute error ux: ', np.max(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
    print('\tMaximum absolute error uy: ', np.max(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
    print('\tMaximum absolute error uz: ', np.max(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))   

    # plot the prediction
    time = wind_data['time'] * 1e-6
    plt.figure()
    ax = plt.subplot(3,1,1)
    ax.plot(time, wind_data['we_pred'], label='prediction')
    ax.plot(time, wind_data['we'], label='measurements')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.legend()
    ax.set_ylabel('ux | we [m/s]')

    ax = plt.subplot(3,1,2)
    ax.plot(time, wind_data['wn_pred'], label='prediction')
    ax.plot(time, wind_data['wn'], label='measurements')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uy | wn [m/s]')

    ax = plt.subplot(3,1,3)
    ax.plot(time, wind_data['wu_pred'], label='prediction')
    ax.plot(time, -wind_data['wd'], label='measurements')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uz | wu [m/s]')
    ax.set_xlabel('time [s]')

    plt.show()

def evaluate_sliding_window_path(wind_data, scale, terrain, grid_dimensions, net, config, device):
    t_start = wind_data['time'][0] * 1e-6 + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    dt_pred = config.params['evaluation']['dt_pred']
    t_end = t_start + dt
    t_max = wind_data['time'][-1] * 1e-6

    prediction_errors = {'we': [],
                         'wn': [],
                         'wu': [],
                         'we_max': [],
                         'wn_max': [],
                         'wu_max': []}
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

        prediction_path['time'] = wind_data_prediction['time']
        predictions.append(prediction_path)
        vlines.append(t_end)

        prediction_errors['we'].append(
            np.mean(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
        prediction_errors['wn'].append(
            np.mean(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
        prediction_errors['wu'].append(
            np.mean(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))
        prediction_errors['we_max'].append(
            np.max(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
        prediction_errors['wn_max'].append(
            np.max(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
        prediction_errors['wu_max'].append(
            np.max(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))

        if not config.params['evaluation']['cumulative']:
            t_start = t_end
            t_end = t_start + dt

        else:
            t_end += dt

    plt.figure()
    ax = plt.subplot(2,1,1)
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

    plot_predictions(wind_data, predictions, vlines)

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

        masked_error_abs = (measurement - prediction).abs() * mask

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
    if not 'time_gps' in wind_data.keys() or not 'time_gps' in wind_data_validation.keys():
        raise ValueError('GPS time is required for cross flight evaluation')

    t_start = wind_data['time_gps'][0] + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    t_end = t_start + dt

    if config.params['evaluation']['compute_baseline']:
        prediction_path = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                       wind_data['time_gps'], binning = False, use_gps_time = True)
        prediction_path_validation = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                                  wind_data_validation['time_gps'], binning = False, use_gps_time = True)

    else:
        prediction = prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config, True)

        flight_path = {'x': wind_data['x'],
                       'y': wind_data['y'],
                       'z': wind_data['alt']}

        flight_path_validation = {'x': wind_data_validation['x'],
                                  'y': wind_data_validation['y'],
                                  'z': wind_data_validation['alt']}

        prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)
        prediction_path_validation = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path_validation)

    for key in prediction_path.keys():
        wind_data[key] = prediction_path[key]

    for key in prediction_path_validation.keys():
        wind_data_validation[key] = prediction_path_validation[key]

    # compute the metrics
    wind_data_prediction = extract_window_wind_data(wind_data, t_end, None, True)
    print('Prediction Metric Input Log:')
    print('\tAverage absolute error ux: ', np.mean(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
    print('\tAverage absolute error uy: ', np.mean(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
    print('\tAverage absolute error uz: ', np.mean(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))
    print('\tMaximum absolute error ux: ', np.max(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
    print('\tMaximum absolute error uy: ', np.max(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
    print('\tMaximum absolute error uz: ', np.max(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))

    print('Prediction Metric Validation Log:')
    print('\tAverage absolute error ux: ', np.mean(np.abs(wind_data_validation['we'] - wind_data_validation['we_pred'])))
    print('\tAverage absolute error uy: ', np.mean(np.abs(wind_data_validation['wn'] - wind_data_validation['wn_pred'])))
    print('\tAverage absolute error uz: ', np.mean(np.abs(-wind_data_validation['wd'] - wind_data_validation['wu_pred'])))
    print('\tMaximum absolute error ux: ', np.max(np.abs(wind_data_validation['we'] - wind_data_validation['we_pred'])))
    print('\tMaximum absolute error uy: ', np.max(np.abs(wind_data_validation['wn'] - wind_data_validation['wn_pred'])))
    print('\tMaximum absolute error uz: ', np.max(np.abs(-wind_data_validation['wd'] - wind_data_validation['wu_pred'])))

    # plot the prediction
    plt.figure()
    ax = plt.subplot(3,1,1)
    ax.plot(wind_data['time_gps'], wind_data['we_pred'], label='pred input')
    ax.plot(wind_data_validation['time_gps'], wind_data_validation['we_pred'], label='pred validation')
    ax.plot(wind_data['time_gps'], wind_data['we'], label='measurements input')
    ax.plot(wind_data_validation['time_gps'], wind_data_validation['we'], label='measurements validation')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.legend()
    ax.set_ylabel('ux | we [m/s]')

    ax = plt.subplot(3,1,2)
    ax.plot(wind_data['time_gps'], wind_data['wn_pred'], label='pred input')
    ax.plot(wind_data_validation['time_gps'], wind_data_validation['wn_pred'], label='pred validation')
    ax.plot(wind_data['time_gps'], wind_data['wn'], label='measurements input')
    ax.plot(wind_data_validation['time_gps'], wind_data_validation['wn'], label='measurements validation')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uy | wn [m/s]')

    ax = plt.subplot(3,1,3)
    ax.plot(wind_data['time_gps'], wind_data['wu_pred'], label='pred input')
    ax.plot(wind_data_validation['time_gps'], wind_data_validation['wu_pred'], label='pred validation')
    ax.plot(wind_data['time_gps'], -wind_data['wd'], label='measurements input')
    ax.plot(wind_data_validation['time_gps'], -wind_data_validation['wd'], label='measurements validation')
    y_lim = ax.get_ylim()
    ax.add_patch(Rectangle((t_start, y_lim[0] - 1.0),
                           t_end - t_start, y_lim[1] - y_lim[0] + 1.0,
                           alpha = 0.2, color = 'grey'))
    ax.set_ylabel('uz | wu [m/s]')
    ax.set_xlabel('time [s]')

    plt.show()

def evaluate_prediction_cross_flight_sliding_window(wind_data, scale, terrain, grid_dimensions, net, config, device, wind_data_validation):
    if not 'time_gps' in wind_data.keys() or not 'time_gps' in wind_data_validation.keys():
        raise ValueError('GPS time is required for cross flight evaluation')

    t_start = wind_data['time_gps'][0] + config.params['evaluation']['t_start']
    dt = config.params['evaluation']['dt_input']
    dt_pred = config.params['evaluation']['dt_pred']
    t_end = t_start + dt
    t_max = wind_data['time_gps'][-1]

    prediction_errors = {'we': [],
                         'wn': [],
                         'wu': [],
                         'we_max': [],
                         'wn_max': [],
                         'wu_max': [],
                         'we_val': [],
                         'wn_val': [],
                         'wu_val': [],
                         'we_max_val': [],
                         'wn_max_val': [],
                         'wu_max_val': []}
    
    predictions = []
    predictions_validation = []
    vlines = [t_start]

    while (t_end < t_max):
        if config.params['evaluation']['full_flight']:
            t_start_pred = None
            t_end_pred = None
        else:
            t_start_pred = t_end
            t_end_pred = t_end + dt_pred

        wind_data_prediction = extract_window_wind_data(wind_data, t_start_pred, t_end_pred, use_gps_time = True)
        wind_data_prediction_val = extract_window_wind_data(wind_data_validation, t_start_pred, t_end_pred, use_gps_time = True)

        if config.params['evaluation']['compute_baseline']:
            prediction_path = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                           wind_data_prediction['time_gps'], binning = False, use_gps_time = True)
            prediction_path_validation = get_baseline(wind_data, t_start, t_end, config.params['evaluation']['baseline_method'],
                                                      wind_data_prediction_val['time_gps'], binning = False, use_gps_time = True)

        else:
            prediction = prediction_sparse(wind_data, grid_dimensions, t_start, t_end, terrain, net, scale, device, config, use_gps_time = True)

            flight_path = {'x': wind_data_prediction['x'],
                           'y': wind_data_prediction['y'],
                           'z': wind_data_prediction['alt']}

            flight_path_validation = {'x': wind_data_prediction_val['x'],
                                      'y': wind_data_prediction_val['y'],
                                      'z': wind_data_prediction_val['alt']}

            prediction_path = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path)
            prediction_path_validation = interpolate_flight_path(prediction['pred'], grid_dimensions, flight_path_validation)

        for key in prediction_path.keys():
            wind_data_prediction[key] = prediction_path[key]

        for key in prediction_path_validation.keys():
            wind_data_prediction_val[key] = prediction_path_validation[key]

        prediction_path['time_gps'] = wind_data_prediction['time_gps']
        prediction_path_validation['time_gps'] = wind_data_prediction_val['time_gps']
        predictions.append(prediction_path)
        predictions_validation.append(prediction_path_validation)
        vlines.append(t_end)

        prediction_errors['we'].append(
            np.mean(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
        prediction_errors['wn'].append(
            np.mean(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
        prediction_errors['wu'].append(
            np.mean(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))
        prediction_errors['we_max'].append(
            np.max(np.abs(wind_data_prediction['we'] - wind_data_prediction['we_pred'])))
        prediction_errors['wn_max'].append(
            np.max(np.abs(wind_data_prediction['wn'] - wind_data_prediction['wn_pred'])))
        prediction_errors['wu_max'].append(
            np.max(np.abs(-wind_data_prediction['wd'] - wind_data_prediction['wu_pred'])))

        if len(wind_data_prediction_val['time']) > 0:
            prediction_errors['we_val'].append(
                np.mean(np.abs(wind_data_prediction_val['we'] - wind_data_prediction_val['we_pred'])))
            prediction_errors['wn_val'].append(
                np.mean(np.abs(wind_data_prediction_val['wn'] - wind_data_prediction_val['wn_pred'])))
            prediction_errors['wu_val'].append(
                np.mean(np.abs(-wind_data_prediction_val['wd'] - wind_data_prediction_val['wu_pred'])))
            prediction_errors['we_max_val'].append(
                np.max(np.abs(wind_data_prediction_val['we'] - wind_data_prediction_val['we_pred'])))
            prediction_errors['wn_max_val'].append(
                np.max(np.abs(wind_data_prediction_val['wn'] - wind_data_prediction_val['wn_pred'])))
            prediction_errors['wu_max_val'].append(
                np.max(np.abs(-wind_data_prediction_val['wd'] - wind_data_prediction_val['wu_pred'])))
        else:
            prediction_errors['we_val'].append(np.NaN)
            prediction_errors['wn_val'].append(np.NaN)
            prediction_errors['wu_val'].append(np.NaN)
            prediction_errors['we_max_val'].append(np.NaN)
            prediction_errors['wn_max_val'].append(np.NaN)
            prediction_errors['wu_max_val'].append(np.NaN)

        if not config.params['evaluation']['cumulative']:
            t_start = t_end
            t_end = t_start + dt

        else:
            t_end += dt

    plt.figure()
    ax = plt.subplot(2,1,1)
    ax.plot(prediction_errors['we'], label='we')
    ax.plot(prediction_errors['wn'], label='wn')
    ax.plot(prediction_errors['wu'], label='wu')
    ax.plot(prediction_errors['we_val'], label='we validation')
    ax.plot(prediction_errors['wn_val'], label='wn validation')
    ax.plot(prediction_errors['wu_val'], label='wu validation')
    ax.set_ylabel('average prediction error [m/s]')

    ax = plt.subplot(2,1,2)
    ax.plot(prediction_errors['we_max'], label='we')
    ax.plot(prediction_errors['wn_max'], label='wn')
    ax.plot(prediction_errors['wu_max'], label='wu')
    ax.plot(prediction_errors['we_max_val'], label='we validation')
    ax.plot(prediction_errors['wn_max_val'], label='wn validation')
    ax.plot(prediction_errors['wu_max_val'], label='wu validation')
    ax.legend()
    ax.set_ylabel('maximum prediction error [m/s]')
    ax.set_xlabel('window number [-]')

    plot_predictions(wind_data, predictions, vlines, use_gps_time = True)
    plot_predictions(wind_data_validation, predictions_validation, vlines, use_gps_time = True)

    plt.show()

def plot_predictions(wind_data, predictions, vlines = None, use_gps_time = False):
    if use_gps_time:
         time = wind_data['time_gps']
    else:
        time = wind_data['time'] * 1e-6
    plt.figure()
    ax1 = plt.subplot(3,1,1)
    ax2 = plt.subplot(3,1,2)
    ax3 = plt.subplot(3,1,3)

    cm = plt.get_cmap('jet')
    num_colors = len(predictions)
    ax1.set_prop_cycle('color', [cm(1.0*i/num_colors) for i in range(num_colors)])
    ax2.set_prop_cycle('color', [cm(1.0*i/num_colors) for i in range(num_colors)])
    ax3.set_prop_cycle('color', [cm(1.0*i/num_colors) for i in range(num_colors)])

    if 'we_raw' in wind_data.keys():
        ax1.plot(time, wind_data['we_raw'], color='grey', alpha=0.5, label='raw measurements')
    if 'wn_raw' in wind_data.keys():
        ax2.plot(time, wind_data['wn_raw'], color='grey', alpha=0.5, label='raw measurements')
    if 'wd_raw' in wind_data.keys():
        ax3.plot(time, -wind_data['wd_raw'], color='grey', alpha=0.5, label='raw measurements')

    ax1.plot(time, wind_data['we'], color='black', label='measurements')
    ax2.plot(time, wind_data['wn'], color='black', label='measurements')
    ax3.plot(time, -wind_data['wd'], color='black', label='measurements')

    for i, pred in enumerate(predictions):
        if use_gps_time:
            time_pred = pred['time_gps']
        else:
            time_pred = pred['time'] * 1e-6

        ax1.plot(time_pred, pred['we_pred'], label='pred window ' + str(i))
        ax2.plot(time_pred, pred['wn_pred'], label='pred window ' + str(i))
        ax3.plot(time_pred, pred['wu_pred'], label='pred window ' + str(i))

    ylim = np.array(ax1.get_ylim())
    text_y = ylim[1]
    ylim[1] += 0.1 * np.diff(ylim)
    ax1.set_ylim(ylim)

    if vlines:
        for i, vl in enumerate(vlines):
            ax1.axvline(x=vl, color='grey', alpha=0.5)
            ax2.axvline(x=vl, color='grey', alpha=0.5)
            ax3.axvline(x=vl, color='grey', alpha=0.5)

            if i > 0:
                ax1.text((vl - vl_old) * 0.5 + vl_old, text_y, str(i - 1),
                         horizontalalignment='center',
                         verticalalignment='center')
            vl_old = vl

    ax1.set_ylabel('ux | we [m/s]')
    ax2.set_ylabel('uy | wn [m/s]')
    ax3.set_ylabel('uz | wu [m/s]')
    ax3.set_xlabel('time [s]')

    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    