import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import h5py
import random
from scipy.interpolate import RegularGridInterpolator
import torch

import nn_wind_prediction.models as models
import nn_wind_prediction.utils as nn_utils
from analysis_utils import utils

def predict_case(dataset, net, index, input_mast, experiment_name, config, speedup_profiles=False, reference_mast=None):
    # load the data
    h5_file = h5py.File(dataset, 'r')
    scale_keys = list(h5_file['terrain'].keys())
    print('Input data: ', experiment_name, ', scale: ', scale_keys[index], 'mast', input_mast)

    terrain = torch.from_numpy(h5_file['terrain'][scale_keys[index]][...])
    input_meas = torch.zeros(tuple([3]) + tuple(terrain.shape))
    input_mask = torch.zeros(tuple(terrain.shape))
    ds_input = h5_file['measurement'][args.experiment][scale_keys[args.index]][input_mast]
    # shuffle measurements according to height making sure that we fill always the highest value into the respective cell
    positions = ds_input['pos'][...]
    u_vel = ds_input['u'][...]
    v_vel = ds_input['v'][...]
    w_vel = ds_input['w'][...]

    idx_shuffle = np.argsort(positions[:,2])
    for idx in idx_shuffle:
        u = u_vel[idx]
        v = v_vel[idx]
        w = w_vel[idx]

        if (u*u + v*v + w+w) > 0.0:
            meas_idx = np.round(positions[idx]).astype(np.int)
            print(meas_idx)
            if meas_idx.min() >=0 and meas_idx[0] < input_mask.shape[2] and meas_idx[1] < input_mask.shape[1] and meas_idx[2] < input_mask.shape[0]:
                input_mask[meas_idx[2], meas_idx[1], meas_idx[0]] = 1
                input_meas[0,meas_idx[2], meas_idx[1], meas_idx[0]] = u
                input_meas[1,meas_idx[2], meas_idx[1], meas_idx[0]] = v
                input_meas[2,meas_idx[2], meas_idx[1], meas_idx[0]] = w

    # remove any samples inside the terrain
    input_mask *= terrain > 0
    input_meas *= terrain > 0

    input_meas = input_meas.to(device)
    input_mask = input_mask.to(device)
    terrain = terrain.to(device)

    if input_mask.sum() == 0:
        print('All measurements lie inside the terrain or outside of the prediction domain')
        return None

    # fill the holes in the input if requested
    input_measurement = input_meas
    if 'input_smoothing' in config.data.keys():
        if config.data['input_smoothing']:
            input_measurement = utils.get_smooth_data(input_meas,
                                                      input_mask.bool(),
                                                      config.model['model_args']['grid_size'],
                                                      config.data['input_smoothing_interpolation'],
                                                      config.data['input_smoothing_interpolation_linear'])

    input_idx = []
    if 'ux'  in config.data['input_channels']:
        input_idx.append(0)
    if 'uy'  in config.data['input_channels']:
        input_idx.append(1)
    if 'uz'  in config.data['input_channels']:
        input_idx.append(2)

    input = torch.cat([terrain.unsqueeze(0).unsqueeze(0), input_measurement[input_idx].unsqueeze(0), input_mask.unsqueeze(0).unsqueeze(0)], dim = 1)

    with torch.no_grad():
        prediction = utils.predict(net, input, None, config.data)

    # compute the error of the prediction compared to the measurements
    nz, ny, nx = prediction['pred'].shape[2:]

    u_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,0].cpu().detach().numpy())
    v_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,1].cpu().detach().numpy())
    w_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,2].cpu().detach().numpy())

    if config.model['model_args']['use_turbulence']:
        tke_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,3].cpu().detach().numpy()) 
        turb_predicted = True
    else:
        tke_interpolator = None
        turb_predicted = False

    results = {
        'labels_all': [],
        'labels_3d': [],
        'u_meas': [],
        'v_meas': [],
        'w_meas': [],
        's_meas': [],
        'tke_meas': [],
        'u_pred': [],
        'v_pred': [],
        'w_pred': [],
        's_pred': [],
        'tke_pred': [],
        'extrapolated': [],
        'in_bounds': [],
        'extrapolated_3d': [],
        'in_bounds_3d': [],
        }

    for mast in h5_file['measurement'][args.experiment][scale_keys[args.index]].keys():
        ds_mast = h5_file['measurement'][args.experiment][scale_keys[args.index]][mast]
        positions = ds_mast['pos'][...]

        u_meas = ds_mast['u'][...]
        v_meas = ds_mast['v'][...]
        w_meas = ds_mast['w'][...]
        s_meas = ds_mast['s'][...]
        tke_meas = ds_mast['tke'][...]

        lower_bound = (positions < 0).any(axis=1)
        upper_bound = (positions > [nx-1, ny-1, nz-1]).any(axis=1)

        extrapolated = [lb or ub for lb, ub in zip(lower_bound, upper_bound)]
        in_bounds = [not e for e in extrapolated]

        positions_clipped = copy.copy(positions)
        positions_clipped[:, 0] = positions_clipped[:, 0].clip(0, nx-1)
        positions_clipped[:, 1] = positions_clipped[:, 1].clip(0, ny-1)
        positions_clipped[:, 2] = positions_clipped[:, 2].clip(0, nz-1)

        u_pred = u_interpolator(np.fliplr(positions_clipped))
        v_pred = v_interpolator(np.fliplr(positions_clipped))
        w_pred = w_interpolator(np.fliplr(positions_clipped))
        s_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
        if tke_interpolator:
            tke_pred = tke_interpolator(np.fliplr(positions_clipped))

        meas_3d_available = u_meas != 0

        results['s_meas'].extend(s_meas)
        results['s_pred'].extend(s_pred)

        results['u_meas'].extend(u_meas[meas_3d_available])
        results['v_meas'].extend(v_meas[meas_3d_available])
        results['w_meas'].extend(w_meas[meas_3d_available])
        results['tke_meas'].extend(tke_meas[meas_3d_available])

        results['u_pred'].extend(u_pred[meas_3d_available])
        results['v_pred'].extend(v_pred[meas_3d_available])
        results['w_pred'].extend(w_pred[meas_3d_available])
        if tke_interpolator:
            results['tke_pred'].extend(tke_pred[meas_3d_available])

        labels_all = [mast + '_' + "{:.2f}".format(positions[idx, 2]) for idx in range(len(s_meas))]
        labels_3d = [mast + '_' + "{:.2f}".format(positions[idx, 2]) for idx in range(len(s_meas)) if meas_3d_available[idx]]

        results['labels_all'].extend(labels_all)
        results['labels_3d'].extend(labels_3d)
        results['extrapolated'].extend(extrapolated)
        results['in_bounds'].extend(in_bounds)
        results['extrapolated_3d'].extend([e for e, a in zip(extrapolated, meas_3d_available) if a])
        results['in_bounds_3d'].extend([i for i, a in zip(in_bounds, meas_3d_available) if a])

    for key in results.keys():
        if 'meas' in key or 'pred' in key:
            results[key] = np.array(results[key])

    ret = {
        'results': results,
        'turb_predicted': turb_predicted,
        'prediction': prediction,
        'input': input,
        'terrain': terrain,
        }

    if speedup_profiles:
        ret['profiles'] = {}
        ds_lines = h5_file['lines'][scale_keys[args.index]]
        for line_key in ds_lines.keys():

            x = ds_lines[line_key]['x'][...]
            y = ds_lines[line_key]['y'][...]
            z = ds_lines[line_key]['z'][...]

            positions_clipped = copy.copy(np.stack((x,y,z), axis=1))
            positions_clipped[:, 0] = positions_clipped[:, 0].clip(0, nx-1)
            positions_clipped[:, 1] = positions_clipped[:, 1].clip(0, ny-1)
            positions_clipped[:, 2] = positions_clipped[:, 2].clip(0, nz-1)

            u_pred = u_interpolator(np.fliplr(positions_clipped))
            v_pred = v_interpolator(np.fliplr(positions_clipped))
            w_pred = w_interpolator(np.fliplr(positions_clipped))
            s_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

            if reference_mast:
                ds_mast = h5_file['measurement'][args.experiment][scale_keys[args.index]][reference_mast]
                positions = ds_mast['pos'][...]
                s_meas = ds_mast['s'][...]

                idx_ref = np.argmin(np.abs(positions[:,2] - z[0]))
                s_ref = s_meas[idx_ref]

            else:
                s_ref = s_pred[0]

            ret['profiles'][line_key] = {
                'terrain': ds_lines[line_key]['terrain'][...],
                'dist': ds_lines[line_key]['dist'][...],
                'z': z,
                'speedup': (s_pred - s_ref) / s_ref,
                }

    return ret

parser = argparse.ArgumentParser(description='Predict the flow based on the sparse measurements')
parser.add_argument('-d', dest='dataset', required=True, help='The dataset file')
parser.add_argument('-m', dest='input_mast', required=True, help='The input measurement mast')
parser.add_argument('-e', dest='experiment', required=True, help='The experiment name')
parser.add_argument('-i', dest='index', default=0, type=int, help='Index of the case in the dataset used for the prediction')
parser.add_argument('-model_dir', dest='model_dir', required=True, help='The directory of the model')
parser.add_argument('-model_version', dest='model_version', default='latest', help='The model version')
parser.add_argument('-reference_mast', help='The reference mast (used for the speedup calculation)')
parser.add_argument('--mayavi', action='store_true', help='Generate some extra plots using mayavi')
parser.add_argument('--no_gpu', action='store_true', help='Force CPU prediction')
parser.add_argument('--benchmark', action='store_true', help='Benchmark the prediction by looping through all input masks')
parser.add_argument('--profile', action='store_true', help='Compute the speedup profiles along lines')
args = parser.parse_args()

if args.no_gpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the NN
config = nn_utils.EDNNParameters(args.model_dir + '/params.yaml')
NetworkType = getattr(models, config.model['model_type'])
net = NetworkType(**config.model_kwargs())
state_dict = torch.load(args.model_dir + '/' + args.model_version + '.model',
                        map_location=lambda storage, loc: storage)
net.load_state_dict(state_dict)
net.to(device)
net.eval()

if config.data['input_mode'] < 3:
    raise ValueError('Models with an input mode other than 5 are not supported')

config.data['input_channels'] += ['mask']

if args.benchmark:
    results = {}

    h5_file = h5py.File(args.dataset, 'r')
    scale_keys = list(h5_file['terrain'].keys())

    mast_keys = list(h5_file['measurement'][args.experiment][scale_keys[args.index]].keys())
    h5_file.close()

    for mast in mast_keys:
        ret = predict_case(args.dataset, net, args.index, mast, args.experiment, config)
        if ret:
            results[mast] = ret['results']
        else:
            print('Input mast outside the prediction region')

    all_measurements = {}
    for key in results.keys():
        print('---------------------------------------------------')
        print('Prediction using mast: ' + key)
        print('Error U: ' + str(np.nanmean(np.abs(results[key]['u_meas'] - results[key]['u_pred']))))
        print('Error V: ' + str(np.nanmean(np.abs(results[key]['v_meas'] - results[key]['v_pred']))))
        print('Error W: ' + str(np.nanmean(np.abs(results[key]['w_meas'] - results[key]['w_pred']))))
        if ret['turb_predicted']:
            print('Error TKE: ' + str(np.nanmean(np.abs(results[key]['tke_meas'] - results[key]['tke_pred']))))
        print('Error S: ' + str(np.nanmean(np.abs(results[key]['s_meas'] - results[key]['s_pred']))))
        print('Error U rel: ' + str(np.nanmean(np.abs(results[key]['u_meas'] - results[key]['u_pred'])/np.abs(results[key]['u_meas']))))
        print('Error V rel: ' + str(np.nanmean(np.abs(results[key]['v_meas'] - results[key]['v_pred'])/np.abs(results[key]['v_meas']))))
        print('Error W rel: ' + str(np.nanmean(np.abs(results[key]['w_meas'] - results[key]['w_pred'])/np.abs(results[key]['w_meas']))))
        if ret['turb_predicted']:
            print('Error TKE rel: ' + str(np.nanmean(np.abs(results[key]['tke_meas'] - results[key]['tke_pred'])/np.abs(results[key]['tke_meas']))))
        print('Error S rel: ' + str(np.nanmean(np.abs(results[key]['s_meas'] - results[key]['s_pred'])/np.abs(results[key]['s_meas']))))

        for field in results[key].keys():
            if 'meas' in field or 'pred' in field:
                if field in all_measurements.keys():
                    all_measurements[field] = np.concatenate((all_measurements[field],results[key][field]))
                else:
                    all_measurements[field] = results[key][field]
                x = np.array([])

    print('---------------------------------------------------')
    print('Prediction error all masts')
    print('Error U: ' + str(np.nanmean(np.abs(all_measurements['u_meas'] - all_measurements['u_pred']))))
    print('Error V: ' + str(np.nanmean(np.abs(all_measurements['v_meas'] - all_measurements['v_pred']))))
    print('Error W: ' + str(np.nanmean(np.abs(all_measurements['w_meas'] - all_measurements['w_pred']))))
    if ret['turb_predicted']:
        print('Error TKE: ' + str(np.nanmean(np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred']))))
    print('Error S: ' + str(np.nanmean(np.abs(all_measurements['s_meas'] - all_measurements['s_pred']))))
    print('Error U rel: ' + str(np.nanmean(np.abs(all_measurements['u_meas'] - all_measurements['u_pred'])/np.abs(all_measurements['u_meas']))))
    print('Error V rel: ' + str(np.nanmean(np.abs(all_measurements['v_meas'] - all_measurements['v_pred'])/np.abs(all_measurements['v_meas']))))
    print('Error W rel: ' + str(np.nanmean(np.abs(all_measurements['w_meas'] - all_measurements['w_pred'])/np.abs(all_measurements['w_meas']))))
    if ret['turb_predicted']:
        print('Error TKE rel: ' + str(np.nanmean(np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred'])/np.abs(all_measurements['tke_meas']))))
    print('Error S rel: ' + str(np.nanmean(np.abs(all_measurements['s_meas'] - all_measurements['s_pred'])/np.abs(all_measurements['s_meas']))))
    print('---------------------------------------------------')

    plt.figure()
    for key in results.keys():
        plt.plot(results[key]['s_meas'], results[key]['s_pred'], 'o', label=key)
    plt.xlabel('Measurement')
    plt.ylabel('Prediction')
    plt.legend()
    plt.title('Total Velocity')

    plt.figure()
    for key in results.keys():
        plt.plot(results[key]['u_meas'], results[key]['u_pred'], 'o', label=key)
    plt.xlabel('Measurement')
    plt.ylabel('Prediction')
    plt.legend()
    plt.title('U')

    plt.figure()
    for key in results.keys():
        plt.plot(results[key]['v_meas'], results[key]['v_pred'], 'o', label=key)
    plt.xlabel('Measurement')
    plt.ylabel('Prediction')
    plt.legend()
    plt.title('V')

    plt.figure()
    for key in results.keys():
        plt.plot(results[key]['w_meas'], results[key]['w_pred'], 'o', label=key)
    plt.xlabel('Measurement')
    plt.ylabel('Prediction')
    plt.legend()
    plt.title('W')

    if ret['turb_predicted']:
        plt.figure()
        for key in results.keys():
            plt.plot(results[key]['tke_meas'], results[key]['tke_pred'], 'o', label=key)
        plt.xlabel('Measurement')
        plt.ylabel('Prediction')
        plt.legend()
        plt.title('TKE')


    plt.show()

else:
    ret = predict_case(args.dataset, net, args.index, args.input_mast, args.experiment, config, args.profile, args.reference_mast)
    
    if not ret:
        raise ValueError('No prediction because input mast not in prediction domain')

    x_all = np.linspace(0, len(ret['results']['labels_all'])-1, len(ret['results']['labels_all']))
    x_3d = np.linspace(0, len(ret['results']['labels_3d'])-1, len(ret['results']['labels_3d']))

    num_plots = 3
    if ret['turb_predicted']:
        num_plots = 4
    fig, ah = plt.subplots(num_plots, 1, squeeze=False)
    ah[0][0].set_title('Prediction vs Measurement')
    ah[0][0].plot(x_3d, ret['results']['u_meas'], 'sr', label='measurements')
    ah[0][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['u_pred'][ret['results']['extrapolated_3d']], '^c', label='prediction extrapolated')
    ah[0][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['u_pred'][ret['results']['in_bounds_3d']], 'ob', label='prediction')

    ah[1][0].plot(x_3d, ret['results']['v_meas'], 'sr', label='measurements')
    ah[1][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['v_pred'][ret['results']['extrapolated_3d']], '^c', label='prediction extrapolated')
    ah[1][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['v_pred'][ret['results']['in_bounds_3d']], 'ob', label='prediction')

    ah[2][0].plot(x_3d, ret['results']['w_meas'], 'sr', label='measurements')
    ah[2][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['w_pred'][ret['results']['extrapolated_3d']], '^c', label='prediction extrapolated')
    ah[2][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['w_pred'][ret['results']['in_bounds_3d']], 'ob', label='prediction')

    ah[0][0].set_ylabel('U [m/s]')
    ah[1][0].set_ylabel('V [m/s]')
    ah[2][0].set_ylabel('W [m/s]')

    ah[0][0].axes.xaxis.set_visible(False)
    ah[1][0].axes.xaxis.set_visible(False)

    ah[0][0].legend()

    if ret['turb_predicted']:
        ah[3][0].plot(x_3d, ret['results']['tke_meas'], 'sr', label='measurements')
        ah[3][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['tke_pred'][ret['results']['extrapolated_3d']], '^c', label='prediction extrapolated')
        ah[3][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['tke_pred'][ret['results']['in_bounds_3d']], 'ob', label='prediction')
        ah[3][0].set_ylabel('TKE [m^2/s^2]')

        ah[2][0].axes.xaxis.set_visible(False)
        ah[3][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[3][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    else:
        ah[2][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[2][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    fig, ah = plt.subplots(num_plots, 1, squeeze=False)
    ah[0][0].set_title('Prediction Errors')
    ah[0][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['u_meas'][ret['results']['extrapolated_3d']] - ret['results']['u_pred'][ret['results']['extrapolated_3d']], '^c', label='extrapolated')
    ah[1][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['v_meas'][ret['results']['extrapolated_3d']] - ret['results']['v_pred'][ret['results']['extrapolated_3d']], '^c', label='extrapolated')
    ah[2][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['w_meas'][ret['results']['extrapolated_3d']] - ret['results']['w_pred'][ret['results']['extrapolated_3d']], '^c', label='extrapolated')

    ah[0][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['u_meas'][ret['results']['in_bounds_3d']] - ret['results']['u_pred'][ret['results']['in_bounds_3d']], 'ob', label='in bounds')
    ah[1][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['v_meas'][ret['results']['in_bounds_3d']] - ret['results']['v_pred'][ret['results']['in_bounds_3d']], 'ob', label='in bounds')
    ah[2][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['w_meas'][ret['results']['in_bounds_3d']] - ret['results']['w_pred'][ret['results']['in_bounds_3d']], 'ob', label='in bounds')

    ah[0][0].set_ylabel('Error U [m/s]')
    ah[1][0].set_ylabel('Error V [m/s]')
    ah[2][0].set_ylabel('Error W [m/s]')

    ah[0][0].axes.xaxis.set_visible(False)
    ah[1][0].axes.xaxis.set_visible(False)

    ah[0][0].legend()

    if ret['turb_predicted']:
        ah[3][0].plot(x_3d[ret['results']['extrapolated_3d']], ret['results']['tke_meas'][ret['results']['extrapolated_3d']] - ret['results']['tke_pred'][ret['results']['extrapolated_3d']], '^c', label='extrapolated')
        ah[3][0].plot(x_3d[ret['results']['in_bounds_3d']], ret['results']['tke_meas'][ret['results']['in_bounds_3d']] - ret['results']['tke_pred'][ret['results']['in_bounds_3d']], 'ob', label='in bounds')
        ah[3][0].set_ylabel('Error TKE [m^2/s^2]')

        ah[2][0].axes.xaxis.set_visible(False)
        ah[3][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[3][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    else:
        ah[2][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[2][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    fig, ah = plt.subplots(num_plots, 1, squeeze=False)
    ah[0][0].set_title('Relative Prediction Errors')
    ah[0][0].plot(x_3d[ret['results']['extrapolated_3d']], (ret['results']['u_meas'][ret['results']['extrapolated_3d']] - ret['results']['u_pred'][ret['results']['extrapolated_3d']]) / np.abs(ret['results']['u_meas'][ret['results']['extrapolated_3d']]), '^c', label='extrapolated')
    ah[1][0].plot(x_3d[ret['results']['extrapolated_3d']], (ret['results']['v_meas'][ret['results']['extrapolated_3d']] - ret['results']['v_pred'][ret['results']['extrapolated_3d']]) / np.abs(ret['results']['v_meas'][ret['results']['extrapolated_3d']]), '^c', label='extrapolated')
    ah[2][0].plot(x_3d[ret['results']['extrapolated_3d']], (ret['results']['w_meas'][ret['results']['extrapolated_3d']] - ret['results']['w_pred'][ret['results']['extrapolated_3d']]) / np.abs(ret['results']['w_meas'][ret['results']['extrapolated_3d']]), '^c', label='extrapolated')

    ah[0][0].plot(x_3d[ret['results']['in_bounds_3d']], (ret['results']['u_meas'][ret['results']['in_bounds_3d']] - ret['results']['u_pred'][ret['results']['in_bounds_3d']]) / np.abs(ret['results']['u_meas'][ret['results']['in_bounds_3d']]), 'ob', label='in bounds')
    ah[1][0].plot(x_3d[ret['results']['in_bounds_3d']], (ret['results']['v_meas'][ret['results']['in_bounds_3d']] - ret['results']['v_pred'][ret['results']['in_bounds_3d']]) / np.abs(ret['results']['v_meas'][ret['results']['in_bounds_3d']]), 'ob', label='in bounds')
    ah[2][0].plot(x_3d[ret['results']['in_bounds_3d']], (ret['results']['w_meas'][ret['results']['in_bounds_3d']] - ret['results']['w_pred'][ret['results']['in_bounds_3d']]) / np.abs(ret['results']['w_meas'][ret['results']['in_bounds_3d']]), 'ob', label='in bounds')

    ah[0][0].set_ylabel('Error Rel U [-]')
    ah[1][0].set_ylabel('Error Rel V [-]')
    ah[2][0].set_ylabel('Error Rel W [-]')

    ah[0][0].axes.xaxis.set_visible(False)
    ah[1][0].axes.xaxis.set_visible(False)

    ah[0][0].legend()

    if ret['turb_predicted']:
        ah[3][0].plot(x_3d[ret['results']['extrapolated_3d']], (ret['results']['tke_meas'][ret['results']['extrapolated_3d']] - ret['results']['tke_pred'][ret['results']['extrapolated_3d']]) / np.abs(ret['results']['tke_meas'][ret['results']['extrapolated_3d']]), '^c', label='extrapolated')
        ah[3][0].plot(x_3d[ret['results']['in_bounds_3d']], (ret['results']['tke_meas'][ret['results']['in_bounds_3d']] - ret['results']['tke_pred'][ret['results']['in_bounds_3d']]) / np.abs(ret['results']['tke_meas'][ret['results']['in_bounds_3d']]), 'ob', label='in bounds')
        ah[3][0].set_ylabel('Error Rel TKE [-]')

        ah[2][0].axes.xaxis.set_visible(False)
        ah[3][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[3][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    else:
        ah[2][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[2][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')


    fig, ah = plt.subplots(1, 1, squeeze=False)
    ah[0][0].set_title('Prediction vs Measurement Magnitude')
    ah[0][0].plot(x_all, ret['results']['s_meas'], 'sr', label='measurements')
    ah[0][0].plot(x_all[ret['results']['extrapolated']], ret['results']['s_pred'][ret['results']['extrapolated']], '^c', label='prediction extrapolated')
    ah[0][0].plot(x_all[ret['results']['in_bounds']], ret['results']['s_pred'][ret['results']['in_bounds']], 'ob', label='prediction')

    ah[0][0].set_ylabel('Magnitude [m/s]')

    ah[0][0].set_xticks(np.arange(len(ret['results']['labels_all'])))
    ah[0][0].set_xticklabels(ret['results']['labels_all'], rotation='vertical')
    ah[0][0].legend()

    fig, ah = plt.subplots(1, 1, squeeze=False)
    ah[0][0].set_title('Prediction Magnitude Error')
    ah[0][0].plot(x_all[ret['results']['extrapolated']], ret['results']['s_meas'][ret['results']['extrapolated']] - ret['results']['s_pred'][ret['results']['extrapolated']], '^c', label='extrapolated')
    ah[0][0].plot(x_all[ret['results']['in_bounds']], ret['results']['s_meas'][ret['results']['in_bounds']] - ret['results']['s_pred'][ret['results']['in_bounds']], 'ob', label='in bounds')

    ah[0][0].set_ylabel('Error Magnitude [m/s]')

    ah[0][0].set_xticks(np.arange(len(ret['results']['labels_all'])))
    ah[0][0].set_xticklabels(ret['results']['labels_all'], rotation='vertical')
    ah[0][0].legend()

    # Display results
    print('---------------------------------------------------')
    print('Prediction using mast: ' + args.input_mast)
    print('Error U: ' + str(np.nanmean(np.abs(ret['results']['u_meas'] - ret['results']['u_pred']))))
    print('Error V: ' + str(np.nanmean(np.abs(ret['results']['v_meas'] - ret['results']['v_pred']))))
    print('Error W: ' + str(np.nanmean(np.abs(ret['results']['w_meas'] - ret['results']['w_pred']))))
    if ret['turb_predicted']:
        print('Error TKE: ' + str(np.nanmean(np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred']))))
    print('Error S: ' + str(np.nanmean(np.abs(ret['results']['s_meas'] - ret['results']['s_pred']))))
    print('Error U rel: ' + str(np.nanmean(np.abs(ret['results']['u_meas'] - ret['results']['u_pred'])/np.abs(ret['results']['u_meas']))))
    print('Error V rel: ' + str(np.nanmean(np.abs(ret['results']['v_meas'] - ret['results']['v_pred'])/np.abs(ret['results']['v_meas']))))
    print('Error W rel: ' + str(np.nanmean(np.abs(ret['results']['w_meas'] - ret['results']['w_pred'])/np.abs(ret['results']['w_meas']))))
    if ret['turb_predicted']:
        print('Error TKE rel: ' + str(np.nanmean(np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred'])/np.abs(ret['results']['tke_meas']))))
    print('Error S rel: ' + str(np.nanmean(np.abs(ret['results']['s_meas'] - ret['results']['s_pred'])/np.abs(ret['results']['s_meas']))))
    print('---------------------------------------------------')

    if args.profile:
        for line_key in ret['profiles'].keys():
            plt.figure()
            plt.plot(ret['profiles'][line_key]['dist'], ret['profiles'][line_key]['speedup'])
            plt.xlabel('Dist [cells]')
            plt.ylabel('Speedup [-]')
            plt.xlim([-100, 150])
            plt.ylim([-1, 1])
            plt.title(line_key)

    if args.mayavi:
        ui = []

        ui.append(
            nn_utils.mlab_plot_prediction(ret['prediction']['pred'], ret['terrain'], terrain_mode='blocks', terrain_uniform_color=True,
                                          prediction_channels=config.data['label_channels'], blocking=False))


    nn_utils.plot_prediction(config.data['label_channels'],
                             prediction = ret['prediction']['pred'][0].cpu().detach(),
                             provided_input_channels = config.data['input_channels'],
                             input = ret['input'][0].cpu().detach(),
                             terrain = ret['terrain'].cpu().squeeze())
