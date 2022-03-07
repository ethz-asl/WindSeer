import argparse
import copy
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import h5py
import random
from scipy.interpolate import RegularGridInterpolator, griddata
import torch

import nn_wind_prediction.models as models
import nn_wind_prediction.utils as nn_utils
from analysis_utils import utils

try:
    import GPy
    gpy_available = True
except ImportError:
    print('GPy could not get imported, GPR baseline not available')
    gpy_available = False

def masts_to_string(input):
    out = ''

    for ele in input:
        out += str(ele)
        out += ' '

    return out

def predict_case(dataset, net, index, input_mast, experiment_name, config, compute_baseline, speedup_profiles=False, reference_mast=None, predict_lidar=False, gpr_baseline=False):
    # load the data
    h5_file = h5py.File(dataset, 'r')
    scale_keys = list(h5_file['terrain'].keys())
    if index >= len(scale_keys):
        print("Requested index out of bounds, available scale keys:")
        print(scale_keys)
        exit()

    if not experiment_name in h5_file['measurement'].keys():
        print('Requested experiment not present in the dataset. Available experiments:')
        print(list(h5_file['measurement'].keys()))
        exit()

    for mast in input_mast:
        if not mast in h5_file['measurement'][args.experiment][scale_keys[args.index]].keys():
            print('Requested input mast not present in the dataset. Available experiments:')
            print(list(h5_file['measurement'][args.experiment][scale_keys[args.index]].keys()))
            exit()

    print('Input data:', experiment_name, '| scale:', scale_keys[index], '| mast:', masts_to_string(input_mast))

    terrain = torch.from_numpy(h5_file['terrain'][scale_keys[index]][...])
    input_meas = torch.zeros(tuple([3]) + tuple(terrain.shape))
    input_mask = torch.zeros(tuple(terrain.shape))
    u_in = []
    v_in = []
    w_in = []
    pos_in = []
    print('Measurement indices:')
    for mast in input_mast:
        ds_input = h5_file['measurement'][experiment_name][scale_keys[args.index]][mast]
        # shuffle measurements according to height making sure that we fill always the highest value into the respective cell
        positions = ds_input['pos'][...]
        u_vel = ds_input['u'][...]
        v_vel = ds_input['v'][...]
        w_vel = ds_input['w'][...]
        u_in.extend(ds_input['u'][...])
        v_in.extend(ds_input['v'][...])
        w_in.extend(ds_input['w'][...])
        pos_in.append(positions)
        tke_available = 'tke' in ds_input.keys()

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
                    input_meas[0,meas_idx[2], meas_idx[1], meas_idx[0]] = u.item()
                    input_meas[1,meas_idx[2], meas_idx[1], meas_idx[0]] = v.item()
                    input_meas[2,meas_idx[2], meas_idx[1], meas_idx[0]] = w.item()

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

    if compute_baseline:
        # clean up data (remove invalid input samples
        scale = float(scale_keys[args.index].split('_')[1])
        u_in = np.array(u_in)
        v_in = np.array(v_in)
        w_in = np.array(w_in)
        tot_vel = u_in**2 + v_in**2 + w_in**2
        u_in = u_in[tot_vel>0]
        v_in = v_in[tot_vel>0]
        w_in = w_in[tot_vel>0]

        pos_in_gp = None
        for pos in pos_in:
            if pos_in_gp is None:
                pos_in_gp = pos
            else:
                pos_in_gp = np.concatenate((pos_in_gp, pos))

        pos_in_gp = pos_in_gp[tot_vel>0] / scale

        num_channels = 3
        if config.model['model_args']['use_turbulence']:
            num_channels = 4

        if gpr_baseline and not gpy_available:
            print('GPR baseline requested but GPy not available')

        if gpr_baseline and gpy_available:
            kernel_u = GPy.kern.RBF(input_dim=3, variance=47.625, lengthscale=6.716)
            gpm_u = GPy.models.GPRegression(pos_in_gp, u_in[:,np.newaxis]-u_in.mean(), kernel_u)
            kernel_v = GPy.kern.RBF(input_dim=3, variance=47.625, lengthscale=6.716)
            gpm_v = GPy.models.GPRegression(pos_in_gp, v_in[:,np.newaxis]-v_in.mean(), kernel_v)
            kernel_w = GPy.kern.RBF(input_dim=3, variance=31.576, lengthscale=6.905)
            gpm_w = GPy.models.GPRegression(pos_in_gp, w_in[:,np.newaxis]-w_in.mean(), kernel_w)

            Z, Y, X = torch.meshgrid(torch.linspace(0, terrain.shape[0]-1, terrain.shape[0]), torch.linspace(0, terrain.shape[1]-1, terrain.shape[1]), torch.linspace(0, terrain.shape[2]-1, terrain.shape[2]))

            grid_pos = torch.stack((X.ravel(), Y.ravel(), Z.ravel())).T

            prediction = {'pred': torch.zeros(tuple([1, num_channels]) + tuple(terrain.shape))}
            prediction['pred'][0,0] = torch.from_numpy(gpm_u.predict(grid_pos.numpy() / scale)[0].reshape(X.shape))+u_in.mean()
            prediction['pred'][0,1] = torch.from_numpy(gpm_v.predict(grid_pos.numpy() / scale)[0].reshape(X.shape))+v_in.mean()
            prediction['pred'][0,2] = torch.from_numpy(gpm_w.predict(grid_pos.numpy() / scale)[0].reshape(X.shape))+w_in.mean()

        else:
            u_mean = np.mean(u_in)
            v_mean = np.mean(v_in)
            w_mean = np.mean(w_in)

            prediction = {'pred': torch.ones(tuple([1, num_channels]) + tuple(terrain.shape))}
            prediction['pred'][0,0] *= u_mean
            prediction['pred'][0,1] *= v_mean
            prediction['pred'][0,2] *= w_mean
            if config.model['model_args']['use_turbulence']:
                prediction['pred'][0,3] *= 0

    else:
        with torch.no_grad():
            prediction = utils.predict(net, input.clone(), None, config.data)

    # compute the error of the prediction compared to the measurements
    nz, ny, nx = prediction['pred'].shape[2:]

    u_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,0].cpu().detach().numpy(), bounds_error=False, fill_value=np.nan)
    v_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,1].cpu().detach().numpy(), bounds_error=False, fill_value=np.nan)
    w_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,2].cpu().detach().numpy(), bounds_error=False, fill_value=np.nan)

    if config.model['model_args']['use_turbulence']:
        tke_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), prediction['pred'][0,3].cpu().detach().numpy(), bounds_error=False, fill_value=np.nan)
        turb_predicted = tke_available
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
        if tke_available:
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
        if tke_available:
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
        'turb_predicted': turb_predicted and tke_available,
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

    if predict_lidar:
        ds_lidar = h5_file['lidar']
        ret['lidar'] = {}
        if experiment_name in ds_lidar.keys():
            ds_lidar_case = ds_lidar[experiment_name][scale_keys[args.index]]
            for scan in ds_lidar_case.keys():
                ret['lidar'][scan] = {}

                x = ds_lidar_case[scan]['x'][...]
                y = ds_lidar_case[scan]['y'][...]
                z = ds_lidar_case[scan]['z'][...]
                vel = ds_lidar_case[scan]['vel'][...]
                azimuth_angle = ds_lidar_case[scan]['azimuth_angle'][...]
                elevation_angle = ds_lidar_case[scan]['elevation_angle'][...]

                positions = copy.copy(np.stack((x.ravel(),y.ravel(),z.ravel()), axis=1))
                u_pred = u_interpolator(np.fliplr(positions))
                v_pred = v_interpolator(np.fliplr(positions))
                w_pred = w_interpolator(np.fliplr(positions))

                dir_vec_x = np.repeat((np.sin(azimuth_angle) * np.cos(elevation_angle))[:, np.newaxis], x.shape[1], axis=1).ravel()
                dir_vec_y = np.repeat((np.cos(azimuth_angle) * np.cos(elevation_angle))[:, np.newaxis], x.shape[1], axis=1).ravel()
                dir_vec_z = np.repeat(np.sin(elevation_angle)[:, np.newaxis], x.shape[1], axis=1).ravel()

                vel_projected = (u_pred * dir_vec_x + v_pred * dir_vec_y + w_pred * dir_vec_z)
                ret['lidar'][scan]['x'] = x.ravel()
                ret['lidar'][scan]['y'] = y.ravel()
                ret['lidar'][scan]['z'] = z.ravel()
                ret['lidar'][scan]['pred'] = vel_projected
                ret['lidar'][scan]['meas'] = vel.ravel()

                terrain_interpolator = RegularGridInterpolator((np.linspace(0,nz-1,nz),np.linspace(0,ny-1,ny),np.linspace(0,nx-1,nx)), terrain.cpu().detach().numpy(), bounds_error=False, fill_value=None, method='nearest')
                ret['lidar'][scan]['terrain'] = terrain_interpolator(np.fliplr(positions))

                if tke_interpolator:
                    ret['lidar'][scan]['tke'] = tke_interpolator(np.fliplr(positions))

        else:
            print('No Lidar scans available for this timestamp. Available scans:')
            print(ds_lidar.keys())

    return ret

parser = argparse.ArgumentParser(description='Predict the flow based on the sparse measurements')
parser.add_argument('-d', dest='dataset', required=True, help='The dataset file')
parser.add_argument('-m', dest='input_mast', required=True, nargs='+', help='The input measurement mast')
parser.add_argument('-e', dest='experiment', required=True, help='The experiment name')
parser.add_argument('-i', dest='index', default=0, type=int, help='Index of the case in the dataset used for the prediction')
parser.add_argument('-model_dir', dest='model_dir', required=True, help='The directory of the model')
parser.add_argument('-model_version', dest='model_version', default='latest', help='The model version')
parser.add_argument('-reference_mast', help='The reference mast (used for the speedup calculation)')
parser.add_argument('--mayavi', action='store_true', help='Generate some extra plots using mayavi')
parser.add_argument('--no_gpu', action='store_true', help='Force CPU prediction')
parser.add_argument('--benchmark', action='store_true', help='Benchmark the prediction by looping through all input masks')
parser.add_argument('--profile', action='store_true', help='Compute the speedup profiles along lines')
parser.add_argument('--lidar', action='store_true', help='Compute the velocities along the lidar planes')
parser.add_argument('--baseline', action='store_true', help='Compute the baseline (average of measurements / GPR)')
parser.add_argument('--gpr', action='store_true', help='Compute the baseline using GPR instead of averaging')

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
        ret = predict_case(args.dataset, net, args.index, [mast], args.experiment, config, args.baseline, gpr_baseline=args.gpr)
        if ret:
            results[mast] = ret['results']
            turbulence_predicted = ret['turb_predicted']
        else:
            print('Input mast outside the prediction region')

    all_measurements = {}
    for key in results.keys():
        print('---------------------------------------------------')
        print('Prediction using mast: ' + key)
        print('Error U: ' + str(np.nanmean(np.abs(results[key]['u_meas'] - results[key]['u_pred']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['u_meas'] - results[key]['u_pred']))))
        print('Error V: ' + str(np.nanmean(np.abs(results[key]['v_meas'] - results[key]['v_pred']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['v_meas'] - results[key]['v_pred']))))
        print('Error W: ' + str(np.nanmean(np.abs(results[key]['w_meas'] - results[key]['w_pred']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['w_meas'] - results[key]['w_pred']))))
        if turbulence_predicted:
            print('Error TKE: ' + str(np.nanmean(np.abs(results[key]['tke_meas'] - results[key]['tke_pred'])))  + ' +- ' + str(np.nanstd(np.abs(results[key]['tke_meas'] - results[key]['tke_pred']))))
        print('Error S: ' + str(np.nanmean(np.abs(results[key]['s_meas'] - results[key]['s_pred']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['s_meas'] - results[key]['s_pred']))))
        print('Error U rel: ' + str(np.nanmean(np.abs(results[key]['u_meas'] - results[key]['u_pred'])/np.abs(results[key]['u_meas']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['u_meas'] - results[key]['u_pred'])/np.abs(results[key]['u_meas']))))
        print('Error V rel: ' + str(np.nanmean(np.abs(results[key]['v_meas'] - results[key]['v_pred'])/np.abs(results[key]['v_meas']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['v_meas'] - results[key]['v_pred'])/np.abs(results[key]['v_meas']))))
        print('Error W rel: ' + str(np.nanmean(np.abs(results[key]['w_meas'] - results[key]['w_pred'])/np.abs(results[key]['w_meas']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['w_meas'] - results[key]['w_pred'])/np.abs(results[key]['w_meas']))))
        if turbulence_predicted:
            print('Error TKE rel: ' + str(np.nanmean(np.abs(results[key]['tke_meas'] - results[key]['tke_pred'])/np.abs(results[key]['tke_meas']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['tke_meas'] - results[key]['tke_pred'])/np.abs(results[key]['tke_meas']))))
        print('Error S rel: ' + str(np.nanmean(np.abs(results[key]['s_meas'] - results[key]['s_pred'])/np.abs(results[key]['s_meas']))) + ' +- ' + str(np.nanstd(np.abs(results[key]['s_meas'] - results[key]['s_pred'])/np.abs(results[key]['s_meas']))))

        for field in results[key].keys():
            if 'meas' in field or 'pred' in field:
                if field in all_measurements.keys():
                    all_measurements[field] = np.concatenate((all_measurements[field],results[key][field]))
                else:
                    all_measurements[field] = results[key][field]
                x = np.array([])

    print('---------------------------------------------------')
    print('Prediction error all masts')
    print('Error U: ' + str(np.nanmean(np.abs(all_measurements['u_meas'] - all_measurements['u_pred']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['u_meas'] - all_measurements['u_pred']))))
    print('Error V: ' + str(np.nanmean(np.abs(all_measurements['v_meas'] - all_measurements['v_pred']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['v_meas'] - all_measurements['v_pred']))))
    print('Error W: ' + str(np.nanmean(np.abs(all_measurements['w_meas'] - all_measurements['w_pred']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['w_meas'] - all_measurements['w_pred']))))
    if turbulence_predicted:
        print('Error TKE: ' + str(np.nanmean(np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred']))))
    print('Error S: ' + str(np.nanmean(np.abs(all_measurements['s_meas'] - all_measurements['s_pred']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['s_meas'] - all_measurements['s_pred']))))
    print('Error U rel: ' + str(np.nanmean(np.abs(all_measurements['u_meas'] - all_measurements['u_pred'])/np.abs(all_measurements['u_meas']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['u_meas'] - all_measurements['u_pred'])/np.abs(all_measurements['u_meas']))))
    print('Error V rel: ' + str(np.nanmean(np.abs(all_measurements['v_meas'] - all_measurements['v_pred'])/np.abs(all_measurements['v_meas']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['v_meas'] - all_measurements['v_pred'])/np.abs(all_measurements['v_meas']))))
    print('Error W rel: ' + str(np.nanmean(np.abs(all_measurements['w_meas'] - all_measurements['w_pred'])/np.abs(all_measurements['w_meas']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['w_meas'] - all_measurements['w_pred'])/np.abs(all_measurements['w_meas']))))
    if turbulence_predicted:
        print('Error TKE rel: ' + str(np.nanmean(np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred'])/np.abs(all_measurements['tke_meas']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred'])/np.abs(all_measurements['tke_meas']))))
    print('Error S rel: ' + str(np.nanmean(np.abs(all_measurements['s_meas'] - all_measurements['s_pred'])/np.abs(all_measurements['s_meas']))) + ' +- ' + str(np.nanstd(np.abs(all_measurements['s_meas'] - all_measurements['s_pred'])/np.abs(all_measurements['s_meas']))))
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

    if turbulence_predicted:
        plt.figure()
        for key in results.keys():
            plt.plot(results[key]['tke_meas'], results[key]['tke_pred'], 'o', label=key)
        plt.xlabel('Measurement')
        plt.ylabel('Prediction')
        plt.legend()
        plt.title('TKE')


    plt.show()

else:
    ret = predict_case(args.dataset, net, args.index, args.input_mast, args.experiment, config, args.baseline, args.profile, args.reference_mast, args.lidar, args.gpr)
    
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
    print('Prediction using mast: ' + masts_to_string(args.input_mast))
    print('Error U: ' + str(np.nanmean(np.abs(ret['results']['u_meas'] - ret['results']['u_pred']))), '+-', str(np.nanstd(np.abs(ret['results']['u_meas'] - ret['results']['u_pred']))))
    print('Error V: ' + str(np.nanmean(np.abs(ret['results']['v_meas'] - ret['results']['v_pred']))), '+-', str(np.nanstd(np.abs(ret['results']['v_meas'] - ret['results']['v_pred']))))
    print('Error W: ' + str(np.nanmean(np.abs(ret['results']['w_meas'] - ret['results']['w_pred']))), '+-', str(np.nanstd(np.abs(ret['results']['w_meas'] - ret['results']['w_pred']))))
    if ret['turb_predicted']:
        print('Error TKE: ' + str(np.nanmean(np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred']))), '+-', str(np.nanstd(np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred']))))
    print('Error S: ' + str(np.nanmean(np.abs(ret['results']['s_meas'] - ret['results']['s_pred']))), '+-', str(np.nanstd(np.abs(ret['results']['s_meas'] - ret['results']['s_pred']))))
    print('Error U rel: ' + str(np.nanmean(np.abs(ret['results']['u_meas'] - ret['results']['u_pred'])/np.abs(ret['results']['u_meas']))), '+-', str(np.nanstd(np.abs(ret['results']['u_meas'] - ret['results']['u_pred'])/np.abs(ret['results']['u_meas']))))
    print('Error V rel: ' + str(np.nanmean(np.abs(ret['results']['v_meas'] - ret['results']['v_pred'])/np.abs(ret['results']['v_meas']))), '+-', str(np.nanstd(np.abs(ret['results']['v_meas'] - ret['results']['v_pred'])/np.abs(ret['results']['v_meas']))))
    print('Error W rel: ' + str(np.nanmean(np.abs(ret['results']['w_meas'] - ret['results']['w_pred'])/np.abs(ret['results']['w_meas']))), '+-', str(np.nanstd(np.abs(ret['results']['w_meas'] - ret['results']['w_pred'])/np.abs(ret['results']['w_meas']))))
    if ret['turb_predicted']:
        print('Error TKE rel: ' + str(np.nanmean(np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred'])/np.abs(ret['results']['tke_meas']))), '+-', str(np.nanstd(np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred'])/np.abs(ret['results']['tke_meas']))))
    print('Error S rel: ' + str(np.nanmean(np.abs(ret['results']['s_meas'] - ret['results']['s_pred'])/np.abs(ret['results']['s_meas']))), '+-', str(np.nanstd(np.abs(ret['results']['s_meas'] - ret['results']['s_pred'])/np.abs(ret['results']['s_meas']))))
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

    if args.lidar:
        for lidar_key in ret['lidar'].keys():
            resolution = 4.0
            x_local = ret['lidar'][lidar_key]['x'] - ret['lidar'][lidar_key]['x'].min()
            y_local = ret['lidar'][lidar_key]['y'] - ret['lidar'][lidar_key]['y'].min()
            z = ret['lidar'][lidar_key]['z']
            dist = np.sqrt(x_local**2 + y_local**2)
            dist_resampled = np.linspace(0, int(dist.max()), int(int(dist.max()+1) / resolution))
            min_z = np.max((z.min(), 0))
            max_z = np.min((z.max(), ret['prediction']['pred'].shape[2]))
            z_resampled =  np.linspace(min_z, max_z, int(int(max_z-min_z+1) / resolution))

            D_grid, Z_grid = np.meshgrid(dist_resampled, z_resampled, indexing='xy')
            measured = griddata((dist, z), ret['lidar'][lidar_key]['meas'], (D_grid, Z_grid), method='linear')
            predicted = griddata((dist, z), ret['lidar'][lidar_key]['pred'], (D_grid, Z_grid), method='linear')

            cmap_terrain = colors.LinearSegmentedColormap.from_list('custom', colors.to_rgba_array(['grey', 'grey']), 2)
            terrain = griddata((dist, z), ret['lidar'][lidar_key]['terrain'], (D_grid, Z_grid), method='nearest')
            no_terrain = np.logical_not(np.logical_not(terrain.astype(bool)))
            terrain_mask = np.ma.masked_where(no_terrain, no_terrain)

            min_val = np.nanmin((np.nanmin(measured), np.nanmin(predicted)))
            max_val = np.nanmax((np.nanmax(measured), np.nanmax(predicted)))

            fig, ah = plt.subplots(2, 2, squeeze=False)
            ah[0][0].set_title('Prediction')
            im = ah[0][0].imshow(predicted, origin='lower', vmin=min_val, vmax=max_val, cmap=cm.jet)
            fig.colorbar(im, ax=ah[0][0])
            ah[0][0].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

            ah[1][0].set_title('Measurement')
            im = ah[1][0].imshow(measured, origin='lower', vmin=min_val, vmax=max_val, cmap=cm.jet)
            fig.colorbar(im, ax=ah[1][0])
            ah[1][0].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

            ah[0][1].set_title('Error')
            im = ah[0][1].imshow(measured - predicted, origin='lower', cmap=cm.jet)
            fig.colorbar(im, ax=ah[0][1])
            ah[0][1].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

            if 'tke' in ret['lidar'][lidar_key].keys():
                tke = griddata((dist, z), ret['lidar'][lidar_key]['tke'], (D_grid, Z_grid), method='linear')
                ah[1][1].set_title('TKE predicted')
                ah[1][1].imshow(tke, origin='lower', cmap=cm.jet)
                ah[1][1].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

    if args.mayavi:
        ui = []

        ui.append(
            nn_utils.mlab_plot_prediction(ret['prediction']['pred'], ret['terrain'], terrain_mode='blocks', terrain_uniform_color=False,
                                          prediction_channels=config.data['label_channels'], blocking=False))

        nn_utils.mlab_plot_measurements(ret['input'][0, 1:-1], ret['input'][0, -1], ret['terrain'], terrain_mode='blocks',
                                     terrain_uniform_color=False, blocking=False)


    nn_utils.plot_prediction(config.data['label_channels'],
                             prediction = ret['prediction']['pred'][0].cpu().detach(),
                             provided_input_channels = config.data['input_channels'],
                             input = ret['input'][0].cpu().detach(),
                             terrain = ret['terrain'].cpu().squeeze())
