import copy
import h5py
from itertools import compress
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

try:
    import GPy
    gpy_available = True
except ImportError:
    print('GPy could not get imported, GPR baseline not available')
    gpy_available = False

import windseer.data as data
import windseer.nn as nn
import windseer.utils as utils


def masts_to_string(input):
    out = ''

    for ele in input:
        out += str(ele)
        out += ' '

    return out


def predict_case(
        dataset,
        net,
        device,
        index,
        input_mast,
        experiment_name,
        config,
        compute_baseline,
        speedup_profiles=False,
        reference_mast=None,
        predict_lidar=False,
        gpr_baseline=False,
        extrapolate_pred=False
    ):
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
        if not mast in h5_file['measurement'][experiment_name][scale_keys[index]
                                                               ].keys():
            print(
                'Requested input mast not present in the dataset. Available experiments:'
                )
            print(
                list(h5_file['measurement'][experiment_name][scale_keys[index]].keys())
                )
            exit()

    print(
        'Input data:', experiment_name, '| scale:', scale_keys[index], '| mast:',
        masts_to_string(input_mast)
        )

    terrain = torch.from_numpy(h5_file['terrain'][scale_keys[index]][...])
    input_meas = torch.zeros(tuple([3]) + tuple(terrain.shape))
    input_mask = torch.zeros(tuple(terrain.shape))
    u_in = []
    v_in = []
    w_in = []
    pos_in = []
    print('Measurement indices:')
    for mast in input_mast:
        ds_input = h5_file['measurement'][experiment_name][scale_keys[index]][mast]
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

        idx_shuffle = np.argsort(positions[:, 2])
        for idx in idx_shuffle:
            u = u_vel[idx]
            v = v_vel[idx]
            w = w_vel[idx]

            if (u * u + v * v + w + w) > 0.0:
                meas_idx = np.round(positions[idx]).astype(np.int)
                print(meas_idx)

                if meas_idx.min(
                ) >= 0 and meas_idx[0] < input_mask.shape[2] and meas_idx[
                    1] < input_mask.shape[1] and meas_idx[2] < input_mask.shape[0]:
                    input_mask[meas_idx[2], meas_idx[1], meas_idx[0]] = 1
                    input_meas[0, meas_idx[2], meas_idx[1], meas_idx[0]] = u.item()
                    input_meas[1, meas_idx[2], meas_idx[1], meas_idx[0]] = v.item()
                    input_meas[2, meas_idx[2], meas_idx[1], meas_idx[0]] = w.item()

    # remove any samples inside the terrain
    input_mask *= terrain > 0
    input_meas *= terrain > 0

    input_meas = input_meas.to(device)
    input_mask = input_mask.to(device)
    terrain = terrain.to(device)

    if input_mask.sum() == 0:
        print(
            'All measurements lie inside the terrain or outside of the prediction domain'
            )
        return None

    # fill the holes in the input if requested
    input_measurement = input_meas
    if 'input_smoothing' in config.data.keys():
        if config.data['input_smoothing']:
            input_measurement = data.get_smooth_data(
                input_meas, input_mask.bool(), config.model['model_args']['grid_size'],
                config.data['input_smoothing_interpolation'],
                config.data['input_smoothing_interpolation_linear']
                )

    input_idx = []
    if 'ux' in config.data['input_channels']:
        input_idx.append(0)
    if 'uy' in config.data['input_channels']:
        input_idx.append(1)
    if 'uz' in config.data['input_channels']:
        input_idx.append(2)

    input = torch.cat([
        terrain.unsqueeze(0).unsqueeze(0), input_measurement[input_idx].unsqueeze(0),
        input_mask.unsqueeze(0).unsqueeze(0)
        ],
                      dim=1)

    if compute_baseline:
        # clean up data (remove invalid input samples
        scale = float(scale_keys[index].split('_')[1])
        u_in = np.array(u_in)
        v_in = np.array(v_in)
        w_in = np.array(w_in)
        tot_vel = u_in**2 + v_in**2 + w_in**2
        u_in = u_in[tot_vel > 0]
        v_in = v_in[tot_vel > 0]
        w_in = w_in[tot_vel > 0]

        pos_in_gp = None
        for pos in pos_in:
            if pos_in_gp is None:
                pos_in_gp = pos
            else:
                pos_in_gp = np.concatenate((pos_in_gp, pos))

        pos_in_gp = pos_in_gp[tot_vel > 0] / scale

        num_channels = 3
        if config.model['model_args']['use_turbulence']:
            num_channels = 4

        if gpr_baseline and not gpy_available:
            print('GPR baseline requested but GPy not available')

        if gpr_baseline and gpy_available:
            kernel_u = GPy.kern.RBF(input_dim=3, variance=47.625, lengthscale=6.716)
            gpm_u = GPy.models.GPRegression(
                pos_in_gp, u_in[:, np.newaxis] - u_in.mean(), kernel_u
                )
            kernel_v = GPy.kern.RBF(input_dim=3, variance=47.625, lengthscale=6.716)
            gpm_v = GPy.models.GPRegression(
                pos_in_gp, v_in[:, np.newaxis] - v_in.mean(), kernel_v
                )
            kernel_w = GPy.kern.RBF(input_dim=3, variance=31.576, lengthscale=6.905)
            gpm_w = GPy.models.GPRegression(
                pos_in_gp, w_in[:, np.newaxis] - w_in.mean(), kernel_w
                )

            Z, Y, X = torch.meshgrid(
                torch.linspace(0, terrain.shape[0] - 1, terrain.shape[0]),
                torch.linspace(0, terrain.shape[1] - 1, terrain.shape[1]),
                torch.linspace(0, terrain.shape[2] - 1, terrain.shape[2])
                )

            grid_pos = torch.stack((X.ravel(), Y.ravel(), Z.ravel())).T

            prediction = {
                'pred': torch.zeros(tuple([1, num_channels]) + tuple(terrain.shape))
                }
            prediction['pred'][0, 0] = torch.from_numpy(
                gpm_u.predict(grid_pos.numpy() / scale)[0].reshape(X.shape)
                ) + u_in.mean()
            prediction['pred'][0, 1] = torch.from_numpy(
                gpm_v.predict(grid_pos.numpy() / scale)[0].reshape(X.shape)
                ) + v_in.mean()
            prediction['pred'][0, 2] = torch.from_numpy(
                gpm_w.predict(grid_pos.numpy() / scale)[0].reshape(X.shape)
                ) + w_in.mean()

        else:
            u_mean = np.mean(u_in)
            v_mean = np.mean(v_in)
            w_mean = np.mean(w_in)

            prediction = {
                'pred': torch.ones(tuple([1, num_channels]) + tuple(terrain.shape))
                }
            prediction['pred'][0, 0] *= u_mean
            prediction['pred'][0, 1] *= v_mean
            prediction['pred'][0, 2] *= w_mean
            if config.model['model_args']['use_turbulence']:
                prediction['pred'][0, 3] *= 0

    else:
        with torch.no_grad():
            prediction, _, _ = nn.get_prediction(
                input.clone(), None, 1.0, device, net, config, scale_input=True
                )

    # compute the error of the prediction compared to the measurements
    nz, ny, nx = prediction['pred'].shape[2:]

    u_interpolator = RegularGridInterpolator(
        (
            np.linspace(0, nz - 1, nz), np.linspace(0, ny - 1,
                                                    ny), np.linspace(0, nx - 1, nx)
            ),
        prediction['pred'][0, 0].cpu().detach().numpy(),
        bounds_error=False,
        fill_value=np.nan
        )
    v_interpolator = RegularGridInterpolator(
        (
            np.linspace(0, nz - 1, nz), np.linspace(0, ny - 1,
                                                    ny), np.linspace(0, nx - 1, nx)
            ),
        prediction['pred'][0, 1].cpu().detach().numpy(),
        bounds_error=False,
        fill_value=np.nan
        )
    w_interpolator = RegularGridInterpolator(
        (
            np.linspace(0, nz - 1, nz), np.linspace(0, ny - 1,
                                                    ny), np.linspace(0, nx - 1, nx)
            ),
        prediction['pred'][0, 2].cpu().detach().numpy(),
        bounds_error=False,
        fill_value=np.nan
        )

    if config.model['model_args']['use_turbulence']:
        tke_interpolator = RegularGridInterpolator(
            (
                np.linspace(0, nz - 1, nz), np.linspace(0, ny - 1,
                                                        ny), np.linspace(0, nx - 1, nx)
                ),
            prediction['pred'][0, 3].cpu().detach().numpy(),
            bounds_error=False,
            fill_value=np.nan
            )
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

    for mast in h5_file['measurement'][experiment_name][scale_keys[index]].keys():
        ds_mast = h5_file['measurement'][experiment_name][scale_keys[index]][mast]
        positions = ds_mast['pos'][...]

        u_meas = ds_mast['u'][...]
        v_meas = ds_mast['v'][...]
        w_meas = ds_mast['w'][...]
        s_meas = ds_mast['s'][...]
        if tke_available:
            tke_meas = ds_mast['tke'][...]

        lower_bound = (positions < 0).any(axis=1)
        upper_bound = (positions > [nx - 1, ny - 1, nz - 1]).any(axis=1)

        extrapolated = [lb or ub for lb, ub in zip(lower_bound, upper_bound)]
        in_bounds = [not e for e in extrapolated]

        positions_clipped = copy.copy(positions)
        positions_clipped[:, 0] = positions_clipped[:, 0].clip(0, nx - 1)
        positions_clipped[:, 1] = positions_clipped[:, 1].clip(0, ny - 1)
        positions_clipped[:, 2] = positions_clipped[:, 2].clip(0, nz - 1)

        u_pred = u_interpolator(np.fliplr(positions_clipped))
        v_pred = v_interpolator(np.fliplr(positions_clipped))
        w_pred = w_interpolator(np.fliplr(positions_clipped))
        s_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)
        if tke_interpolator:
            tke_pred = tke_interpolator(np.fliplr(positions_clipped))

        meas_3d_available = u_meas != 0

        positions_labels = copy.copy(positions)

        if not extrapolate_pred:
            positions_labels = copy.copy(positions[in_bounds, :])

            u_meas = u_meas[in_bounds]
            v_meas = v_meas[in_bounds]
            w_meas = w_meas[in_bounds]
            s_meas = s_meas[in_bounds]
            if tke_available:
                tke_meas = tke_meas[in_bounds]

            u_pred = u_pred[in_bounds]
            v_pred = v_pred[in_bounds]
            w_pred = w_pred[in_bounds]
            s_pred = s_pred[in_bounds]
            if tke_available:
                tke_pred = tke_pred[in_bounds]

            meas_3d_available = list(compress(meas_3d_available, in_bounds))
            extrapolated = list(compress(extrapolated, in_bounds))
            in_bounds = list(compress(in_bounds, in_bounds))

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

        labels_all = [
            mast + '_' + "{:.2f}".format(positions_labels[idx, 2])
            for idx in range(len(s_meas))
            ]
        labels_3d = [
            mast + '_' + "{:.2f}".format(positions_labels[idx, 2])
            for idx in range(len(s_meas)) if meas_3d_available[idx]
            ]

        results['labels_all'].extend(labels_all)
        results['labels_3d'].extend(labels_3d)
        results['extrapolated'].extend(extrapolated)
        results['in_bounds'].extend(in_bounds)
        results['extrapolated_3d'].extend([
            e for e, a in zip(extrapolated, meas_3d_available) if a
            ])
        results['in_bounds_3d'].extend([
            i for i, a in zip(in_bounds, meas_3d_available) if a
            ])

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
        ds_lines = h5_file['lines'][scale_keys[index]]
        for line_key in ds_lines.keys():
            x = ds_lines[line_key]['x'][...]
            y = ds_lines[line_key]['y'][...]
            z = ds_lines[line_key]['z'][...]

            positions_clipped = copy.copy(np.stack((x, y, z), axis=1))
            positions_clipped[:, 0] = positions_clipped[:, 0].clip(0, nx - 1)
            positions_clipped[:, 1] = positions_clipped[:, 1].clip(0, ny - 1)
            positions_clipped[:, 2] = positions_clipped[:, 2].clip(0, nz - 1)

            u_pred = u_interpolator(np.fliplr(positions_clipped))
            v_pred = v_interpolator(np.fliplr(positions_clipped))
            w_pred = w_interpolator(np.fliplr(positions_clipped))
            s_pred = np.sqrt(u_pred**2 + v_pred**2 + w_pred**2)

            if reference_mast:
                ds_mast = h5_file['measurement'][experiment_name][scale_keys[index]
                                                                  ][reference_mast]
                positions = ds_mast['pos'][...]
                s_meas = ds_mast['s'][...]

                idx_ref = np.argmin(np.abs(positions[:, 2] - z[0]))
                s_ref = s_meas[idx_ref]

            else:
                s_ref = s_pred[0]

            ret['profiles'][line_key] = {
                'terrain': ds_lines[line_key]['terrain'][...],
                'dist': ds_lines[line_key]['dist'][...],
                'z': z,
                'speedup': (s_pred - s_ref) / s_ref,
                's_pred': s_pred,
                'u_pred': u_pred,
                'v_pred': v_pred,
                'w_pred': w_pred,
                }

            if tke_interpolator:
                tke_pred = tke_interpolator(np.fliplr(positions_clipped))
                ret['profiles'][line_key]['tke_pred'] = tke_pred

    if predict_lidar:
        ds_lidar = h5_file['lidar']
        ret['lidar'] = {}
        if experiment_name in ds_lidar.keys():
            ds_lidar_case = ds_lidar[experiment_name][scale_keys[index]]
            for scan in ds_lidar_case.keys():
                ret['lidar'][scan] = {}

                x = ds_lidar_case[scan]['x'][...]
                y = ds_lidar_case[scan]['y'][...]
                z = ds_lidar_case[scan]['z'][...]
                vel = ds_lidar_case[scan]['vel'][...]
                azimuth_angle = ds_lidar_case[scan]['azimuth_angle'][...]
                elevation_angle = ds_lidar_case[scan]['elevation_angle'][...]

                positions = copy.copy(
                    np.stack((x.ravel(), y.ravel(), z.ravel()), axis=1)
                    )
                u_pred = u_interpolator(np.fliplr(positions))
                v_pred = v_interpolator(np.fliplr(positions))
                w_pred = w_interpolator(np.fliplr(positions))

                dir_vec_x = np.repeat((np.sin(azimuth_angle) *
                                       np.cos(elevation_angle))[:, np.newaxis],
                                      x.shape[1],
                                      axis=1).ravel()
                dir_vec_y = np.repeat((np.cos(azimuth_angle) *
                                       np.cos(elevation_angle))[:, np.newaxis],
                                      x.shape[1],
                                      axis=1).ravel()
                dir_vec_z = np.repeat(
                    np.sin(elevation_angle)[:, np.newaxis], x.shape[1], axis=1
                    ).ravel()

                vel_projected = (
                    u_pred * dir_vec_x + v_pred * dir_vec_y + w_pred * dir_vec_z
                    )
                ret['lidar'][scan]['x'] = x.ravel()
                ret['lidar'][scan]['y'] = y.ravel()
                ret['lidar'][scan]['z'] = z.ravel()
                ret['lidar'][scan]['pred'] = vel_projected
                ret['lidar'][scan]['meas'] = vel.ravel()

                terrain_interpolator = RegularGridInterpolator(
                    (
                        np.linspace(0, nz - 1, nz), np.linspace(0, ny - 1, ny),
                        np.linspace(0, nx - 1, nx)
                        ),
                    terrain.cpu().detach().numpy(),
                    bounds_error=False,
                    fill_value=None,
                    method='nearest'
                    )
                ret['lidar'][scan]['terrain'] = terrain_interpolator(
                    np.fliplr(positions)
                    )

                if tke_interpolator:
                    ret['lidar'][scan]['tke'] = tke_interpolator(np.fliplr(positions))

        else:
            print('No Lidar scans available for this timestamp. Available scans:')
            print(ds_lidar.keys())

    return ret
