import argparse
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import h5py
from scipy.interpolate import griddata
from scipy.stats import pearsonr
import torch

import windseer.measurement_campaigns as mc_utils
import windseer.plotting as plotting
import windseer.utils as utils

parser = argparse.ArgumentParser(
    description='Predict the flow based on the sparse measurements'
    )
parser.add_argument('-d', dest='dataset', required=True, help='The dataset file')
parser.add_argument(
    '-m',
    dest='input_mast',
    required=True,
    nargs='+',
    help='The input measurement mast'
    )
parser.add_argument('-e', dest='experiment', required=True, help='The experiment name')
parser.add_argument(
    '-i',
    dest='index',
    default=0,
    type=int,
    help='Index of the case in the dataset used for the prediction'
    )
parser.add_argument(
    '-model_dir', dest='model_dir', required=True, help='The directory of the model'
    )
parser.add_argument(
    '-model_version', dest='model_version', default='latest', help='The model version'
    )
parser.add_argument(
    '-reference_mast', help='The reference mast (used for the speedup calculation)'
    )
parser.add_argument(
    '--mayavi', action='store_true', help='Generate some extra plots using mayavi'
    )
parser.add_argument('--no_gpu', action='store_true', help='Force CPU prediction')
parser.add_argument(
    '--benchmark',
    action='store_true',
    help='Benchmark the prediction by looping through all input masks'
    )
parser.add_argument(
    '--profile', action='store_true', help='Compute the speedup profiles along lines'
    )
parser.add_argument(
    '--lidar',
    action='store_true',
    help='Compute the velocities along the lidar planes'
    )
parser.add_argument(
    '--baseline',
    action='store_true',
    help='Compute the baseline (average of measurements / GPR)'
    )
parser.add_argument(
    '--gpr',
    action='store_true',
    help='Compute the baseline using GPR instead of averaging'
    )
parser.add_argument(
    '--extrapolate',
    action='store_true',
    help='Extrapolate predictions to mast positions outside the domain.'
    )
parser.add_argument(
    '--streamlines', action='store_true', help='Display the streamlines with mayavi.'
    )
parser.add_argument(
    '--save_animation',
    action='store_true',
    help='Save an animation of the streamline plot.'
    )
parser.add_argument(
    '--azimuth', type=float, help='Set the azimuth angle of the mayavi view'
    )
parser.add_argument(
    '--elevation', type=float, help='Set the elevation angle of the mayavi view'
    )
parser.add_argument(
    '--distance', type=float, help='Set the distance of the mayavi view'
    )
parser.add_argument(
    '--focalpoint', type=float, nargs=3, help='Set the focalpoint of the mayavi view'
    )
parser.add_argument('--save', action='store_true', help='Save the prediction results')

args = parser.parse_args()

mayavi_configs = {'view_settings': {}}
if not args.azimuth is None:
    mayavi_configs['view_settings']['azimuth'] = args.azimuth
if not args.elevation is None:
    mayavi_configs['view_settings']['elevation'] = args.elevation
if not args.distance is None:
    mayavi_configs['view_settings']['distance'] = args.distance
if not args.focalpoint is None:
    mayavi_configs['view_settings']['focalpoint'] = args.focalpoint
if len(mayavi_configs['view_settings']) == 0:
    mayavi_configs['view_settings'] = None

if args.no_gpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load the NN
net, config = utils.load_model(args.model_dir, args.model_version, None, device, True)

if config.data['input_mode'] < 3:
    raise ValueError('Models with an input mode other than 5 are not supported')

config.data['input_channels'] += ['mask']

if args.benchmark:
    results = {}

    h5_file = h5py.File(args.dataset, 'r')
    scale_keys = list(h5_file['terrain'].keys())

    mast_keys = list(
        h5_file['measurement'][args.experiment][scale_keys[args.index]].keys()
        )
    h5_file.close()

    for mast in mast_keys:
        ret = mc_utils.predict_case(
            args.dataset,
            net,
            device,
            args.index, [mast],
            args.experiment,
            config,
            args.baseline,
            gpr_baseline=args.gpr,
            extrapolate_pred=args.extrapolate
            )
        if ret:
            results[mast] = ret['results']
            turbulence_predicted = ret['turb_predicted']
        else:
            print('Input mast outside the prediction region')

    all_measurements = {}
    pearson_r = {'u': [], 'v': [], 'w': [], 'tke': [], 's': []}
    for key in results.keys():
        print('---------------------------------------------------')
        print('Prediction using mast: ' + key)
        print(
            'Error U: ' +
            str(np.nanmean(np.abs(results[key]['u_meas'] - results[key]['u_pred']))) +
            ' +- ' +
            str(np.nanstd(np.abs(results[key]['u_meas'] - results[key]['u_pred'])))
            )
        print(
            'Error V: ' +
            str(np.nanmean(np.abs(results[key]['v_meas'] - results[key]['v_pred']))) +
            ' +- ' +
            str(np.nanstd(np.abs(results[key]['v_meas'] - results[key]['v_pred'])))
            )
        print(
            'Error W: ' +
            str(np.nanmean(np.abs(results[key]['w_meas'] - results[key]['w_pred']))) +
            ' +- ' +
            str(np.nanstd(np.abs(results[key]['w_meas'] - results[key]['w_pred'])))
            )
        if turbulence_predicted:
            print(
                'Error TKE: ' + str(
                    np.nanmean(
                        np.abs(results[key]['tke_meas'] - results[key]['tke_pred'])
                        )
                    ) + ' +- ' +
                str(
                    np.
                    nanstd(np.abs(results[key]['tke_meas'] - results[key]['tke_pred']))
                    )
                )
        print(
            'Error S: ' +
            str(np.nanmean(np.abs(results[key]['s_meas'] - results[key]['s_pred']))) +
            ' +- ' +
            str(np.nanstd(np.abs(results[key]['s_meas'] - results[key]['s_pred'])))
            )

        stats = pearsonr(
            results[key]['u_meas'][~np.isnan(results[key]['u_meas'])],
            results[key]['u_pred'][~np.isnan(results[key]['u_meas'])]
            )
        print('Pearson Corr U: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        pearson_r['u'].append(stats[0])
        stats = pearsonr(
            results[key]['v_meas'][~np.isnan(results[key]['v_meas'])],
            results[key]['v_pred'][~np.isnan(results[key]['v_meas'])]
            )
        print('Pearson Corr V: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        pearson_r['v'].append(stats[0])
        stats = pearsonr(
            results[key]['w_meas'][~np.isnan(results[key]['w_meas'])],
            results[key]['w_pred'][~np.isnan(results[key]['w_meas'])]
            )
        print('Pearson Corr W: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        pearson_r['w'].append(stats[0])
        if turbulence_predicted:
            stats = pearsonr(
                results[key]['tke_meas'][~np.isnan(results[key]['tke_meas'])],
                results[key]['tke_pred'][~np.isnan(results[key]['tke_meas'])]
                )
            print('Pearson Corr TKE: ' + str(stats[0]) + ', p: ' + str(stats[1]))
            pearson_r['tke'].append(stats[0])
        stats = pearsonr(
            results[key]['s_meas'][~np.isnan(results[key]['s_meas'])],
            results[key]['s_pred'][~np.isnan(results[key]['s_meas'])]
            )
        print('Pearson Corr S: ' + str(stats[0]) + ', p: ' + str(stats[1]))
        pearson_r['s'].append(stats[0])

        for field in results[key].keys():
            if 'meas' in field or 'pred' in field:
                if field in all_measurements.keys():
                    all_measurements[field] = np.concatenate(
                        (all_measurements[field], results[key][field])
                        )
                else:
                    all_measurements[field] = results[key][field]
                x = np.array([])

    print('---------------------------------------------------')
    print('Prediction error all masts')
    print(
        'Error U: ' + str(
            np.nanmean(np.abs(all_measurements['u_meas'] - all_measurements['u_pred']))
            ) + ' +- ' +
        str(np.nanstd(np.abs(all_measurements['u_meas'] - all_measurements['u_pred'])))
        )
    print(
        'Error V: ' + str(
            np.nanmean(np.abs(all_measurements['v_meas'] - all_measurements['v_pred']))
            ) + ' +- ' +
        str(np.nanstd(np.abs(all_measurements['v_meas'] - all_measurements['v_pred'])))
        )
    print(
        'Error W: ' + str(
            np.nanmean(np.abs(all_measurements['w_meas'] - all_measurements['w_pred']))
            ) + ' +- ' +
        str(np.nanstd(np.abs(all_measurements['w_meas'] - all_measurements['w_pred'])))
        )
    if turbulence_predicted:
        print(
            'Error TKE: ' + str(
                np.nanmean(
                    np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred'])
                    )
                ) + ' +- ' +
            str(
                np.nanstd(
                    np.abs(all_measurements['tke_meas'] - all_measurements['tke_pred'])
                    )
                )
            )
    print(
        'Error S: ' + str(
            np.nanmean(np.abs(all_measurements['s_meas'] - all_measurements['s_pred']))
            ) + ' +- ' +
        str(np.nanstd(np.abs(all_measurements['s_meas'] - all_measurements['s_pred'])))
        )

    stats = pearsonr(
        all_measurements['u_meas'][~np.isnan(all_measurements['u_meas'])],
        all_measurements['u_pred'][~np.isnan(all_measurements['u_meas'])]
        )
    print('Pearson Corr U: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    stats = pearsonr(
        all_measurements['v_meas'][~np.isnan(all_measurements['v_meas'])],
        all_measurements['v_pred'][~np.isnan(all_measurements['v_meas'])]
        )
    print('Pearson Corr V: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    stats = pearsonr(
        all_measurements['w_meas'][~np.isnan(all_measurements['w_meas'])],
        all_measurements['w_pred'][~np.isnan(all_measurements['w_meas'])]
        )
    print('Pearson Corr W: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    if turbulence_predicted:
        stats = pearsonr(
            all_measurements['tke_meas'][~np.isnan(all_measurements['tke_meas'])],
            all_measurements['tke_pred'][~np.isnan(all_measurements['tke_meas'])]
            )
        print('Pearson Corr TKE: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    stats = pearsonr(
        all_measurements['s_meas'][~np.isnan(all_measurements['s_meas'])],
        all_measurements['s_pred'][~np.isnan(all_measurements['s_meas'])]
        )
    print('Pearson Corr S: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    print('---------------------------------------------------')
    print('Pearson corr per case, then averaged over all cases')
    print('U:   ' + str(np.nanmean(pearson_r['u'])))
    print('V:   ' + str(np.nanmean(pearson_r['v'])))
    print('W:   ' + str(np.nanmean(pearson_r['w'])))
    if turbulence_predicted:
        print('TKE: ' + str(np.nanmean(pearson_r['tke'])))
    print('S:   ' + str(np.nanmean(pearson_r['s'])))
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
    ret = mc_utils.predict_case(
        args.dataset, net, device, args.index, args.input_mast, args.experiment, config,
        args.baseline, args.profile, args.reference_mast, args.lidar, args.gpr,
        args.extrapolate
        )

    if not ret:
        raise ValueError('No prediction because input mast not in prediction domain')

    if args.save:
        if args.baseline:
            if args.gpr:
                model_name = 'GPR'
            else:
                model_name = 'AVG'
        else:
            if args.model_dir.split('/')[-1] is '':
                model_name = args.model_dir.split('/')[-2]
            else:
                model_name = args.model_dir.split('/')[-1]

        savename = args.dataset.split('/')[-1].split('.')[
            0] + '_' + args.experiment + '_' + args.input_mast[0] + '_' + model_name
        savedata = [ret['results']]
        if 'profiles' in ret.keys():
            savedata.append(ret['profiles'])
        np.save(savename, savedata)
        exit()

    x_all = np.linspace(
        0,
        len(ret['results']['labels_all']) - 1, len(ret['results']['labels_all'])
        )
    x_3d = np.linspace(
        0,
        len(ret['results']['labels_3d']) - 1, len(ret['results']['labels_3d'])
        )

    num_plots = 3
    if ret['turb_predicted']:
        num_plots = 4
    fig, ah = plt.subplots(num_plots, 1, squeeze=False)
    ah[0][0].set_title('Prediction vs Measurement')
    ah[0][0].plot(x_3d, ret['results']['u_meas'], 'sr', label='measurements')
    ah[0][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        ret['results']['u_pred'][ret['results']['extrapolated_3d']],
        '^c',
        label='prediction extrapolated'
        )
    ah[0][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        ret['results']['u_pred'][ret['results']['in_bounds_3d']],
        'ob',
        label='prediction'
        )

    ah[1][0].plot(x_3d, ret['results']['v_meas'], 'sr', label='measurements')
    ah[1][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        ret['results']['v_pred'][ret['results']['extrapolated_3d']],
        '^c',
        label='prediction extrapolated'
        )
    ah[1][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        ret['results']['v_pred'][ret['results']['in_bounds_3d']],
        'ob',
        label='prediction'
        )

    ah[2][0].plot(x_3d, ret['results']['w_meas'], 'sr', label='measurements')
    ah[2][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        ret['results']['w_pred'][ret['results']['extrapolated_3d']],
        '^c',
        label='prediction extrapolated'
        )
    ah[2][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        ret['results']['w_pred'][ret['results']['in_bounds_3d']],
        'ob',
        label='prediction'
        )

    ah[0][0].set_ylabel('U [m/s]')
    ah[1][0].set_ylabel('V [m/s]')
    ah[2][0].set_ylabel('W [m/s]')

    ah[0][0].axes.xaxis.set_visible(False)
    ah[1][0].axes.xaxis.set_visible(False)

    ah[0][0].legend()

    if ret['turb_predicted']:
        ah[3][0].plot(x_3d, ret['results']['tke_meas'], 'sr', label='measurements')
        ah[3][0].plot(
            x_3d[ret['results']['extrapolated_3d']],
            ret['results']['tke_pred'][ret['results']['extrapolated_3d']],
            '^c',
            label='prediction extrapolated'
            )
        ah[3][0].plot(
            x_3d[ret['results']['in_bounds_3d']],
            ret['results']['tke_pred'][ret['results']['in_bounds_3d']],
            'ob',
            label='prediction'
            )
        ah[3][0].set_ylabel('TKE [m2/s2]')

        ah[2][0].axes.xaxis.set_visible(False)
        ah[3][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[3][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    else:
        ah[2][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[2][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    fig, ah = plt.subplots(num_plots, 1, squeeze=False)
    ah[0][0].set_title('Prediction Errors')
    ah[0][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        ret['results']['u_meas'][ret['results']['extrapolated_3d']] -
        ret['results']['u_pred'][ret['results']['extrapolated_3d']],
        '^c',
        label='extrapolated'
        )
    ah[1][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        ret['results']['v_meas'][ret['results']['extrapolated_3d']] -
        ret['results']['v_pred'][ret['results']['extrapolated_3d']],
        '^c',
        label='extrapolated'
        )
    ah[2][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        ret['results']['w_meas'][ret['results']['extrapolated_3d']] -
        ret['results']['w_pred'][ret['results']['extrapolated_3d']],
        '^c',
        label='extrapolated'
        )

    ah[0][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        ret['results']['u_meas'][ret['results']['in_bounds_3d']] -
        ret['results']['u_pred'][ret['results']['in_bounds_3d']],
        'ob',
        label='in bounds'
        )
    ah[1][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        ret['results']['v_meas'][ret['results']['in_bounds_3d']] -
        ret['results']['v_pred'][ret['results']['in_bounds_3d']],
        'ob',
        label='in bounds'
        )
    ah[2][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        ret['results']['w_meas'][ret['results']['in_bounds_3d']] -
        ret['results']['w_pred'][ret['results']['in_bounds_3d']],
        'ob',
        label='in bounds'
        )

    ah[0][0].set_ylabel('Error U [m/s]')
    ah[1][0].set_ylabel('Error V [m/s]')
    ah[2][0].set_ylabel('Error W [m/s]')

    ah[0][0].axes.xaxis.set_visible(False)
    ah[1][0].axes.xaxis.set_visible(False)

    ah[0][0].legend()

    if ret['turb_predicted']:
        ah[3][0].plot(
            x_3d[ret['results']['extrapolated_3d']],
            ret['results']['tke_meas'][ret['results']['extrapolated_3d']] -
            ret['results']['tke_pred'][ret['results']['extrapolated_3d']],
            '^c',
            label='extrapolated'
            )
        ah[3][0].plot(
            x_3d[ret['results']['in_bounds_3d']],
            ret['results']['tke_meas'][ret['results']['in_bounds_3d']] -
            ret['results']['tke_pred'][ret['results']['in_bounds_3d']],
            'ob',
            label='in bounds'
            )
        ah[3][0].set_ylabel('Error TKE [m2/s2]')

        ah[2][0].axes.xaxis.set_visible(False)
        ah[3][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[3][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    else:
        ah[2][0].set_xticks(np.arange(len(ret['results']['labels_3d'])))
        ah[2][0].set_xticklabels(ret['results']['labels_3d'], rotation='vertical')

    fig, ah = plt.subplots(num_plots, 1, squeeze=False)
    ah[0][0].set_title('Relative Prediction Errors')
    ah[0][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        (
            ret['results']['u_meas'][ret['results']['extrapolated_3d']] -
            ret['results']['u_pred'][ret['results']['extrapolated_3d']]
            ) / np.abs(ret['results']['u_meas'][ret['results']['extrapolated_3d']]),
        '^c',
        label='extrapolated'
        )
    ah[1][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        (
            ret['results']['v_meas'][ret['results']['extrapolated_3d']] -
            ret['results']['v_pred'][ret['results']['extrapolated_3d']]
            ) / np.abs(ret['results']['v_meas'][ret['results']['extrapolated_3d']]),
        '^c',
        label='extrapolated'
        )
    ah[2][0].plot(
        x_3d[ret['results']['extrapolated_3d']],
        (
            ret['results']['w_meas'][ret['results']['extrapolated_3d']] -
            ret['results']['w_pred'][ret['results']['extrapolated_3d']]
            ) / np.abs(ret['results']['w_meas'][ret['results']['extrapolated_3d']]),
        '^c',
        label='extrapolated'
        )

    ah[0][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        (
            ret['results']['u_meas'][ret['results']['in_bounds_3d']] -
            ret['results']['u_pred'][ret['results']['in_bounds_3d']]
            ) / np.abs(ret['results']['u_meas'][ret['results']['in_bounds_3d']]),
        'ob',
        label='in bounds'
        )
    ah[1][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        (
            ret['results']['v_meas'][ret['results']['in_bounds_3d']] -
            ret['results']['v_pred'][ret['results']['in_bounds_3d']]
            ) / np.abs(ret['results']['v_meas'][ret['results']['in_bounds_3d']]),
        'ob',
        label='in bounds'
        )
    ah[2][0].plot(
        x_3d[ret['results']['in_bounds_3d']],
        (
            ret['results']['w_meas'][ret['results']['in_bounds_3d']] -
            ret['results']['w_pred'][ret['results']['in_bounds_3d']]
            ) / np.abs(ret['results']['w_meas'][ret['results']['in_bounds_3d']]),
        'ob',
        label='in bounds'
        )

    ah[0][0].set_ylabel('Error Rel U [-]')
    ah[1][0].set_ylabel('Error Rel V [-]')
    ah[2][0].set_ylabel('Error Rel W [-]')

    ah[0][0].axes.xaxis.set_visible(False)
    ah[1][0].axes.xaxis.set_visible(False)

    ah[0][0].legend()

    if ret['turb_predicted']:
        ah[3][0].plot(
            x_3d[ret['results']['extrapolated_3d']],
            (
                ret['results']['tke_meas'][ret['results']['extrapolated_3d']] -
                ret['results']['tke_pred'][ret['results']['extrapolated_3d']]
                ) /
            np.abs(ret['results']['tke_meas'][ret['results']['extrapolated_3d']]),
            '^c',
            label='extrapolated'
            )
        ah[3][0].plot(
            x_3d[ret['results']['in_bounds_3d']],
            (
                ret['results']['tke_meas'][ret['results']['in_bounds_3d']] -
                ret['results']['tke_pred'][ret['results']['in_bounds_3d']]
                ) / np.abs(ret['results']['tke_meas'][ret['results']['in_bounds_3d']]),
            'ob',
            label='in bounds'
            )
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
    ah[0][0].plot(
        x_all[ret['results']['extrapolated']],
        ret['results']['s_pred'][ret['results']['extrapolated']],
        '^c',
        label='prediction extrapolated'
        )
    ah[0][0].plot(
        x_all[ret['results']['in_bounds']],
        ret['results']['s_pred'][ret['results']['in_bounds']],
        'ob',
        label='prediction'
        )

    ah[0][0].set_ylabel('Magnitude [m/s]')

    ah[0][0].set_xticks(np.arange(len(ret['results']['labels_all'])))
    ah[0][0].set_xticklabels(ret['results']['labels_all'], rotation='vertical')
    ah[0][0].legend()

    fig, ah = plt.subplots(1, 1, squeeze=False)
    ah[0][0].set_title('Prediction Magnitude Error')
    ah[0][0].plot(
        x_all[ret['results']['extrapolated']],
        ret['results']['s_meas'][ret['results']['extrapolated']] -
        ret['results']['s_pred'][ret['results']['extrapolated']],
        '^c',
        label='extrapolated'
        )
    ah[0][0].plot(
        x_all[ret['results']['in_bounds']],
        ret['results']['s_meas'][ret['results']['in_bounds']] -
        ret['results']['s_pred'][ret['results']['in_bounds']],
        'ob',
        label='in bounds'
        )

    ah[0][0].set_ylabel('Error Magnitude [m/s]')

    ah[0][0].set_xticks(np.arange(len(ret['results']['labels_all'])))
    ah[0][0].set_xticklabels(ret['results']['labels_all'], rotation='vertical')
    ah[0][0].legend()

    # Display results
    print('---------------------------------------------------')
    print('Prediction using mast: ' + mc_utils.masts_to_string(args.input_mast))
    print(
        'Error U: ' +
        str(np.nanmean(np.abs(ret['results']['u_meas'] - ret['results']['u_pred']))),
        '+-',
        str(np.nanstd(np.abs(ret['results']['u_meas'] - ret['results']['u_pred'])))
        )
    print(
        'Error V: ' +
        str(np.nanmean(np.abs(ret['results']['v_meas'] - ret['results']['v_pred']))),
        '+-',
        str(np.nanstd(np.abs(ret['results']['v_meas'] - ret['results']['v_pred'])))
        )
    print(
        'Error W: ' +
        str(np.nanmean(np.abs(ret['results']['w_meas'] - ret['results']['w_pred']))),
        '+-',
        str(np.nanstd(np.abs(ret['results']['w_meas'] - ret['results']['w_pred'])))
        )
    if ret['turb_predicted']:
        print(
            'Error TKE: ' + str(
                np.nanmean(
                    np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred'])
                    )
                ), '+-',
            str(
                np.nanstd(
                    np.abs(ret['results']['tke_meas'] - ret['results']['tke_pred'])
                    )
                )
            )
    print(
        'Error S: ' +
        str(np.nanmean(np.abs(ret['results']['s_meas'] - ret['results']['s_pred']))),
        '+-',
        str(np.nanstd(np.abs(ret['results']['s_meas'] - ret['results']['s_pred'])))
        )

    stats = pearsonr(
        ret['results']['u_meas'][~np.isnan(ret['results']['u_meas'])],
        ret['results']['u_pred'][~np.isnan(ret['results']['u_meas'])]
        )
    print('Pearson Corr U: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    stats = pearsonr(
        ret['results']['v_meas'][~np.isnan(ret['results']['v_meas'])],
        ret['results']['v_pred'][~np.isnan(ret['results']['v_meas'])]
        )
    print('Pearson Corr V: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    stats = pearsonr(
        ret['results']['w_meas'][~np.isnan(ret['results']['w_meas'])],
        ret['results']['w_pred'][~np.isnan(ret['results']['w_meas'])]
        )
    print('Pearson Corr W: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    if ret['turb_predicted']:
        stats = pearsonr(
            ret['results']['tke_meas'][~np.isnan(ret['results']['tke_meas'])],
            ret['results']['tke_pred'][~np.isnan(ret['results']['tke_meas'])]
            )
        print('Pearson Corr TKE: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    stats = pearsonr(
        ret['results']['s_meas'][~np.isnan(ret['results']['s_meas'])],
        ret['results']['s_pred'][~np.isnan(ret['results']['s_meas'])]
        )
    print('Pearson Corr S: ' + str(stats[0]) + ', p: ' + str(stats[1]))
    print('---------------------------------------------------')

    if args.profile:
        for line_key in ret['profiles'].keys():
            plt.figure()
            plt.plot(
                ret['profiles'][line_key]['dist'], ret['profiles'][line_key]['speedup']
                )
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
            dist_resampled = np.linspace(
                0, int(dist.max()), int(int(dist.max() + 1) / resolution)
                )
            min_z = np.max((z.min(), 0))
            max_z = np.min((z.max(), ret['prediction']['pred'].shape[2]))
            z_resampled = np.linspace(
                min_z, max_z, int(int(max_z - min_z + 1) / resolution)
                )

            D_grid, Z_grid = np.meshgrid(dist_resampled, z_resampled, indexing='xy')
            measured = griddata((dist, z),
                                ret['lidar'][lidar_key]['meas'], (D_grid, Z_grid),
                                method='linear')
            predicted = griddata((dist, z),
                                 ret['lidar'][lidar_key]['pred'], (D_grid, Z_grid),
                                 method='linear')

            cmap_terrain = colors.LinearSegmentedColormap.from_list(
                'custom', colors.to_rgba_array(['grey', 'grey']), 2
                )
            terrain = griddata((dist, z),
                               ret['lidar'][lidar_key]['terrain'], (D_grid, Z_grid),
                               method='nearest')
            no_terrain = np.logical_not(np.logical_not(terrain.astype(bool)))
            terrain_mask = np.ma.masked_where(no_terrain, no_terrain)

            min_val = np.nanmin((np.nanmin(measured), np.nanmin(predicted)))
            max_val = np.nanmax((np.nanmax(measured), np.nanmax(predicted)))

            fig, ah = plt.subplots(2, 2, squeeze=False)
            ah[0][0].set_title('Prediction')
            im = ah[0][0].imshow(
                predicted, origin='lower', vmin=min_val, vmax=max_val, cmap=cm.jet
                )
            fig.colorbar(im, ax=ah[0][0])
            ah[0][0].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

            ah[1][0].set_title('Measurement')
            im = ah[1][0].imshow(
                measured, origin='lower', vmin=min_val, vmax=max_val, cmap=cm.jet
                )
            fig.colorbar(im, ax=ah[1][0])
            ah[1][0].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

            ah[0][1].set_title('Error')
            im = ah[0][1].imshow(measured - predicted, origin='lower', cmap=cm.jet)
            fig.colorbar(im, ax=ah[0][1])
            ah[0][1].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

            if 'tke' in ret['lidar'][lidar_key].keys():
                tke = griddata((dist, z),
                               ret['lidar'][lidar_key]['tke'], (D_grid, Z_grid),
                               method='linear')
                ah[1][1].set_title('TKE predicted')
                ah[1][1].imshow(tke, origin='lower', cmap=cm.jet)
                ah[1][1].imshow(terrain_mask, cmap=cmap_terrain, origin='lower')

    if args.mayavi:
        ui = []

        ui.append(
            plotting.mlab_plot_prediction(
                ret['prediction']['pred'],
                ret['terrain'],
                terrain_mode='blocks',
                terrain_uniform_color=False,
                prediction_channels=config.data['label_channels'],
                view_settings=mayavi_configs['view_settings'],
                blocking=False
                )
            )

        plotting.mlab_plot_measurements(
            ret['input'][0, 1:-1],
            ret['input'][0, -1],
            ret['terrain'],
            terrain_mode='blocks',
            terrain_uniform_color=False,
            view_settings=mayavi_configs['view_settings'],
            blocking=False
            )

    if args.streamlines:
        plotting.mlab_plot_streamlines(
            ret['prediction']['pred'],
            ret['terrain'],
            terrain_mode='blocks',
            terrain_uniform_color=True,
            blocking=False,
            view_settings=mayavi_configs['view_settings'],
            animate=args.save_animation,
            save_animation=args.save_animation,
            title='Predicted Flow'
            )

    plotting.plot_prediction(
        config.data['label_channels'],
        prediction=ret['prediction']['pred'][0].cpu().detach(),
        provided_input_channels=config.data['input_channels'],
        input=ret['input'][0].cpu().detach(),
        terrain=ret['terrain'].cpu().squeeze()
        )
