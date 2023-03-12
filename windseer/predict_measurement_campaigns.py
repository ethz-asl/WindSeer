import argparse
import matplotlib.pyplot as plt
import numpy as np
import h5py
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

if args.save:
    if args.baseline:
        if args.gpr:
            model_name = 'GPR'
        else:
            model_name = 'AVG'
    else:
        if args.model_dir.split('/')[-1] == '':
            model_name = args.model_dir.split('/')[-2]
        else:
            model_name = args.model_dir.split('/')[-1]

    h5_file = h5py.File(args.dataset, 'r')
    scale_keys = list(h5_file['terrain'].keys())
    scale_key = scale_keys[args.index]
    h5_file.close()

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

    if args.save:
        savename = args.dataset.split('/')[-1].split('.')[
            0] + '_' + args.experiment + '_' + scale_key + '_' + model_name
        out_dict = {}
        channels = ['u_', 'v_', 'w_']
        if turbulence_predicted:
            channels += ['tke_']
        for ch in channels:
            for property in ['meas', 'pred']:
                key = ch + property
                if not key in out_dict:
                    out_dict[key] = []

                for mast in results.keys():
                    out_dict[key] += results[mast][key].tolist()
        np.save(savename, out_dict)
        exit()

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
        savename = args.dataset.split('/')[-1].split('.')[
            0] + '_' + args.experiment + '_' + args.input_mast[0] + '_' + scale_key + '_' + model_name
        savedata = [ret['results']]
        if 'profiles' in ret.keys():
            savedata.append(ret['profiles'])
        np.save(savename, savedata)
        exit()

    # scatter plots of the measurements compared to the predictions
    plotting.plot_measurement_campaign_scatter(ret)

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
        measurements_masts = mc_utils.get_tower_measurements(
            args.dataset, args.experiment, args.index
            )
        plotting.plot_measurement_campaigns_prediction_lines(ret, measurements_masts)

    if args.lidar:
        plotting.plot_measurement_campaings_lidar(ret)

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
