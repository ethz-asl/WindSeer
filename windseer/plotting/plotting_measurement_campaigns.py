import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

from windseer.measurement_campaigns.measurements_line import get_tower_distances_on_line


def plot_measurement_campaign_scatter(ret):
    '''
    Plot the predictions of the models on the measurement campaigns data compared to the
    measurements at each measurement location.

    Parameters
    ----------
    ret : dict
        Output from predict_case containing the measurements and the predictions
    '''
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


def plot_measurement_campaings_lidar(ret):
    '''
    Plot the measured lidar data and compare it to the prediction.

    Parameters
    ----------
    ret : dict
        Output from predict_case containing the measurements and the predictions
    '''
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


def plot_measurement_campaigns_prediction_lines(ret, measurements_masts):
    '''
    Plot the prediction and measurements along the prediction lines.

    Parameters
    ----------
    ret : dict
        Output from predict_case containing the measurements and the predictions
    '''
    fig_size = (16.0, 5.6)

    for line_key in ret['profiles'].keys():
        fig, ah = plt.subplots(4, 1, squeeze=False, figsize=fig_size)
        ah[0][0].set_title(line_key)
        if ret['turb_predicted']:
            channel_keys = ['s_pred', 'w_pred', 'tke_pred']
            ah[0][0].set_ylabel(r'$S$ $[m/s]$')
            ah[1][0].set_ylabel(r'$W$ $[m/s]$')
            ah[2][0].set_ylabel(r'$TKE$ $[m^2/s^2]$')
        else:
            channel_keys = ['u_pred', 'v_pred', 'w_pred']
            ah[0][0].set_ylabel(r'$U$ $[m/s]$')
            ah[1][0].set_ylabel(r'$V$ $[m/s]$')
            ah[2][0].set_ylabel(r'$W$ $[m/s]$')

        towers = get_tower_distances_on_line(line_key)

        if line_key == 'lineB_5m':
            line_height = 5
            tower_height = 23
            terrain_limits = [0, 30]
            x_lims = [-100, 150]

        elif line_key == 'lineA_10m':
            line_height = 10
            tower_height = 170
            terrain_limits = [0, 250]
            x_lims = [-900, 450]

        elif line_key == 'lineTSE_30m':
            line_height = 30
            tower_height = 330
            terrain_limits = [0, 530]
            x_lims = [-1680, 810]

        elif line_key == 'lineTNW_20m':
            x_lims = [-1450, 1100]
            tower_height = 150
            terrain_limits = [0, 530]
            line_height = 20

        # prediction
        for i, ch in enumerate(channel_keys):
            ah[i][0].plot(
                ret['profiles'][line_key]['dist'], ret['profiles'][line_key][ch], lw=1
                )

        ah[0][0].axes.xaxis.set_visible(False)
        ah[1][0].axes.xaxis.set_visible(False)
        ah[2][0].axes.xaxis.set_visible(False)

        # terrain and measurements
        for twr in towers.keys():
            index_dist = np.argmin(
                np.abs(ret['profiles'][line_key]['dist'] - towers[twr])
                )
            line_height = ret['profiles'][line_key]['z'][index_dist]
            terrain_height = ret['profiles'][line_key]['terrain'][index_dist]
            tower_data = measurements_masts[twr.lower().replace('*', '')]
            measurement_heights = tower_data['pos'][:, 2]
            idx_measurement = np.argmin(np.abs(line_height - measurement_heights))

            for i, ch in enumerate(channel_keys):
                ah[i][0].errorbar(
                    towers[twr],
                    tower_data[ch.split('_')[0]][idx_measurement],
                    fmt='D',
                    color='black',
                    ecolor='black',
                    barsabove=False,
                    markersize=2.5,
                    elinewidth=2,
                    capsize=3,
                    zorder=50
                    )

            ah[3][0].annotate(
                twr,
                xy=(towers[twr], 0),
                xytext=(towers[twr], terrain_height + tower_height),
                horizontalalignment="center",
                arrowprops=dict(arrowstyle="-"),
                verticalalignment="top",
                zorder=0
                )

        ah[3][0].plot(
            ret['profiles'][line_key]['dist'],
            ret['profiles'][line_key]['terrain'] + line_height,
            color='black',
            lw=1.0
            )
        ah[3][0].fill_between(
            ret['profiles'][line_key]['dist'],
            ret['profiles'][line_key]['terrain'],
            color='lightgrey',
            linewidth=0.0
            )
        ah[3][0].plot(
            ret['profiles'][line_key]['dist'],
            ret['profiles'][line_key]['terrain'],
            color='dimgrey',
            lw=0.3
            )
        ah[3][0].set_ylim(terrain_limits)
        ah[3][0].set_ylabel(r'Terrain $[m]$')
        ah[3][0].set_xlabel(r'Distance $[m]$')
