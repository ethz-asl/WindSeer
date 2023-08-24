import argparse
import copy
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import re
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy import ndimage
from scipy.ndimage.filters import uniform_filter
import torch

import pyproj

import windseer.measurement_campaigns as mc_utils

parser = argparse.ArgumentParser(
    description='Convert the Bolund data from the zip files to the hdf5 format'
    )
parser.add_argument(
    '-t', dest='terrain_file', required=True, help='Filename of the terrain zip file'
    )
parser.add_argument(
    '-m',
    dest='measurement_file',
    required=True,
    help='Filename of the measurement zip file'
    )
parser.add_argument('-l', dest='lidar_folder', help='Folder containing the lidar data')
parser.add_argument(
    '-p',
    dest='plot',
    action='store_true',
    help='Plot the data from the different cases'
    )
parser.add_argument(
    '-s',
    dest='save',
    action='store_true',
    help='Convert and save the data in different hdf5 datasets'
    )
parser.add_argument(
    '-rh',
    dest='resolution_hor',
    type=float,
    default=16.64,
    help='The horizontal resolution of the grid'
    )
parser.add_argument(
    '-rv',
    dest='resolution_ver',
    type=float,
    default=11.57,
    help='The vertical resolution of the grid'
    )
parser.add_argument(
    '-factors',
    dest='factors',
    nargs='+',
    type=float,
    default=[1.0],
    help='Downscaling factors'
    )
parser.add_argument(
    '-n',
    dest='n_grid',
    nargs='+',
    type=int,
    default=[64],
    help='Number of cells in the grid'
    )
parser.add_argument(
    '-height_ratio',
    dest='height_ratio',
    type=float,
    default=0.5,
    help='Ratio of cells in z direction compared to the horizontal direction'
    )
parser.add_argument(
    '-o',
    dest='output_directory',
    default='',
    help='The output directory where the converted datasets are stored'
    )

args = parser.parse_args()

# tf into the local frame
wgs84_to_local = pyproj.Proj(
    "+proj=tmerc +lat_0=39.66825833333333 +lon_0=-8.133108333333334 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"
    )

# center of the map
lat_center = 39.710497
lon_center = -7.739925
x_center, y_center = wgs84_to_local(lon_center, lat_center)
z_offset = 250.0  # roughly the height of the lowest tower

# load the terrain data
terrain_dict = mc_utils.get_terrain_Perdigao(
    args.terrain_file, wgs84_to_local, lon_center, lat_center
    )

x_terrain, y_terrain = wgs84_to_local(terrain_dict['lon'], terrain_dict['lat'])
terrain_interpolator = LinearNDInterpolator(
    list(zip(x_terrain.ravel() - x_center,
             y_terrain.ravel() - y_center)),
    terrain_dict['z'].ravel() - z_offset,
    fill_value=0.0
    )

# load the measurements
tower_data = []
tower_heights = []
tower_names = mc_utils.get_Perdigao_tower_names(args.measurement_file)

for twr in tower_names:
    twr_heights = mc_utils.get_Perdigao_tower_heights(args.measurement_file, twr)
    tower_heights.append(twr_heights)
    tower_data.append(
        mc_utils.get_Perdigao_tower_data(args.measurement_file, twr, twr_heights)
        )

tower_positions = mc_utils.get_tower_positions()
warning_threshold = 111.0  # m
measurements_dict = {}
for data, heights, twr in zip(tower_data, tower_heights, tower_names):
    times = data['time']
    ms_dict = {'pos': [], 'u': [], 'v': [], 'w': [], 'spd': []}

    x_highres = tower_positions[twr]['x']
    y_highres = tower_positions[twr]['y']
    x_highres -= x_center
    y_highres -= y_center

    for ht in heights:
        for key in ['u', 'v', 'w', 'spd']:
            val = data[key + '_' + ht + '_' + twr]
            val[np.ma.getmask(val)] = np.nan
            ms_dict[key].append(val.data)
        z = int(re.sub("[^0-9]", "", ht)) + terrain_interpolator((x_highres, y_highres))
        ms_dict['pos'].append([x_highres, y_highres, z])

    for key in ms_dict:
        if len(ms_dict[key]) > 0:
            ms_dict[key] = np.vstack(ms_dict[key])

    if len(heights) > 0:
        measurements_dict[twr] = ms_dict

# load the lidar data if available
filename = args.measurement_file.split('/')[-1]
datestring = filename.split('.')[0].split('_')[-1]
lidar_filenames = []
if not args.lidar_folder is None:
    try:
        lidar_filenames = [
            f for f in os.listdir(args.lidar_folder)
            if os.path.isfile(os.path.join(args.lidar_folder, f))
            ]
    except FileNotFoundError:
        pass

lidar_data = {}
for lfn in lidar_filenames:
    if datestring in lfn:
        l_data = mc_utils.get_Perdigao_lidar_data(os.path.join(args.lidar_folder, lfn))
        for key in l_data.keys():
            if isinstance(l_data[key], np.ma.MaskedArray):
                l_data[key] = l_data[key].filled(np.nan)
        timestring = lfn.split('.')[0].split('_')[-1].replace(datestring, "")
        scanner_key = lfn.split('.')[0].split('_')[0]
        # one recording takes about 10 minutes, use the middle as the timestamp since the string indicates the end of the recording
        timestamp = int(timestring[0:2]) * 3600 + int(timestring[2:4]) * 60 + int(
            timestring[4:6]
            ) - 300
        time_idx = np.argmin(np.abs(times - timestamp))

        l_data['x'] -= x_center
        l_data['y'] -= y_center
        l_data['z'] -= z_offset

        # set all values occluded by terrain to NAN
        z_terr = terrain_interpolator(l_data['x'], l_data['y'])
        mask = (z_terr + 20) > l_data['z']
        for ray in mask:
            invalid_idx = np.argmax(ray == True)
            # argmax returns 0 if there is no True in the ray
            if ray[invalid_idx] == True:
                ray[invalid_idx:] = True

        l_data['vel'][mask] = np.nan

        if not time_idx in lidar_data.keys():
            lidar_data[time_idx] = {}

        lidar_data[time_idx][scanner_key] = l_data

if args.save:
    terrain_grids = []
    x_interpolators = []
    y_interpolators = []
    z_interpolators = []
    for n in args.n_grid:
        tg = []
        x_int = []
        y_int = []
        z_int = []
        for f in args.factors:
            nz = int(n * args.height_ratio)
            x_grid = (np.linspace(0, n - 1, n) -
                      0.5 * n) * args.resolution_hor / float(f)
            y_grid = (np.linspace(0, n - 1, n) -
                      0.5 * n) * args.resolution_hor / float(f)
            z_grid = (np.linspace(0, nz - 1, nz) - 2.5) * args.resolution_ver / float(f)
            x_int.append(
                interp1d(
                    x_grid,
                    np.linspace(0, n - 1, n),
                    bounds_error=False,
                    fill_value='extrapolate'
                    )
                )
            y_int.append(
                interp1d(
                    y_grid,
                    np.linspace(0, n - 1, n),
                    bounds_error=False,
                    fill_value='extrapolate'
                    )
                )
            z_int.append(
                interp1d(
                    z_grid,
                    np.linspace(0, nz - 1, nz),
                    bounds_error=False,
                    fill_value='extrapolate'
                    )
                )

            X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='xy')
            # check the lower bound of the cell for terrain occlusion
            Z_terrain = terrain_interpolator((X_grid, Y_grid)
                                             ) - 0.5 * args.resolution_ver / float(f)

            # we need to extend the domain in z temporarily since negative heights can be present in the larger terrain patches
            num_extra_cells = int(
                np.ceil(np.abs(np.nanmin(Z_terrain) / args.resolution_ver * float(f)))
                )
            z_grid_tmp = (
                np.linspace(-num_extra_cells, nz - 1, nz + num_extra_cells) - 2.5
                ) * args.resolution_ver / float(f)
            is_free = z_grid_tmp[:, np.newaxis,
                                 np.newaxis] > Z_terrain[np.newaxis, :, :]
            dist_field = ndimage.distance_transform_edt(is_free).astype(np.float32)
            tg.append(dist_field[num_extra_cells:])

        terrain_grids.append(tg)
        x_interpolators.append(x_int)
        y_interpolators.append(y_int)
        z_interpolators.append(z_int)

    print('Terrain interpolation finished ....')

    ds_file = h5py.File('perdigao_' + datestring + '.hdf5', 'w')
    ds_file.create_group('terrain')
    ds_file.create_group('lines')
    ds_file.create_group('measurement')
    ds_file.create_group('lidar')

    for t in times:
        ds_file['measurement'].create_group(mc_utils.get_Perdigao_time_key(t))

    for i in range(24):
        t_key = 'avg_' + str(i) + ':00-' + str(i + 1) + ':00'
        ds_file['measurement'].create_group(t_key)

    for i, n in enumerate(args.n_grid):
        for f, terrain, x_inter, y_inter, z_inter in zip(
            args.factors, terrain_grids[i], x_interpolators[i], y_interpolators[i],
            z_interpolators[i]
            ):
            s_key = 'scale_' + str(f) + '_n_' + str(n)
            ds_file.create_group(s_key)
            ds_file['terrain'].create_dataset(s_key, data=terrain)

            # convert the prediction lines
            mc_utils.add_Perdigao_measurement_lines(
                ds_file, s_key, x_inter, y_inter, z_inter, terrain_interpolator,
                measurements_dict
                )

            # Add the mast measurements
            for i in range(len(times)):
                t_key = mc_utils.get_Perdigao_time_key(times[i])
                ds_file['measurement'][t_key].create_group(s_key)

                for ms_post in measurements_dict.keys():
                    ds_file['measurement'][t_key][s_key].create_group(ms_post)

                    for d_key in measurements_dict[ms_post].keys():
                        if d_key == 'pos':
                            pos_idx = copy.copy(measurements_dict[ms_post][d_key])
                            pos_idx[:, 0] = x_inter(pos_idx[:, 0])
                            pos_idx[:, 1] = y_inter(pos_idx[:, 1])
                            pos_idx[:, 2] = z_inter(pos_idx[:, 2])

                            ds_file['measurement'][t_key][s_key][ms_post
                                                                 ].create_dataset(
                                                                     d_key,
                                                                     data=pos_idx
                                                                     )
                        elif d_key == 'spd':
                            ds_file['measurement'][t_key][s_key][
                                ms_post].create_dataset(
                                    's', data=measurements_dict[ms_post][d_key][:, i]
                                    )
                        else:
                            ds_file['measurement'][t_key][s_key][
                                ms_post].create_dataset(
                                    d_key, data=measurements_dict[ms_post][d_key][:, i]
                                    )

            # Add the lidar scans
            for i in lidar_data.keys():
                t_key = mc_utils.get_Perdigao_time_key(times[i])

                if not t_key in ds_file['lidar'].keys():
                    ds_file['lidar'].create_group(t_key)

                ds_file['lidar'][t_key].create_group(s_key)

                for lidar_id in lidar_data[i].keys():
                    ds_file['lidar'][t_key][s_key].create_group(lidar_id)
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset(
                        'x', data=x_inter(lidar_data[i][lidar_id]['x'])
                        )
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset(
                        'y', data=y_inter(lidar_data[i][lidar_id]['y'])
                        )
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset(
                        'z', data=z_inter(lidar_data[i][lidar_id]['z'])
                        )
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset(
                        'vel', data=lidar_data[i][lidar_id]['vel']
                        )
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset(
                        'elevation_angle',
                        data=lidar_data[i][lidar_id]['elevation_angle']
                        )
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset(
                        'azimuth_angle', data=lidar_data[i][lidar_id]['azimuth_angle']
                        )

            # Add the hourly averaged measurements
            for i in range(24):
                t_key = 'avg_' + str(i) + ':00-' + str(i + 1) + ':00'
                ds_file['measurement'][t_key].create_group(s_key)
                for ms_post in measurements_dict.keys():
                    ds_file['measurement'][t_key][s_key].create_group(ms_post)

                    for d_key in measurements_dict[ms_post].keys():
                        if d_key == 'pos':
                            pos_idx = copy.copy(measurements_dict[ms_post][d_key])
                            pos_idx[:, 0] = x_inter(pos_idx[:, 0])
                            pos_idx[:, 1] = y_inter(pos_idx[:, 1])
                            pos_idx[:, 2] = z_inter(pos_idx[:, 2])

                            ds_file['measurement'][t_key][s_key][ms_post
                                                                 ].create_dataset(
                                                                     d_key,
                                                                     data=pos_idx
                                                                     )
                        elif d_key == 'spd':
                            val = np.nanmean(
                                measurements_dict[ms_post][d_key][:,
                                                                  12 * i:12 * (i + 1)],
                                axis=1
                                )
                            ds_file['measurement'][t_key][s_key][ms_post
                                                                 ].create_dataset(
                                                                     's', data=val
                                                                     )
                        else:
                            val = np.nanmean(
                                measurements_dict[ms_post][d_key][:,
                                                                  12 * i:12 * (i + 1)],
                                axis=1
                                )
                            ds_file['measurement'][t_key][s_key][ms_post
                                                                 ].create_dataset(
                                                                     d_key, data=val
                                                                     )

    ds_file.close()

if args.plot:
    # plot the data
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        x_terrain - x_center,
        y_terrain - y_center,
        terrain_dict['z'] - z_offset,
        cmap=cm.terrain
        )
    idx = 0
    sum_vel = None
    counter = 0
    for key in measurements_dict.keys():
        for vel in measurements_dict[key]['spd']:
            counter += 1
            if sum_vel is None:
                sum_vel = vel
            else:
                sum_vel = np.nansum([vel, sum_vel], axis=0)

    idx = np.nanargmax(sum_vel)

    print(
        'Maximum average speed encountered in Dataset:', sum_vel[idx] / counter, 'm/s'
        )
    print(
        'Time key of maximum average velocity:',
        mc_utils.get_Perdigao_time_key(times[idx])
        )
    scale = 20

    s = []
    u = []
    v = []
    w = []
    x = []
    y = []
    z = []
    for key in measurements_dict.keys():
        x.extend(measurements_dict[key]['pos'][:, 0].tolist())
        y.extend(measurements_dict[key]['pos'][:, 1].tolist())
        z.extend(measurements_dict[key]['pos'][:, 2].tolist())
        u.extend((measurements_dict[key]['u'][:, idx] * scale).tolist())
        v.extend((measurements_dict[key]['v'][:, idx] * scale).tolist())
        w.extend((measurements_dict[key]['w'][:, idx] * scale).tolist())
        s.extend((measurements_dict[key]['spd'][:, idx] * scale).tolist())

    # Get colors
    c_array = [vel for vel in s]
    for vel in s:
        c_array.extend([vel, vel])
    c_array = np.array(c_array)
    c_array /= np.nanmax(c_array)
    q = ax.quiver(x, y, z, u, v, w, colors=cm.jet(c_array))
    q2 = ax2.quiver(x, y, z, u, v, w, colors=cm.jet(c_array))

    ax.set_xlim3d([-2000, 2000])
    ax.set_ylim3d([-2000, 2000])
    ax.set_zlim3d([0, 1000])
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Altitude (m)')

    ax2.set_xlim3d([-2000, 2000])
    ax2.set_ylim3d([-2000, 2000])
    ax2.set_zlim3d([0, 1000])
    ax2.set_xlabel('Easting (m)')
    ax2.set_ylabel('Northing (m)')
    ax2.set_zlabel('Altitude (m)')
    plt.show()
