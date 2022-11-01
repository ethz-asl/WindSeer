import argparse
import copy
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy import ndimage
import torch

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
parser.add_argument(
    '-x_shift', dest='x_shift', type=float, default=0.0, help='Shift in x direction'
    )
parser.add_argument(
    '-y_shift', dest='y_shift', type=float, default=0.0, help='Shift in x direction'
    )
args = parser.parse_args()

# load the terrain data
terrain_dict = mc_utils.get_terrain_Bolund(args.terrain_file)

# load the measurements
measurements, measurements_names, names = mc_utils.get_measurements_Bolund(
    args.measurement_file
    )

if args.save:
    grid_interpolator = RegularGridInterpolator((terrain_dict['y'], terrain_dict['x']),
                                                terrain_dict['Z'].T,
                                                method='linear',
                                                bounds_error=False,
                                                fill_value=None)
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
                      0.5 * n) * args.resolution_hor / float(f) + args.x_shift
            y_grid = (np.linspace(0, n - 1, n) -
                      0.5 * n) * args.resolution_hor / float(f) + args.y_shift
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
            Z_terrain = grid_interpolator((Y_grid, X_grid)
                                          ) - 0.5 * args.resolution_ver / float(f)

            is_free = z_grid[:, np.newaxis, np.newaxis] > Z_terrain[np.newaxis, :, :]
            tg.append(ndimage.distance_transform_edt(is_free).astype(np.float32))

        terrain_grids.append(tg)
        x_interpolators.append(x_int)
        y_interpolators.append(y_int)
        z_interpolators.append(z_int)

    measurements_dict = []
    for meas, meas_names in zip(measurements, measurements_names):
        ms_dict = {}
        for ms, ms_n in zip(meas, meas_names):
            if not ms_n[:2] in ms_dict.keys():
                ms_dict[ms_n[:2]] = {
                    'pos': [],
                    'u': [],
                    'v': [],
                    'w': [],
                    's': [],
                    'tke': []
                    }
            ms_dict[ms_n[:2]]['pos'].append(ms[2:5])
            u_star0 = ms[6]
            ms_dict[ms_n[:2]]['s'].append(ms[7] * u_star0)
            ms_dict[ms_n[:2]]['u'].append(ms[8] * u_star0)
            ms_dict[ms_n[:2]]['v'].append(ms[9] * u_star0)
            ms_dict[ms_n[:2]]['w'].append(ms[10] * u_star0)
            ms_dict[ms_n[:2]]['tke'].append(ms[11] * u_star0 * u_star0)

        for ms_id in ms_dict.keys():
            for key in ms_dict[ms_id]:
                if key == 'pos':
                    ms_dict[ms_id][key] = np.vstack(ms_dict[ms_id][key])
                else:
                    ms_dict[ms_id][key] = np.array(ms_dict[ms_id][key])

        measurements_dict.append(ms_dict)

    ds_file = h5py.File('bolund.hdf5', 'w')
    ds_file.create_group('terrain')
    ds_file.create_group('lines')
    ds_file.create_group('measurement')
    for name in names:
        ds_file['measurement'].create_group(name)

    for i, n in enumerate(args.n_grid):
        for f, terrain, x_inter, y_inter, z_inter in zip(
            args.factors, terrain_grids[i], x_interpolators[i], y_interpolators[i],
            z_interpolators[i]
            ):
            key = 'scale_' + str(f) + '_n_' + str(n)
            ds_file['terrain'].create_dataset(key, data=terrain)

            # convert the prediction lines
            mc_utils.add_Bolund_measurement_lines(
                ds_file, key, x_inter, y_inter, z_inter, grid_interpolator
                )

            # add the measurements
            for meas, meas_names, name in zip(
                measurements_dict, measurements_names, names
                ):
                ds_file['measurement'][name].create_group(key)

                # Add the mast measurements
                for ms_post in meas.keys():
                    ds_file['measurement'][name][key].create_group(ms_post)
                    for d_key in meas[ms_post].keys():
                        if d_key == 'pos':
                            pos_idx = copy.copy(meas[ms_post][d_key])
                            pos_idx[:, 0] = x_inter(pos_idx[:, 0])
                            pos_idx[:, 1] = y_inter(pos_idx[:, 1])
                            pos_idx[:, 2] = z_inter(pos_idx[:, 2])

                            ds_file['measurement'][name][key][ms_post].create_dataset(
                                d_key, data=pos_idx
                                )
                        else:
                            ds_file['measurement'][name][key][ms_post].create_dataset(
                                d_key, data=meas[ms_post][d_key]
                                )

    ds_file.close()

if args.plot:
    for meas, name in zip(measurements, names):
        # plot the data
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(
            terrain_dict['X'], terrain_dict['Y'], terrain_dict['Z'].T, cmap=cm.terrain
            )

        # Get colors
        c_array = [v for v in meas[:, 7]]
        for v in meas[:, 7]:
            c_array.extend([v, v])
        c_array = np.array(c_array)
        c_array /= c_array.max()

        u, v, w = meas[:, 6] * meas[:, 8], meas[:, 6] * meas[:, 9], meas[:,
                                                                         6] * meas[:,
                                                                                   10]

        q = ax.quiver(
            meas[:, 2], meas[:, 3], meas[:, 4], u, v, w, colors=cm.jet(c_array)
            )

        ax.set_xlim3d([-200, 350])
        ax.set_ylim3d([-275, 275])
        ax.set_zlim3d([0, 100])
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(name)
    plt.show()
