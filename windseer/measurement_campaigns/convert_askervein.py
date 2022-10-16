import argparse
import copy
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from osgeo import gdal
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy import ndimage
from scipy.io import loadmat

from askervein_measurements import get_measurements


def read_mat_terrain(infile):
    data = loadmat(infile)

    return np.squeeze(data['x']), np.squeeze(data['y']), data['X'], data['Y'], data['Z']


def get_terrain_tif(filename):
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)

    geoTransform = dataset.GetGeoTransform()
    x_grid = geoTransform[0] + np.arange(
        dataset.RasterXSize
        ) * geoTransform[1] + np.arange(dataset.RasterYSize) * geoTransform[2]
    y_grid = geoTransform[3] + np.arange(
        dataset.RasterXSize
        ) * geoTransform[4] + np.arange(dataset.RasterYSize) * geoTransform[5]

    terrain_offset = np.array([75383, 823737])
    idx_ht_x = np.argmin(np.abs(x_grid - terrain_offset[0]))
    idx_ht_y = np.argmin(np.abs(y_grid - terrain_offset[1]))

    # extract a 5km x 5km patch
    num_elements = int(5000.0 / geoTransform[1])
    num_elements_half = int(0.5 * num_elements)
    start_idx_x = int(idx_ht_x - num_elements_half)
    start_idx_y = int(idx_ht_y - num_elements_half)
    Z_geo = dataset.GetRasterBand(1).ReadAsArray(
        start_idx_x, start_idx_y, num_elements, num_elements
        )

    x, y = np.meshgrid(
        np.arange(dataset.RasterXSize)[idx_ht_x - num_elements_half:idx_ht_x -
                                       num_elements_half + num_elements],
        np.arange(dataset.RasterYSize)[idx_ht_y - num_elements_half:idx_ht_y -
                                       num_elements_half + num_elements]
        )
    X_geo = geoTransform[
        0] + x * geoTransform[1] + y * geoTransform[2] - terrain_offset[0]
    Y_geo = geoTransform[
        3] + x * geoTransform[4] + y * geoTransform[5] - terrain_offset[1]

    X_geo = np.flipud(X_geo)
    Y_geo = np.flipud(Y_geo)
    Z_geo = np.flipud(Z_geo)

    return X_geo[0], Y_geo[:, 0], X_geo, Y_geo, Z_geo


parser = argparse.ArgumentParser(
    description='Convert the Askervein data from the zip files to the hdf5 format'
    )
parser.add_argument(
    '-t', dest='terrain_file', required=True, help='Filename of the terrain zip file'
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
if args.terrain_file.lower().endswith(('.mat')):
    x, y, X, Y, Z = read_mat_terrain(args.terrain_file)

elif args.terrain_file.lower().endswith(('.tif')):
    x, y, X, Y, Z = get_terrain_tif(args.terrain_file)

else:
    print('Unknown terrain file type: ', args.terrain_file)
    exit()

# load the measurements
experiment_names = [
    'TU25', 'TU30A', 'TU30B', 'TU01A', 'TU01B', 'TU01C', 'TU01D', 'TU03A', 'TU03B',
    'TU05A', 'TU05B', 'TU05C', 'TU07B'
    ]
measurements = []
for name in experiment_names:
    measurements.append(get_measurements(name))

# Terrain has the center at HT, the measurements are defined in a different frame
measurement_offset = np.array([984, 1695])
grid_interpolator = RegularGridInterpolator((y, x),
                                            Z,
                                            method='linear',
                                            bounds_error=False,
                                            fill_value=None)

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

    # convert the measurements in the correct frame
    measurements_dict = []
    for meas in measurements:
        ms_dict = {}
        for ms_key in meas.keys():
            if not ms_key in ms_dict.keys():
                ms_dict[ms_key] = {
                    'pos': [],
                    'u': [],
                    'v': [],
                    'w': [],
                    's': [],
                    'tke': []
                    }

            for data in meas[ms_key]:
                # convert measurement coordinates to the terrain frame
                x = data['x'] - measurement_offset[0]
                y = data['y'] - measurement_offset[1]
                z = data['z'] + grid_interpolator((y, x))
                ms_dict[ms_key]['pos'].append([x, y, z])
                ms_dict[ms_key]['s'].append(data['s'])
                ms_dict[ms_key]['u'].append(data['u'])
                ms_dict[ms_key]['v'].append(data['v'])
                ms_dict[ms_key]['w'].append(data['w'])
                if 'tke' in data.keys():
                    ms_dict[ms_key]['tke'].append(data['tke'])
                else:
                    ms_dict[ms_key]['tke'].append(np.nan)

        for ms_id in ms_dict.keys():
            for key in ms_dict[ms_id]:
                if key == 'pos':
                    ms_dict[ms_id][key] = np.vstack(ms_dict[ms_id][key])
                else:
                    ms_dict[ms_id][key] = np.array(ms_dict[ms_id][key])

        measurements_dict.append(ms_dict)

    ds_file = h5py.File('askervein.hdf5', 'w')
    ds_file.create_group('terrain')
    ds_file.create_group('lines')
    ds_file.create_group('measurement')
    for name in experiment_names:
        ds_file['measurement'].create_group(name)

    for i, n in enumerate(args.n_grid):
        for f, terrain, x_inter, y_inter, z_inter in zip(
            args.factors, terrain_grids[i], x_interpolators[i], y_interpolators[i],
            z_interpolators[i]
            ):
            key = 'scale_' + str(f) + '_n_' + str(n)
            ds_file['terrain'].create_dataset(key, data=terrain)

            # convert the prediction lines
            ds_file['lines'].create_group(key)

            # line A
            t = np.linspace(0, 1500, 301)
            start = np.array([414.0, 1080.0]) - measurement_offset  #ASW85
            end = np.array([1262.0, 1975.0]) - measurement_offset  #ANE40
            dir = (end - start)
            dir /= np.linalg.norm(dir)
            positions = np.expand_dims(start,
                                       1) + np.expand_dims(t,
                                                           0) * np.expand_dims(dir, 1)
            z = grid_interpolator((positions[1], positions[0]))
            # minimum distance to ht
            idx_center = np.argmin(
                np.linalg.norm(
                    positions -
                    np.expand_dims((np.array([984, 1695]) - measurement_offset), 1),
                    axis=0
                    )
                )
            t -= idx_center * (t[1] - t[0])
            ds_file['lines'][key].create_group('lineA_10m')
            ds_file['lines'][key]['lineA_10m'].create_dataset(
                'x', data=x_inter(positions[0])
                )
            ds_file['lines'][key]['lineA_10m'].create_dataset(
                'y', data=y_inter(positions[1])
                )
            ds_file['lines'][key]['lineA_10m'].create_dataset(
                'z', data=z_inter(z + 10.0)
                )
            ds_file['lines'][key]['lineA_10m'].create_dataset('terrain', data=z)
            ds_file['lines'][key]['lineA_10m'].create_dataset('dist', data=t)

            # line AA
            t = np.linspace(0, 1500, 301)
            start = np.array([670.0, 778.0]) - measurement_offset  #AASW90
            end = np.array([1674.0, 1844.0]) - measurement_offset  #AANE60
            dir = (end - start)
            dir /= np.linalg.norm(dir)
            positions = np.expand_dims(start,
                                       1) + np.expand_dims(t,
                                                           0) * np.expand_dims(dir, 1)
            z = grid_interpolator((positions[1], positions[0]))
            # minimum distance to ht
            idx_center = np.argmin(
                np.linalg.norm(
                    positions -
                    np.expand_dims((np.array([984, 1695]) - measurement_offset), 1),
                    axis=0
                    )
                )
            t -= idx_center * (t[1] - t[0])
            ds_file['lines'][key].create_group('lineAA_10m')
            ds_file['lines'][key]['lineAA_10m'].create_dataset(
                'x', data=x_inter(positions[0])
                )
            ds_file['lines'][key]['lineAA_10m'].create_dataset(
                'y', data=y_inter(positions[1])
                )
            ds_file['lines'][key]['lineAA_10m'].create_dataset(
                'z', data=z_inter(z + 10.0)
                )
            ds_file['lines'][key]['lineAA_10m'].create_dataset('terrain', data=z)
            ds_file['lines'][key]['lineAA_10m'].create_dataset('dist', data=t)

            # line B
            t = np.linspace(0, 2200, 441)
            start = np.array([2237.0, 543.0]) - measurement_offset  #BSE170
            end = np.array([844.0, 1833.0]) - measurement_offset  #BNW20
            dir = (end - start)
            dir /= np.linalg.norm(dir)
            positions = np.expand_dims(start,
                                       1) + np.expand_dims(t,
                                                           0) * np.expand_dims(dir, 1)
            z = grid_interpolator((positions[1], positions[0]))
            # minimum distance to ht
            idx_center = np.argmin(
                np.linalg.norm(
                    positions -
                    np.expand_dims((np.array([984, 1695]) - measurement_offset), 1),
                    axis=0
                    )
                )
            t -= idx_center * (t[1] - t[0])
            ds_file['lines'][key].create_group('lineB_10m')
            ds_file['lines'][key]['lineB_10m'].create_dataset(
                'x', data=x_inter(positions[0])
                )
            ds_file['lines'][key]['lineB_10m'].create_dataset(
                'y', data=y_inter(positions[1])
                )
            ds_file['lines'][key]['lineB_10m'].create_dataset(
                'z', data=z_inter(z + 10.0)
                )
            ds_file['lines'][key]['lineB_10m'].create_dataset('terrain', data=z)
            ds_file['lines'][key]['lineB_10m'].create_dataset('dist', data=t)

            for meas, name in zip(measurements_dict, experiment_names):
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
    for meas, name in zip(measurements, experiment_names):
        # plot the data
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(X, Y, Z, cmap=cm.terrain)
        scale = 20

        s = []
        u = []
        v = []
        w = []
        x = []
        y = []
        z = []
        for key in meas.keys():
            for data in meas[key]:
                if np.isfinite(data['u']) and np.isfinite(data['v']
                                                          ) and np.isfinite(data['w']):
                    x.append(data['x'] - measurement_offset[0])
                    y.append(data['y'] - measurement_offset[1])
                    z.append(data['z'] + grid_interpolator((y[-1], x[-1])))
                    u.append(data['u'] * scale)
                    v.append(data['v'] * scale)
                    w.append(data['w'] * scale)
                    s.append(data['s'] * scale)

        # Get colors
        c_array = [vel for vel in s]
        for vel in s:
            c_array.extend([vel, vel])
        c_array = np.array(c_array)
        c_array /= c_array.max()
        q = ax.quiver(x, y, z, u, v, w, colors=cm.jet(c_array))

        ax.set_xlim3d([-1000, 1000])
        ax.set_ylim3d([-1000, 1000])
        ax.set_zlim3d([0, 300])
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(name)

    plt.show()
