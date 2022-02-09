import argparse
import copy
import csv
import datetime
import h5py
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
from osgeo import gdal
import re
from scipy.interpolate import LinearNDInterpolator, interp1d
from scipy import ndimage
from scipy.ndimage.filters import uniform_filter
import torch

import pyproj

try:
    import netCDF4 as nc
except ImportError:
    print('Install netCDF4: pip3 install netCDF4')
    exit(1)

tower_positions = {
    'tnw01': {'x': 32611.28, 'y': 4623.17},
    'tnw02': {'x': 32804.29, 'y': 4750.27},
    'tnw03': {'x': 32917.67, 'y': 4860.89},
    'tnw04': {'x': 33158.81, 'y': 4964.16},
    'tnw05': {'x': 33203.54, 'y': 5041.16},
    'tnw06': {'x': 33434.16, 'y': 5224.15},
    'tnw07': {'x': 33587.18, 'y': 5351.36},
    'tnw08': {'x': 33749.53, 'y': 5479.08},
    'tnw09': {'x': 33864.06, 'y': 5587.77},
    'tnw10': {'x': 33952.07, 'y': 5628.10},
    'tnw11': {'x': 34043.02, 'y': 5696.25},
    'tnw12': {'x': 34095.80, 'y': 5717.10},
    'tnw13': {'x': 34140.92, 'y': 5777.09},
    'tnw14': {'x': 34220.92, 'y': 5817.32},
    'tnw15': {'x': 34272.97, 'y': 5843.29},
    'tnw16': {'x': 34349.78, 'y': 5868.89},
    'tse01': {'x': 32960.86, 'y': 3942.30},
    'tse02': {'x': 33260.11, 'y': 4138.28},
    'tse04': {'x': 33394.18, 'y': 4258.87},
    'tse05': {'x': 33539.00, 'y': 4362.00},
    'tse06': {'x': 33636.59, 'y': 4487.36},
    'tse07': {'x': 33820.59, 'y': 4487.36},
    'tse08': {'x': 33977.69, 'y': 4634.04},
    'tse09': {'x': 34153.02, 'y': 4844.78},
    'tse10': {'x': 34274.35, 'y': 4922.95},
    'tse11': {'x': 34334.33, 'y': 4973.22},
    'tse12': {'x': 34448.07, 'y': 5044.25},
    'tse13': {'x': 34533.60, 'y': 5112.01},
    'rsw01': {'x': 33730.49, 'y': 3803.31},
    'rsw02': {'x': 33633.50, 'y': 3901.20},
    'rsw03': {'x': 33569.86, 'y': 4006.84},
    'rsw04': {'x': 33453.60, 'y': 4169.17},
    'rsw05': {'x': 33195.80, 'y': 4548.30},
    'rsw06': {'x': 33087.97, 'y': 4686.07},
    'rsw07': {'x': 32822.82, 'y': 5000.93},
    'rsw08': {'x': 32734.37, 'y': 5141.90},
    'rne01': {'x': 34886.81, 'y': 4772.73},
    'rne02': {'x': 34737.46, 'y': 4877.19},
    'rne03': {'x': 34630.65, 'y': 5025.52},
    'rne04': {'x': 34414.87, 'y': 5281.94},
    'rne06': {'x': 34178.28, 'y': 5565.25},
    'rne07': {'x': 33886.57, 'y': 5852.00},
    'v01': {'x': 34575.06, 'y': 4503.58},
    'v03': {'x': 34235.98, 'y': 4696.84},
    'v04': {'x': 33951.14, 'y': 4978.78},
    'v05': {'x': 33814.75, 'y': 5122.54},
    'v06': {'x': 33704.27, 'y': 5238.11},
    'v07': {'x': 33388.58, 'y': 5457.11}
    }    

def sliding_std(in_arr, window_size):
    c1 = uniform_filter(in_arr, window_size*2, mode='constant', origin = -window_size)
    c2 = uniform_filter(in_arr * in_arr, window_size*2, mode='constant', origin = -window_size)
    return (np.sqrt(c2 - c1*c1))[:-window_size*2 + 1]

def get_terrain_tif(filename):
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    geoTransform = dataset.GetGeoTransform()    
    if not (geoTransform == (2000000.0, 25.0, 0.0, 3000000.0, 0.0, -25.0)):
        print('This function expect the eu_dem_v11_E20N20.tif as an input file')
        exit()

    x_start = 31750
    y_start = 38500
    num_elements = 1000
    Z_geo = dataset.GetRasterBand(1).ReadAsArray(x_start, y_start, num_elements, num_elements)

    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * dataset.RasterXSize
    miny = maxy + geoTransform[5] * dataset.RasterYSize

    x, y = np.meshgrid(np.arange(num_elements) + x_start, np.arange(num_elements) + y_start)
    X_geo = geoTransform[0] + x * geoTransform[1] + y * geoTransform[2]
    Y_geo = geoTransform[3] + x * geoTransform[4] + y * geoTransform[5]

    proj1 = pyproj.Proj(init='epsg:3035')
    proj2 = pyproj.Proj(init='epsg:4326')
    X_wgs84, Y_wgs84, Z_wgs84 = pyproj.transform(proj1, proj2, X_geo, Y_geo, Z_geo)

    terrain_dict = {
        'lat': Y_wgs84,
        'lon': X_wgs84,
        'z': Z_wgs84,
        }
    return terrain_dict

def get_terrain_nc(filename, wgs84_to_local, x_center, y_center):
    # load the terrain data
    terrain_dict = read_nc_file(filename, ['x', 'y', 'z', 'lat', 'lon'])
    x_terrain, y_terrain = wgs84_to_local(terrain_dict['lon'], terrain_dict['lat'])

    # create an interpolator object with a subset of the data
    idx_x = np.argmin(np.abs(x_terrain[0,:] - x_center))
    idx_y = np.argmin(np.abs(y_terrain[:,0] - y_center))

    for key in terrain_dict.keys():
        terrain_dict[key] = terrain_dict[key][idx_y-250:idx_y+250, idx_x-250:idx_x+250]

    return terrain_dict

def read_nc_file(filename, variable_list):
    # open input file, throw warning if it fails
    try:
        ds = nc.Dataset(filename)
    except TypeError:
        print("ERROR: COSMO input file '" + filename + "'is not a NetCDF 3 file!")
        raise IOError
    except IOError as e:
        print("ERROR: COSMO input file '" + filename + "'does not exist!")
        raise e

    # read in all required variables, generate warning if error occurs
    try:
        out = {}
        for variable in variable_list:
            out[variable] = ds[variable][:].copy()
    except:
        print("ERROR: Variable(s) of NetCDF input file '" + filename + "'not valid, at least one variable does not exist. ")
        raise IOError
    ds.close()
    return out

def get_tower_names(filename):
    # open input file, throw warning if it fails
    try:
        ds = nc.Dataset(filename)
    except TypeError:
        print("ERROR: COSMO input file '" + filename + "'is not a NetCDF 3 file!")
        raise IOError
    except IOError as e:
        print("ERROR: COSMO input file '" + filename + "'does not exist!")
        raise e

    tower_keys = [k.split('_')[1] for k in ds.variables.keys() if 'latitude' in k]

    # special treatment for tnw12-tnw16 since they do not have lat/lon reported
    tnw_id = [k.split('_')[-1] for k in ds.variables.keys() if 'm_tnw1' in k]
    towers = ['tnw12', 'tnw13', 'tnw14', 'tnw15', 'tnw16']

    for twr in towers:
        if twr in tnw_id:
            tower_keys.append(twr)

    return tower_keys

def get_tower_heights(filename, twr):
    # open input file, throw warning if it fails
    try:
        ds = nc.Dataset(filename)
    except TypeError:
        print("ERROR: COSMO input file '" + filename + "'is not a NetCDF 3 file!")
        raise IOError
    except IOError as e:
        print("ERROR: COSMO input file '" + filename + "'does not exist!")
        raise e

    twr_keys = [k for k in ds.variables.keys() if twr in k]

    regex = re.compile('u_[0-9].*')
    vel_keys = list(filter(regex.match, twr_keys))

    return [s.split('_')[1] for s in vel_keys]

def get_tower_data(filename, tower_id, heights):
    variables = ['time']
    for height in heights:
        variables.append('u_' + height + '_' + tower_id)
        variables.append('v_' + height + '_' + tower_id)
        variables.append('w_' + height + '_' + tower_id)
        variables.append('spd_' + height + '_' + tower_id)

    return read_nc_file(filename, variables)

def get_time_key(seconds):
    return str(datetime.timedelta(seconds=seconds))

def get_lidar_data(filename):
    data_dict = read_nc_file(filename, ['position_x', 'position_y', 'position_z', 'range', 'time',
                                        'VEL', 'CNR', 'azimuth_angle', 'elevation_angle', 'elevation_sweep'])
    # the data consists of several sweeps
    data_dict['elevation_sweep'][0] = np.nanmax(data_dict['elevation_sweep']) - 0.1 # fix the first nan value
    period = np.argmax(data_dict['elevation_sweep'])
    num_sweeps = np.sum(data_dict['elevation_sweep'] > data_dict['elevation_sweep'].max() - 0.2)


    elevation_angle = np.radians(data_dict['elevation_angle'])[:period]
    try:
        azimuth_angle = np.radians(data_dict['azimuth_angle'])[:period]
    except IndexError:
        # it is a 0 dim array
        azimuth_angle = np.radians(data_dict['azimuth_angle'])

    if period * num_sweeps != len(data_dict['elevation_angle']):
        print('Elevation sweep data for the lidar could not be properly parsed:')
        print(filename)
        exit()

    out_dict = {}
    out_dict['x'] = data_dict['position_x'] + (np.sin(azimuth_angle) * np.cos(elevation_angle))[:, np.newaxis] * data_dict['range']
    out_dict['y'] = data_dict['position_y'] + (np.cos(azimuth_angle) * np.cos(elevation_angle))[:, np.newaxis] * data_dict['range']
    out_dict['z'] = data_dict['position_z'] + np.sin(elevation_angle[:, np.newaxis]) * data_dict['range']
    out_dict['elevation_angle'] = elevation_angle
    out_dict['azimuth_angle'] = azimuth_angle

    vel = np.zeros((num_sweeps, period, len(data_dict['range'])))
    for i in range(num_sweeps):
        mask = data_dict['CNR'][i*period:(i+1)*period] < -20.0
        vel[i] = data_dict['VEL'][i*period:(i+1)*period].copy()
        vel[i][mask] = np.nan

    out_dict['vel'] = np.nanmean(vel, axis=0)
 
    return out_dict

parser = argparse.ArgumentParser(description='Convert the Bolund data from the zip files to the hdf5 format')
parser.add_argument('-t', dest='terrain_file', required=True, help='Filename of the terrain zip file')
parser.add_argument('-m', dest='measurement_file', required=True, help='Filename of the measurement zip file')
parser.add_argument('-l', dest='lidar_folder', default='perdigao_lidar_data', help='Folder containing the lidar data')
parser.add_argument('-p', dest='plot', action='store_true', help='Plot the data from the different cases')
parser.add_argument('-s', dest='save', action='store_true', help='Convert and save the data in different hdf5 datasets')
parser.add_argument('-rh', dest='resolution_hor', type=float, default=16.64, help='The horizontal resolution of the grid')
parser.add_argument('-rv', dest='resolution_ver', type=float, default=11.57, help='The vertical resolution of the grid')
parser.add_argument('-factors', dest='factors', nargs='+', type=float, default=[1.0], help='Downscaling factors')
parser.add_argument('-n', dest='n_grid', nargs='+', type=int, default=[64], help='Number of cells in the grid')
parser.add_argument('-height_ratio', dest='height_ratio', type=float, default=0.5, help='Ratio of cells in z direction compared to the horizontal direction')
parser.add_argument('-o', dest='output_directory', default='', help='The output directory where the converted datasets are stored')

args = parser.parse_args()

# tf into the local frame
wgs84_to_local = pyproj.Proj("+proj=tmerc +lat_0=39.66825833333333 +lon_0=-8.133108333333334 +k=1 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs")

# center of the map 
x_center, y_center = wgs84_to_local(-7.739925, 39.710497)
z_offset = 250.0 # roughtly the height of the lowest tower

# load the terrain data
if args.terrain_file.lower().endswith(('.nc')):
    terrain_dict = get_terrain_nc(args.terrain_file, wgs84_to_local, x_center, y_center)

elif args.terrain_file.lower().endswith(('.tif')):
    terrain_dict = get_terrain_tif(args.terrain_file)

else:
    print('Unknown terrain file type: ', args.terrain_file)
    exit()

x_terrain, y_terrain = wgs84_to_local(terrain_dict['lon'], terrain_dict['lat'])
terrain_interpolator = LinearNDInterpolator(list(zip(x_terrain.ravel() - x_center,
                                                     y_terrain.ravel() - y_center)),
                                                     terrain_dict['z'].ravel() - z_offset)

# load the measurements
tower_data = []
tower_heights = []
tower_names = get_tower_names(args.measurement_file)

for twr in tower_names:
    twr_heights = get_tower_heights(args.measurement_file, twr)
    tower_heights.append(twr_heights)
    tower_data.append(get_tower_data(args.measurement_file, twr, twr_heights))

warning_threshold = 111.0 # m
measurements_dict = {}
for data, heights, twr in zip(tower_data, tower_heights, tower_names):
    times = data['time']
    ms_dict = {'pos': [], 'u': [], 'v': [], 'w': [], 'spd': []}

    x_highres = tower_positions[twr]['x']
    y_highres = tower_positions[twr]['y']
    x_highres -=  x_center
    y_highres -=  y_center

    for ht in heights:
        for key in ['u', 'v', 'w', 'spd']:
            val = data[key + '_' + ht + '_' + twr]
            val[np.ma.getmask(val)] = np.nan
            ms_dict[key].append(val.data)
        z = int(re.sub("[^0-9]", "", ht)) + terrain_interpolator((x_highres, y_highres))
        ms_dict['pos'].append([x_highres, y_highres, z])

    for key in ms_dict:
        if len(ms_dict[key]) > 0:
            ms_dict[key] =  np.vstack(ms_dict[key])

    if len(heights) > 0:
        measurements_dict[twr] = ms_dict


# load the lidar data if available
filename = args.measurement_file.split('/')[-1]
datestring = filename.split('.')[0].split('_')[-1]
lidar_filenames = [f for f in os.listdir(args.lidar_folder) if os.path.isfile(os.path.join(args.lidar_folder, f))]
lidar_data = {}
for lfn in lidar_filenames:
    if datestring in lfn:
        l_data = get_lidar_data(os.path.join(args.lidar_folder, lfn))
        for key in l_data.keys():
            if isinstance(l_data[key], np.ma.MaskedArray):
                l_data[key] = l_data[key].filled(np.nan)
        timestring = lfn.split('.')[0].split('_')[-1].replace(datestring,"")
        scanner_key = lfn.split('.')[0].split('_')[0]
        # one recording takes about 10 minutes, use the middle as the timestamp since the string indicates the end of the recording
        timestamp = int(timestring[0:2]) * 3600 + int(timestring[2:4]) * 60 + int(timestring[4:6]) - 300
        time_idx = np.argmin(np.abs(times - timestamp))
        
        l_data['x'] -= x_center
        l_data['y'] -= y_center
        l_data['z'] -= z_offset

        # set all values occluded by terrain to NAN
        z_terr = terrain_interpolator(l_data['x'], l_data['y'])
        mask = (z_terr + 20) > l_data['z']
        for ray in mask:
            invalid_idx = np.argmax(ray==True)
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
            x_grid = (np.linspace(0,n-1,n) - 0.5 * n) * args.resolution_hor / float(f)
            y_grid = (np.linspace(0,n-1,n) - 0.5 * n) * args.resolution_hor / float(f)
            z_grid = (np.linspace(0,nz-1,nz) - 2.5) * args.resolution_ver / float(f)
            x_int.append(interp1d(x_grid, np.linspace(0,n-1,n), bounds_error=False, fill_value='extrapolate'))
            y_int.append(interp1d(y_grid, np.linspace(0,n-1,n), bounds_error=False, fill_value='extrapolate'))
            z_int.append(interp1d(z_grid, np.linspace(0,nz-1,nz), bounds_error=False, fill_value='extrapolate'))

            X_grid, Y_grid = np.meshgrid(x_grid, y_grid, indexing='xy')
            # check the lower bound of the cell for terrain occlusion
            Z_terrain = terrain_interpolator((Y_grid, X_grid)) - 0.5 * args.resolution_ver / float(f)

            # we need to extend the domain in z temporarily since negative heights can be present in the larger terrain patches
            num_extra_cells = int(np.ceil(np.abs(Z_terrain.min() / args.resolution_ver * float(f))))
            z_grid_tmp = (np.linspace(-num_extra_cells,nz-1,nz+num_extra_cells) - 2.5) * args.resolution_ver / float(f)
            is_free = z_grid_tmp[:, np.newaxis, np.newaxis] > Z_terrain[np.newaxis, :, :]
            dist_field = ndimage.distance_transform_edt(is_free).astype(np.float32)
            tg.append(dist_field[num_extra_cells:])

        terrain_grids.append(tg)
        x_interpolators.append(x_int)
        y_interpolators.append(y_int)
        z_interpolators.append(z_int)

    ds_file = h5py.File('perdigao_' + datestring + '.hdf5', 'w')
    ds_file.create_group('terrain')
    ds_file.create_group('lines')
    ds_file.create_group('measurement')
    ds_file.create_group('lidar')

    for t in times:
        ds_file['measurement'].create_group(get_time_key(t))

    for i in range(24):
        t_key = 'avg_' + str(i) + ':00-' + str(i+1) + ':00'
        ds_file['measurement'].create_group(t_key)

    for i, n in enumerate(args.n_grid):
        for f, terrain, x_inter, y_inter, z_inter in zip(args.factors, terrain_grids[i], x_interpolators[i], y_interpolators[i], z_interpolators[i]):
            s_key = 'scale_' + str(f) + '_n_' + str(n)
            ds_file.create_group(s_key)
            ds_file['terrain'].create_dataset(s_key, data=terrain)

            # Add the mast measurements
            for i in range(len(times)):
                t_key = get_time_key(times[i])
                ds_file['measurement'][t_key].create_group(s_key)

                for ms_post in measurements_dict.keys():
                    ds_file['measurement'][t_key][s_key].create_group(ms_post)

                    for d_key in measurements_dict[ms_post].keys():
                        if d_key == 'pos':
                            pos_idx = copy.copy(measurements_dict[ms_post][d_key])
                            pos_idx[:,0] = x_inter(pos_idx[:,0])
                            pos_idx[:,1] = y_inter(pos_idx[:,1])
                            pos_idx[:,2] = z_inter(pos_idx[:,2])

                            ds_file['measurement'][t_key][s_key][ms_post].create_dataset(d_key, data=pos_idx)
                        elif d_key == 'spd':
                            ds_file['measurement'][t_key][s_key][ms_post].create_dataset('s', data=measurements_dict[ms_post][d_key][:, i])
                        else:
                            ds_file['measurement'][t_key][s_key][ms_post].create_dataset(d_key, data=measurements_dict[ms_post][d_key][:, i])

            # Add the lidar scans
            for i in lidar_data.keys():
                t_key = get_time_key(times[i])

                if not t_key in ds_file['lidar'].keys():
                    ds_file['lidar'].create_group(t_key)

                ds_file['lidar'][t_key].create_group(s_key)

                for lidar_id in lidar_data[i].keys():
                    ds_file['lidar'][t_key][s_key].create_group(lidar_id)
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset('x', data=x_inter(lidar_data[i][lidar_id]['x']))
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset('y', data=y_inter(lidar_data[i][lidar_id]['y']))
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset('z', data=z_inter(lidar_data[i][lidar_id]['z']))
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset('vel', data=lidar_data[i][lidar_id]['vel'])
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset('elevation_angle', data=lidar_data[i][lidar_id]['elevation_angle'])
                    ds_file['lidar'][t_key][s_key][lidar_id].create_dataset('azimuth_angle', data=lidar_data[i][lidar_id]['azimuth_angle'])

            # Add the hourly averaged measurements
            for i in range(24):
                t_key = 'avg_' + str(i) + ':00-' + str(i+1) + ':00'
                ds_file['measurement'][t_key].create_group(s_key)
                for ms_post in measurements_dict.keys():
                    ds_file['measurement'][t_key][s_key].create_group(ms_post)

                    for d_key in measurements_dict[ms_post].keys():
                        if d_key == 'pos':
                            pos_idx = copy.copy(measurements_dict[ms_post][d_key])
                            pos_idx[:,0] = x_inter(pos_idx[:,0])
                            pos_idx[:,1] = y_inter(pos_idx[:,1])
                            pos_idx[:,2] = z_inter(pos_idx[:,2])

                            ds_file['measurement'][t_key][s_key][ms_post].create_dataset(d_key, data=pos_idx)
                        elif d_key == 'spd':
                            val =  np.nanmean(measurements_dict[ms_post][d_key][:,12*i:12*(i+1)], axis=1)
                            ds_file['measurement'][t_key][s_key][ms_post].create_dataset('s', data=val)
                        else:
                            val =  np.nanmean(measurements_dict[ms_post][d_key][:,12*i:12*(i+1)], axis=1)
                            ds_file['measurement'][t_key][s_key][ms_post].create_dataset(d_key, data=val)

    ds_file.close()

if args.plot:
    # plot the data
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x_terrain - x_center,
                           y_terrain - y_center,
                           terrain_dict['z'] - z_offset, cmap=cm.terrain)
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

    print('Maximum average speed encountered in Dataset:', sum_vel[idx]/counter, 'm/s')
    print('Time key of maximum average velocity:', get_time_key(times[idx]))
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
