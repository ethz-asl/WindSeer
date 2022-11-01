import csv
import datetime
import numpy as np
import os
from osgeo import gdal
import pyproj
import re
from scipy.io import loadmat
import zipfile

try:
    import netCDF4 as nc
except ImportError:
    print('Install netCDF4: pip3 install netCDF4')
    exit(1)


def get_terrain_Askervein(filename):
    '''
    Get the Askervein terrain

    Parameters
    ----------
    filename : str
        Path to the input terrain file

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
    if filename.lower().endswith(('.mat')):
        return get_terrain_Askervein_mat(filename)

    elif filename.lower().endswith(('.tif')):
        return get_terrain_Askervein_tif(filename)

    else:
        print('Unknown terrain file type: ', filename)
        exit()


def get_terrain_Bolund(filename):
    '''
    Get the Bolund terrain

    Parameters
    ----------
    filename : str
        Path to the input terrain file

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
    archive_terrain = zipfile.ZipFile(filename, 'r')
    archive_terrain.extract('Bolund.grd')
    x, y, Z = read_terrain_Bolund_grd('Bolund.grd')
    X, Y = np.meshgrid(x, y, indexing='xy')
    os.remove('Bolund.grd')
    archive_terrain.close()

    terrain_dict = {'x': x, 'y': y, 'X': X, 'Y': Y, 'Z': Z, }

    return terrain_dict


def get_terrain_Perdigao(filename, wgs84_to_local, lon_center, lat_center):
    '''
    Get the Perdigao terrain

    Parameters
    ----------
    filename : str
        Path to the input terrain file
    wgs84_to_local : pyproj.Proj
        Transformation from the WGS84 coordinates to the local coordinates
    lon_center : float
        Longitude of the requested center of the terrain patch
    lat_center : float
        Latitude of the requested center of the terrain patch

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
    if filename.lower().endswith(('.nc')):
        return get_terrain_Perdigao_nc(filename, wgs84_to_local, lon_center, lat_center)

    elif filename.lower().endswith(('.tif')):
        return get_terrain_Perdigao_tif(filename, lon_center, lat_center)

    else:
        print('Unknown terrain file type: ', filename)
        exit()


def get_measurements_Bolund(measurement_file):
    '''
    Get the Bolund measurements

    Parameters
    ----------
    measurement_file : str
        Path to the archive containing the measurements

    Returns
    -------
    measurements : list of np.array
        Measurements from each experiments
    measurements_names : list of list of str
        List of string identifiers of the measurements of each experiment
    names : list of str
        Experiment names
    '''
    archive_meas = zipfile.ZipFile(measurement_file, 'r')
    measurements = []
    measurements_names = []
    meas, meas_name = get_measurement_Bolund_from_archive(archive_meas, 'Dir_90.dat')
    measurements.append(meas)
    measurements_names.append(meas_name)
    meas, meas_name = get_measurement_Bolund_from_archive(archive_meas, 'Dir_239.dat')
    measurements.append(meas)
    measurements_names.append(meas_name)
    meas, meas_name = get_measurement_Bolund_from_archive(archive_meas, 'Dir_255.dat')
    measurements.append(meas)
    measurements_names.append(meas_name)
    meas, meas_name = get_measurement_Bolund_from_archive(archive_meas, 'Dir_270.dat')
    measurements.append(meas)
    measurements_names.append(meas_name)
    names = ['dir_90', 'dir_239', 'dir_255', 'dir_270']

    return measurements, measurements_names, names


def get_terrain_Askervein_mat(filename):
    '''
    Parse the Askervein terrain from the mat file.

    Parameters
    ----------
    filename : str
        Path to the input .mat file

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
    data = loadmat(filename)

    terrain_dict = {
        'x': np.squeeze(data['x']),
        'y': np.squeeze(data['y']),
        'X': data['X'],
        'Y': data['Y'],
        'Z': data['Z'],
        }

    return terrain_dict


def get_terrain_Askervein_tif(filename):
    '''
    Parse the Askervein terrain from a geotiff file.
    Extract a 5x5km patch around the center [75383, 823737] in the local coordinates.

    Parameters
    ----------
    filename : str
        Path to the input .tif file

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
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

    terrain_dict = {'x': X_geo[0], 'y': Y_geo[0], 'X': X_geo, 'Y': Y_geo, 'Z': Z_geo, }

    return terrain_dict


def read_terrain_Bolund_grd(filename):
    '''
    Parse the Bolund terrain from the grd file.

    Parameters
    ----------
    filename : str
        Path to the input .grd file

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        type_str = next(reader)
        nx, ny = [int(v) for v in next(reader)]
        min_x, max_x = [float(v) for v in next(reader)]  # west, east
        min_y, max_y = [float(v) for v in next(reader)]  # north, south
        min_z, max_z = [float(v) for v in next(reader)]
        next(reader)
        Z = np.zeros((nx, ny))
        cx, cy = 0, 0
        for line in reader:
            nnx = len(line)
            Z[cx:(cx + nnx), cy] = [float(v) for v in line]
            cx += nnx
            if cx >= nx - 1:
                cx = 0
                cy += 1

    x, y = np.linspace(min_x, max_x, nx), np.linspace(min_y, max_y, ny)
    return x, y, Z


def read_measurements_Bolund_dat(filename):
    '''
    Parse the Bolund measurements from the .dat file.

    Parameters
    ----------
    filename : str
        Path to the input .dat file

    Returns
    -------
    measurements : np.array
        Array of the measurements
    measurements_names : list of str
        String identifiers of the measurements
    '''
    csv.register_dialect('bolund_measurements', delimiter=' ', skipinitialspace=True)
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile, 'bolund_measurements')
        next(reader)  # skip the headers

        measurements = []
        measurements_names = []
        for line in reader:
            # check if that sensor recorded any measurement
            if float(line[2]) > 0:
                measurements.append([float(v) for v in line[1:]])
                measurements_names.append(line[0])

    measurements = np.array(measurements, dtype=float)
    return measurements, measurements_names


def get_measurement_Bolund_from_archive(archive, filename):
    '''
    Extract the requested file from the archive and read measurements from the extracted file.

    Parameters
    ----------
    filename : str
        Path to the input archive file

    Returns
    -------
    measurements : np.array
        Array of the measurements
    measurements_names : list of str
        String identifiers of the measurements
    '''
    archive.extract(filename)
    measurements, measurements_names = read_measurements_Bolund_dat(filename)
    os.remove(filename)
    return measurements, measurements_names


def get_terrain_Perdigao_tif(filename, lon_center, lat_center):
    '''
    Parse the Perdigao terrain from a geotiff file.
    Extract a 6x6km patch around the requested center.

    Parameters
    ----------
    filename : str
        Path to the input .tif file
    lon_center : float
        Longitude of the requested center of the terrain patch
    lat_center : float
        Latitude of the requested center of the terrain patch

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    geoTransform = dataset.GetGeoTransform()

    Z_geo = dataset.GetRasterBand(1).ReadAsArray()

    x, y = np.meshgrid(
        np.arange(dataset.RasterXSize), np.arange(dataset.RasterYSize), indexing='xy'
        )
    X_geo = geoTransform[0] + x * geoTransform[1] + y * geoTransform[2]
    Y_geo = geoTransform[3] + x * geoTransform[4] + y * geoTransform[5]

    projection_string = dataset.GetProjection().split('PROJ4","')[-1].split('"]')[0]
    proj1 = pyproj.Proj(projection_string)
    proj2 = pyproj.Proj(init='epsg:4326')
    X_wgs84, Y_wgs84, Z_wgs84 = pyproj.transform(proj1, proj2, X_geo, Y_geo, Z_geo)

    # cut the data to an extent of +-3 km around the center
    distance = (X_wgs84 - lon_center)**2 + (Y_wgs84 - lat_center)**2
    idx_center = np.unravel_index(np.argmin(distance), distance.shape)
    size_x_half = int(np.abs(3100.0 / geoTransform[1]))
    size_y_half = int(np.abs(3100.0 / geoTransform[5]))

    terrain_dict = {
        'lat':
            Y_wgs84[idx_center[0] - size_y_half:idx_center[0] + size_y_half,
                    idx_center[1] - size_x_half:idx_center[1] + size_x_half],
        'lon':
            X_wgs84[idx_center[0] - size_y_half:idx_center[0] + size_y_half,
                    idx_center[1] - size_x_half:idx_center[1] + size_x_half],
        'z':
            Z_wgs84[idx_center[0] - size_y_half:idx_center[0] + size_y_half,
                    idx_center[1] - size_x_half:idx_center[1] + size_x_half],
        }

    return terrain_dict


def get_terrain_Perdigao_nc(filename, wgs84_to_local, lon_center, lat_center):
    '''
    Parse the Perdigao terrain from a netcdf file.

    Parameters
    ----------
    filename : str
        Path to the input .tif file
    wgs84_to_local : pyproj.Proj
        Transformation from the WGS84 coordinates to the local coordinates
    lon_center : float
        Longitude of the requested center of the terrain patch
    lat_center : float
        Latitude of the requested center of the terrain patch

    Returns
    -------
    terrain_dict : dict
        Dictionary with the terrain information
    '''
    # load the terrain data
    x_center, y_center = wgs84_to_local(lon_center, lat_center)
    terrain_dict = read_nc_file(filename, ['x', 'y', 'z', 'lat', 'lon'])
    x_terrain, y_terrain = wgs84_to_local(terrain_dict['lon'], terrain_dict['lat'])

    # create an interpolator object with a subset of the data
    idx_x = np.argmin(np.abs(x_terrain[0, :] - x_center))
    idx_y = np.argmin(np.abs(y_terrain[:, 0] - y_center))

    for key in terrain_dict.keys():
        terrain_dict[key] = terrain_dict[key][idx_y - 250:idx_y + 250,
                                              idx_x - 250:idx_x + 250]

    return terrain_dict


def read_nc_file(filename, variable_list):
    '''
    Read data from an netcdf file.

    Parameters
    ----------
    filename : str
        Path to the input .nc file
    variable_list : list of str
        List of the requested variables

    Returns
    -------
    out : dict
        Dictionary with the data
    '''
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
        print(
            "ERROR: Variable(s) of NetCDF input file '" + filename +
            "'not valid, at least one variable does not exist. "
            )
        raise IOError
    ds.close()
    return out


def get_Perdigao_lidar_data(filename):
    '''
    Get the Lidar data from the Perdiago experiment.

    Parameters
    ----------
    filename : str
        Path to the input .nc file

    Returns
    -------
    out : dict
        Dictionary with the data
    '''
    data_dict = read_nc_file(
        filename, [
            'position_x', 'position_y', 'position_z', 'range', 'time', 'VEL', 'CNR',
            'azimuth_angle', 'elevation_angle', 'elevation_sweep'
            ]
        )
    # the data consists of several sweeps
    data_dict['elevation_sweep'][0] = np.nanmax(
        data_dict['elevation_sweep']
        ) - 0.1  # fix the first nan value
    period = np.argmax(data_dict['elevation_sweep'])
    num_sweeps = np.sum(
        data_dict['elevation_sweep'] > data_dict['elevation_sweep'].max() - 0.2
        )

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
    out_dict['x'] = data_dict['position_x'] + (
        np.sin(azimuth_angle) * np.cos(elevation_angle)
        )[:, np.newaxis] * data_dict['range']
    out_dict['y'] = data_dict['position_y'] + (
        np.cos(azimuth_angle) * np.cos(elevation_angle)
        )[:, np.newaxis] * data_dict['range']
    out_dict['z'] = data_dict['position_z'] + np.sin(elevation_angle[:, np.newaxis]
                                                     ) * data_dict['range']
    out_dict['elevation_angle'] = elevation_angle
    out_dict['azimuth_angle'] = azimuth_angle

    vel = np.zeros((num_sweeps, period, len(data_dict['range'])))
    for i in range(num_sweeps):
        mask = data_dict['CNR'][i * period:(i + 1) * period] < -20.0
        vel[i] = data_dict['VEL'][i * period:(i + 1) * period].copy()
        vel[i][mask] = np.nan

    out_dict['vel'] = np.nanmean(vel, axis=0)

    return out_dict


def get_Perdigao_tower_names(filename):
    '''
    Return a list of all available towers.

    Parameters
    ----------
    filename : str
        Path to the input .nc file

    Returns
    -------
    tower_keys : list of str
        Available tower keys
    '''
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


def get_Perdigao_tower_heights(filename, twr):
    '''
    Return a list of all available heights for a specific towers.

    Parameters
    ----------
    filename : str
        Path to the input .nc file
    twr : str
        Key of the tower

    Returns
    -------
    heights : list of str
        Available heights for the requested tower
    '''
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


def get_Perdigao_tower_data(filename, tower_id, heights):
    '''
    Get the measurements for a tower at the requested heights.

    Parameters
    ----------
    filename : str
        Path to the input .nc file
    tower_id : str
        Key of the tower
    heights : list of str
        Key of the requested heights

    Returns
    -------
    data : dict
        Dictionary with the data
    '''
    variables = ['time']
    for height in heights:
        variables.append('u_' + height + '_' + tower_id)
        variables.append('v_' + height + '_' + tower_id)
        variables.append('w_' + height + '_' + tower_id)
        variables.append('spd_' + height + '_' + tower_id)

    return read_nc_file(filename, variables)


def get_Perdigao_time_key(seconds):
    '''
    Get the time key from the seconds elapsed since midnight.

    Parameters
    ----------
    seconds : int
        Time elapsed since midnight

    Returns
    -------
    key : str
        Time key
    '''
    return str(datetime.timedelta(seconds=seconds))


def sliding_std(in_arr, window_size):
    c1 = uniform_filter(in_arr, window_size * 2, mode='constant', origin=-window_size)
    c2 = uniform_filter(
        in_arr * in_arr, window_size * 2, mode='constant', origin=-window_size
        )
    return (np.sqrt(c2 - c1 * c1))[:-window_size * 2 + 1]
