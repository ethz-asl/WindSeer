#!/usr/bin/env python
from __future__ import print_function

try:
    import numpy as np
except ImportError:
    print("'numpy' is not installed. Use the command below to install it:")
    print("     sudo apt-get install python-numpy")

try:
    from scipy.io import netcdf
except ImportError:
    print("'scipy' is not installed. Use the command below to install it:")
    print("     sudo apt-get install python-scipy")

try:
    import pyproj
except ImportError:
    print("'pyproj' is not installed. Use the command below to install it:")
    print("     pip3 install pyproj")


def read_cosmo_nc_file(filename, variable_list):
    '''
    Read data from an nc file.

    Parameters
    ----------
    filename : str
        Filename of the nc file
    variable_list : list of str
        Requested variables to be read from the nc file

    Returns
    -------
    out : list
        List of the data
    '''

    # open input file, throw warning if it fails
    file_err = False
    try:
        f = netcdf.netcdf_file(filename, "r")
    except TypeError:
        print("ERROR: COSMO input file '" + filename + "'is not a NetCDF 3 file!")
        raise IOError
    except IOError as e:
        print("ERROR: COSMO input file '" + filename + "'does not exist!")
        raise e

    # read in all required variables, generate warning if error occurs
    try:
        out = []
        for variable in variable_list:
            out.append(f.variables[variable][:].copy())
    except:
        print(
            "ERROR: Variable(s) of NetCDF input file '" + filename +
            "'not valid, at least one variable does not exist. "
            )
        raise IOError
    f.close()
    return out


def rotate_and_scale_wind(wx, wy, angle=0.0, scale=1.0):
    '''
    Rotate and scale the wind.

    Parameters
    ----------
    wx : np.array
        Array of wind velocities in x-direction
    wy : np.array
        Array of wind velocities in y-direction
    angle : float, default : 0.0
        Requested yaw rotation angle in rad
    scale : float, default : 1.0
        Requested scale change of the wind

    Returns
    -------
    wx_out : np.array
        Scaled and rotated wind velocities in x-direction
    wy_out : np.array
        Scaled and rotated wind velocities in y-direction
    '''
    wx2 = scale * (wx * np.cos(angle) - wy * np.sin(angle))
    wy2 = scale * (wx * np.sin(angle) + wy * np.cos(angle))
    return wx2, wy2


def above_line_check(x1, y1, x2, y2, x_test, y_test):
    '''
    Check if the test point is above a line between two points

    Parameters
    ----------
    x1 : float
        X-position of the first point on the line
    y1 : float
        Y-position of the first point on the line
    x2 : float
        X-position of the second point on the line
    y2 : float
        Y-position of the second point on the line
    x_test : float
        X-position of the test point
    y_test : float
        Y-position of the test point

    Returns
    -------
    ret : bool
        True if the test point is above the line, else False
    '''
    # fit a line y = a * x + b
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1

    if a * x_test + b < y_test:
        return True
    else:
        return False


def determine_cell_indices(
        x_grid, y_grid, x_cell, y_cell, lat, lon, lat_requested, lon_requested
    ):
    '''
    Determine the cell index of the cosmo cell for a given location.

    Parameters
    ----------
    x_grid : np.array
        Array of the x-positions in the local frame of the cosmo grid
    y_grid : np.array
        Array of the x-positions in the local frame of the cosmo grid
    x_cell : float
        Requested x-location in the local frame
    y_cell : float
        Requested y-location in the local frame
    lat : np.array
        Array of the WGS84 latitude coordinates of the cosmo gri
    lon : np.array
        Array of the WGS84 longitude coordinates of the cosmo grid
    lat_requested : float
        Latitude of the test point
    lon_requested : float
        Longitude of the test point

    Returns
    -------
    slice_x : slice
        Slice in the x-axis
    slice_y : slice
        Slice in the y-axis
    '''

    # find the closest gridpoint
    dist = (x_cell - x_grid)**2 + (y_cell - y_grid)**2
    idx_r, idx_c = np.unravel_index(dist.argmin(), dist.shape)

    if (np.sqrt(dist[idx_r, idx_c]) > x_grid[0, 1] - x_grid[0, 0]):
        print(
            'ERROR: The distance to the closest gridpoint (',
            np.sqrt(dist[idx_r, idx_c]), ') is larger than the grid resolution:',
            x_grid[0, 1] - x_grid[0, 0]
            )
        exit()

    # check in which quadrant the requested point lies with respect to the closest gridpoint
    #      c3
    #  II  | I
    # -c2--*-c1--
    #  III | IV
    #      c4
    quadrant_mask = np.zeros((2, 2), dtype=bool)

    check1 = above_line_check(
        lon[idx_r, idx_c], lat[idx_r, idx_c], lon[idx_r, idx_c + 1],
        lat[idx_r, idx_c + 1], lon_requested, lat_requested
        )
    check2 = above_line_check(
        lon[idx_r, idx_c], lat[idx_r, idx_c], lon[idx_r, idx_c - 1],
        lat[idx_r, idx_c - 1], lon_requested, lat_requested
        )
    check3 = above_line_check(
        lat[idx_r, idx_c], lon[idx_r, idx_c], lat[idx_r + 1, idx_c],
        lon[idx_r + 1, idx_c], lat_requested, lon_requested
        )
    check4 = above_line_check(
        lat[idx_r, idx_c], lon[idx_r, idx_c], lat[idx_r - 1, idx_c],
        lon[idx_r - 1, idx_c], lat_requested, lon_requested
        )

    # quadrant II
    quadrant_mask[0, 0] = check2 and not check3

    # quadrant I
    quadrant_mask[0, 1] = check1 and check3

    # quadrant III
    quadrant_mask[1, 0] = not check2 and not check4

    # quadrant IV
    quadrant_mask[1, 1] = not check1 and check4

    # confirm that only one quadrant is true
    if quadrant_mask.sum() != 1:
        print('ERROR: Could not uniquely determine correct cell to extract')
        exit()

    # extract the data
    if quadrant_mask[0, 1]:
        # Quadrant I
        slice_x = slice(idx_c, idx_c + 2)
        slice_y = slice(idx_r, idx_r + 2)

    elif quadrant_mask[0, 0]:
        # Quadrant II
        slice_x = slice(idx_c - 1, idx_c + 1)
        slice_y = slice(idx_r, idx_r + 2)

    elif quadrant_mask[1, 0]:
        # Quadrant III
        slice_x = slice(idx_c - 1, idx_c + 1)
        slice_y = slice(idx_r - 1, idx_r + 1)

    elif quadrant_mask[1, 1]:
        # Quadrant IV
        slice_x = slice(idx_c, idx_c + 2)
        slice_y = slice(idx_r - 1, idx_r + 1)

    else:
        print('ERROR: Could not determine correct cell to extract')
        exit

    return slice_x, slice_y


def extract_cosmo_data(
    filename,
    lat_requested,
    lon_requested,
    time_requested,
    terrain_file=None,
    cosmo_projection=pyproj.Proj(proj='latlong', datum='WGS84'),
    output_projection=pyproj.Proj(init="CH:1903_LV03")
    ):
    '''
    Opens the requested COSMO NetCDF file and extracts all wind profiles that are required to calculate the
    initial wind field for the complete meteo grid domain.

    Parameters
    ----------
    filename : str
        Path to the cosmo nc file
    lat_requested : float
        Requested location, latitude in deg
    lon_requested : float
        Requested location, longitude in deg
    time_requested : int
        Requested time in hours of the forecast
    terrain_file : str or None, default : None
        Path to the terrain file, if None the terrain is assumed to contained in the cosmo nc file
    cosmo_projection : pyproj.Proj, default : pyproj.Proj(proj='latlong', datum='WGS84')
        Projection of for the Cosmo frame
    output_projection : pyproj.Proj, default : pyproj.Proj(init="CH:1903_LV03")
        Projection of for the local output frame

    Returns
    -------
    out : dict
        Dictionary with the wind and topography data
    '''

    if terrain_file is None:
        terrain_file = filename
    # create a dummy output
    out = {}
    out['valid'] = False

    try:
        time, lon, lat, z_bnd, u_c, v_c, w_c = read_cosmo_nc_file(
            filename, ['time', 'lon_1', 'lat_1', 'z_bnds_1', 'U', 'V', 'W']
            )
        hsurf, hfl = read_cosmo_nc_file(terrain_file, ['HSURF', 'HFL'])

    except IOError as e:
        print('ERROR: Data file read failed')
        return out

    time_true = time == time_requested * 3600
    if (sum(time_true * 1) != 1):
        print("ERROR: Requested COSMO hour invalid!")
        return out
    t = sum(time_true * np.arange(0, time.shape[0], 1))

    # convert to output coordinate projection
    x_cell, y_cell = pyproj.transform(
        cosmo_projection, output_projection, lon_requested, lat_requested
        )
    x_grid, y_grid, h_grid = pyproj.transform(
        cosmo_projection, output_projection, lon, lat, hsurf
        )

    # e_cell, n_cell, zone_num0, zone_letter0 = utm.from_latlon(lat_requested, lon_requested)
    # e_grid, n_grid, zone_num, zone_letter = utm.from_latlon(lat, lon, force_zone_number=zone_num0, force_zone_letter=zone_letter0)

    slice_x, slice_y = determine_cell_indices(
        x_grid, y_grid, x_cell, y_cell, lat, lon, lat_requested, lon_requested
        )

    # determine the correct vertical slices to extract
    # TODO extract the correct vertical slices
    z_start = 0
    z_stop = u_c.shape[1]
    slice_z = slice(z_start, z_stop)

    out['lat'] = lat[slice_y, slice_x]
    out['lon'] = lon[slice_y, slice_x]
    out['x'] = x_grid[slice_y, slice_x]
    out['y'] = y_grid[slice_y, slice_x]
    out['wind_x'] = u_c[t, slice_z, slice_y, slice_x]
    out['wind_y'] = v_c[t, slice_z, slice_y, slice_x]
    out['wind_z'] = w_c[t, slice_z, slice_y, slice_x]
    out['hsurf'] = h_grid[slice_y, slice_x]  # These are on the CH1903 (like x and y)

    # Note that the hfl altitudes are incorrect (WGS84, need to convert to CH1903)
    out['z'] = hfl[slice_z, slice_y, slice_x]
    for i, hfli in enumerate(out['z']):
        _x, _y, hi_ch = pyproj.transform(
            cosmo_projection, output_projection, out['lon'], out['lat'], hfli
            )
        out['z'][i] = hi_ch

    out['valid'] = True
    return out


def cosmo_corner_wind(cosmo_data, z_target, terrain_height=None, rotate=0.0, scale=1.0):
    '''
    Get winds at cosmo corners for specified z heights
    If terrain_height is specified, the winds will be offset to start at the heights specified in the terrain_height
    Otherwise, we just sample directly (heights may not match)
    NOTE: cosmo_wind is from top to bottom (i.e. first element is max height, down to surface at end)...
    BUT: Output is bottom to top

    Output wind dimensions: [3, nz, ny, nx]
    First dimension is wind speeds, in order W_East(0), W_North(1), W_Up(2)
    Opens the requested COSMO NetCDF file and extracts all wind profiles that are required to calculate the
    initial wind field for the complete meteo grid domain.

    Parameters
    ----------
    cosmo_data : dict
        Dictionary with the wind and topography data
    z_target : np.array
        Array of the requested altitudes
    terrain_height : str or None, default : None
        Array with the terrain heights
    rotate : float, default : 0.0
        Heading rotation applied to the wind in rad
    scale : float, default : 1.0
        scale factor applied to the wind

    Returns
    -------
    out : dict
        Dictionary with the wind and topography data
    '''

    nz, ny, nx = cosmo_data['z'].shape
    winds = np.zeros((3, len(z_target), ny, nx), dtype='float')

    for yi in range(ny):
        for xi in range(nx):
            # I append a point at the surface with zero wind speed for properly interpolating from the ground
            cosmo_z = np.append([cosmo_data['hsurf'][yi, xi]], cosmo_data['z'][::-1, yi,
                                                                               xi])
            cosmo_wx = np.append([0.0], cosmo_data['wind_x'][::-1, yi, xi])
            cosmo_wy = np.append([0.0], cosmo_data['wind_y'][::-1, yi, xi])
            cosmo_wx, cosmo_wy = rotate_and_scale_wind(
                cosmo_wx, cosmo_wy, angle=rotate, scale=scale
                )
            cosmo_wz = np.append([0.0], cosmo_data['wind_z'][::-1, yi, xi])
            if terrain_height is not None:
                cosmo_z = cosmo_z - (cosmo_z[0] - terrain_height[yi, xi])
            valid_z = (z_target >= cosmo_z[0])
            winds[0, valid_z, yi, xi] = np.interp(z_target[valid_z], cosmo_z, cosmo_wx)
            winds[1, valid_z, yi, xi] = np.interp(z_target[valid_z], cosmo_z, cosmo_wy)
            winds[2, valid_z, yi, xi] = np.interp(z_target[valid_z], cosmo_z, cosmo_wz)

    return winds


def get_cosmo_cell(
    filename,
    lat_requested,
    lon_requested,
    z_min,
    d_hor,
    d_ver,
    cosmo_projection=pyproj.Proj(proj='latlong', datum='WGS84'),
    output_projection=pyproj.Proj(init="CH:1903_LV03")
    ):
    '''
    Get the extent of the cosmo cell containing the requested lat/lon position

    Parameters
    ----------
    filename : str
        Path to the Cosmo terrain file
    lat_requested : float
        Requested longitude
    lat_requested : float
        Requested latitude
    z_min : float
        Minimum altitude
    d_hor : float
        Horizontal extent of the cell
    d_ver : float
        Vertical extent of the cell
    cosmo_projection : pyproj.Proj, default : pyproj.Proj(proj='latlong', datum='WGS84')
        Projection of for the Cosmo frame
    output_projection : pyproj.Proj, default : pyproj.Proj(init="CH:1903_LV03")
        Projection of for the local output frame

    Returns
    -------
    cell : dict
        Dictionary with the cell extent
    '''
    try:
        lon, lat, hsurf, hfl = read_cosmo_nc_file(
            filename, ['lon_1', 'lat_1', 'HSURF', 'HFL']
            )

    except IOError as e:
        print('ERROR: Data file read failed')
        raise e

    x_cell, y_cell = pyproj.transform(
        cosmo_projection, output_projection, lon_requested, lat_requested
        )
    x_grid, y_grid, h_grid = pyproj.transform(
        cosmo_projection, output_projection, lon, lat, hsurf
        )

    slice_x, slice_y = determine_cell_indices(
        x_grid, y_grid, x_cell, y_cell, lat, lon, lat_requested, lon_requested
        )

    x = x_grid[slice_y, slice_x]
    y = y_grid[slice_y, slice_x]

    corners = {
        'x_min': x[0][0],
        'x_max': x[0][0] + d_hor,
        'y_min': y[0][0],
        'y_max': y[0][0] + d_hor,
        'z_min': z_min,
        'z_max': z_min + d_ver,
        }

    return corners
