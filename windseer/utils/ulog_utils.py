import numpy as np
import pyproj
import csv
import h5py
import time

from pyulog.core import ULog


def quaternion_to_rotation_matrix(q):
    '''
    Convert a quaternion to a rotation matrix
    
    Parameters
    ----------
    q: np.array
        Input quaternion

    Returns
    -------
    R: np.array
        Rotation matrix
    '''
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    R = np.array(
        [[1 - 2 * (qj**2 + qk**2), 2 * (qi * qj - qk * qr), 2 * (qi * qk + qj * qr)],
         [2 * (qi * qj + qk * qr), 1 - 2 * (qi**2 + qk**2), 2 * (qk * qj - qi * qr)],
         [2 * (qi * qk - qj * qr), 2 * (qj * qk + qi * qr), 1 - 2 * (qi**2 + qj**2)]]
        )
    return R


def slerp(v0, v1, t_array):
    '''
    Interpolate between two quaternions.
    
    >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    From Wikipedia: https://en.wikipedia.org/wiki/Slerp

    Parameters
    ----------
    v1: np.array
        Quaternion 1
    v2: np.array
        Quaternion 2
    t_array: np.array
        Array of the requested interpolation times [0, 1]

    Returns
    -------
    res: np.array
        Interpolated quaternions
    '''
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)
    if (dot < 0.0):
        v1 = -v1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis, :] + t_array[:, np.newaxis] * (v1 - v0)[np.newaxis, :]
        result = result / np.linalg.norm(result)
        return result
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (
        s1[:, np.newaxis] * v1[np.newaxis, :]
        )


def get_quat(data, index):
    '''
    Get the quaternion value at index from the data dict.
    
    Parameters
    ----------
    data: dict
        Dictionary containing the ulog data
    index : int
        Index of the requested quaternion value

    Returns
    -------
    q: np.array
        Quaternion
    '''
    return (
        np.array([
            data['q[0]'][index], data['q[1]'][index], data['q[2]'][index],
            data['q[3]'][index]
            ])
        )


def get_log_data(
        logfile, proj_logfile=None, proj_output=None, skip_amount=10, verbose=False
    ):
    '''
    Get the data from a log file
    
    Parameters
    ----------
    logfile: str
        Path to the ulog file
    proj_logfile : None or str, default : None
        Path to the projection file of the log coordinates, if None the pyproj.Proj(proj='latlong', datum='WGS84') is used
    proj_output : None or str, default : None
        Path to the projection file of the output coordinates, if None the pyproj.Proj(init="CH:1903_LV03") is used
    skip_amount : int, default : 10
        Only keep every nth logged data point.
    verbose : bool, default : False
        Print additional debugging information to the console

    Returns
    -------
    ulog_data: dict
        Dictionary with the log data
    '''
    if proj_logfile is None:
        proj_logfile = pyproj.Proj(proj='latlong', datum='WGS84')
    if proj_output is None:
        proj_output = pyproj.Proj(init="CH:1903_LV03")

    log_data = ULog(logfile).data_list

    if (verbose):
        print('Message types found:')
    all_names = [log_data[i].name for i in range(len(log_data))]
    for d in log_data:
        message_size = sum([ULog.get_field_size(f.type_str) for f in d.field_data])
        num_data_points = len(d.data['timestamp'])
        name_id = "{:} ({:}, {:})".format(d.name, d.multi_id, message_size)
        if (verbose):
            print(
                " {:<40} {:7d} {:10d}".format(
                    name_id, num_data_points, message_size * num_data_points
                    )
                )

    # Create dictionary because otherwise unsorted
    target_fields = [
        'wind_estimate', 'wind', 'vehicle_global_position', 'vehicle_local_position',
        'sensor_hall', 'sensor_hall_01', 'airspeed', 'vehicle_attitude',
        'vehicle_gps_position'
        ]
    data_dict = {}
    for field in target_fields:
        try:
            data_dict[field] = log_data[all_names.index(field)].data
        except ValueError:
            print('WARN: Requested field {0} not found in log_file.'.format(field))

    ulog_data = {}

    ulog_data['gp_time'] = data_dict['vehicle_global_position']['timestamp'
                                                                ][::skip_amount]
    ulog_data['lat'] = data_dict['vehicle_global_position']['lat'][::skip_amount]
    ulog_data['lon'] = data_dict['vehicle_global_position']['lon'][::skip_amount]
    ulog_data['alt_amsl'] = data_dict['vehicle_global_position']['alt'][::skip_amount]
    ulog_data['x'], ulog_data['y'], ulog_data['alt'] = \
        pyproj.transform(proj_logfile, proj_output, ulog_data['lon'], ulog_data['lat'], ulog_data['alt_amsl'])

    # PX4 deleted the vehicle fields at some point from the global position message
    try:
        ulog_data['vel_d'] = data_dict['vehicle_global_position']['vel_d'][::skip_amount
                                                                           ]
        ulog_data['vel_e'] = data_dict['vehicle_global_position']['vel_e'][::skip_amount
                                                                           ]
        ulog_data['vel_n'] = data_dict['vehicle_global_position']['vel_n'][::skip_amount
                                                                           ]
    except KeyError:
        ulog_data['vel_d'] = data_dict['vehicle_local_position']['vx'][::skip_amount]
        ulog_data['vel_e'] = data_dict['vehicle_local_position']['vy'][::skip_amount]
        ulog_data['vel_n'] = data_dict['vehicle_local_position']['vz'][::skip_amount]

    # PX4 changed at some point the name of the wind estimate message
    wind_key = 'wind_estimate'
    if not 'wind_estimate' in data_dict.keys():
        wind_key = 'wind'

    # Get the wind estimates at each location in global_pos
    ulog_data['we_east'] = np.interp(
        ulog_data['gp_time'], data_dict[wind_key]['timestamp'],
        data_dict[wind_key]['windspeed_east']
        )
    ulog_data['we_north'] = np.interp(
        ulog_data['gp_time'], data_dict[wind_key]['timestamp'],
        data_dict[wind_key]['windspeed_north']
        )
    ulog_data['we_down'] = np.zeros(ulog_data['we_north'].shape)

    # Raw wind estimates from alpha, beta !THIS IS UNCORRECTED!
    V_skip = np.interp(
        ulog_data['gp_time'], data_dict['airspeed']['timestamp'],
        data_dict['airspeed']['true_airspeed_m_s']
        )
    try:
        alpha_skip = np.interp(
            ulog_data['gp_time'], data_dict['sensor_hall']['timestamp'],
            data_dict['sensor_hall']['mag_T']
            ) * np.pi / 180.0
    except KeyError:
        print("Alpha vane values not found!!")
        alpha_skip = np.zeros(V_skip.shape)
    try:
        beta_skip = np.interp(
            ulog_data['gp_time'], data_dict['sensor_hall_01']['timestamp'],
            data_dict['sensor_hall_01']['mag_T']
            ) * np.pi / 180.0
    except KeyError:
        print("Beta vane values not found!!")
        beta_skip = np.zeros(V_skip.shape)

    # Body axis velocities
    u = V_skip * np.cos(alpha_skip) * np.cos(beta_skip)
    v = V_skip * np.sin(beta_skip)
    w = V_skip * np.sin(alpha_skip) * np.cos(beta_skip)

    # I NEED INTERPOLATED QUATERNIONS!!
    # Get interpolated time steps
    n_skip = len(ulog_data['gp_time'])
    wn, we, wd = np.zeros(n_skip), np.zeros(n_skip), np.zeros(n_skip)
    for i, st in enumerate(ulog_data['gp_time']):
        tdex = np.sum(st >= data_dict['vehicle_attitude']['timestamp'])
        if tdex > 0 and tdex <= len(data_dict['vehicle_attitude']['timestamp']):
            tl, tu = data_dict['vehicle_attitude']['timestamp'][
                tdex - 1], data_dict['vehicle_attitude']['timestamp'][tdex]
            t_scale = (st - tl) / (tu - tl)
            qq = slerp(
                get_quat(data_dict['vehicle_attitude'], tdex - 1),
                get_quat(data_dict['vehicle_attitude'], tdex), [t_scale]
                )
        else:
            qq = get_quat(data_dict['vehicle_attitude'], tdex)
        R = quaternion_to_rotation_matrix(qq.flat)
        Vg = np.matmul(R, np.array([[u[i]], [v[i]], [w[i]]]))
        wn[i] = -(Vg[0] - ulog_data['vel_n'][i])
        we[i] = -(Vg[1] - ulog_data['vel_e'][i])
        wd[i] = -(Vg[2] - ulog_data['vel_d'][i])

    ulog_data['wn'], ulog_data['we'], ulog_data['wd'] = wn, we, wd

    # Get the UTC time offset from GPS time
    ulog_data['utc_microsec'] = np.interp(
        ulog_data['gp_time'], data_dict['vehicle_gps_position']['timestamp'],
        data_dict['vehicle_gps_position']['time_utc_usec']
        )

    return ulog_data


def build_csv(
    x_terr, y_terr, z_terr, full_block, cosmo_corners, outfile, origin=(0.0, 0.0, 0.0)
    ):
    '''
    Build a CSV file of the terrain and cosmo corner wind data.

    Parameters
    ----------
    x_terr: np.array
        Array of the x-positions of the terrain
    y_terr: np.array
        Array of the y-positions of the terrain
    z_terr: np.array
        Array of the z-positions of the terrain
    full_block: np.array
        Occupancy mask of the terrain
    cosmo_corners: np.array
        Cosmo corner winds
    outfile: np.array
        Path to the output file
    origin: tuple, default : (0, 0, 0)
        Position offset of the origin
    '''
    # Make a zero array for the wind
    s = [3]
    s.extend([n for n in full_block.shape])
    full_wind = np.zeros(s, dtype='float')  # full_wind should be [3, z, y, x]
    full_wind[:, :, ::full_block.shape[0] - 1, ::full_block.shape[1] -
              1] = cosmo_corners
    with open(outfile, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        types = [
            "p", "U:0", "U:1", "U:2", "epsilon", "k", "nut", "vtkValidPointMask",
            "Points:0", "Points:1", "Points:2"
            ]
        base = np.zeros(len(types))
        csv_writer.writerow(types)
        for k, zt in enumerate(z_terr - origin[2]):
            base[types.index("Points:2")] = zt
            for j, yt in enumerate(y_terr - origin[1]):
                base[types.index("Points:1")] = yt
                for i, xt in enumerate(x_terr - origin[0]):
                    base[types.index("Points:0")] = xt
                    base[types.index("U:0")] = full_wind[0, k, j, i]
                    base[types.index("U:1")] = full_wind[1, k, j, i]
                    base[types.index("U:2")] = full_wind[2, k, j, i]
                    base[types.index("vtkValidPointMask")
                         ] = (not full_block[k, j, i]) * 1.0
                    csv_writer.writerow(base)


def read_filtered_hdf5(filename, proj_logfile=None, proj_output=None, skip_amount=1):
    '''
    Get the data from a log file
    
    Parameters
    ----------
    filename: str
        Path to the hdf5 file
    proj_logfile : None or str, default : None
        Path to the projection file of the log coordinates, if None the pyproj.Proj(proj='latlong', datum='WGS84') is used
    proj_output : None or str, default : None
        Path to the projection file of the output coordinates, if None the pyproj.Proj(init="CH:1903_LV03") is used
    skip_amount : int, default : 1
        Only keep every nth logged data point.

    Returns
    -------
    out_dict: dict
        Dictionary with the log data
    '''
    # default projections
    if proj_logfile is None:
        proj_logfile = pyproj.Proj(proj='latlong', datum='WGS84')
    if proj_output is None:
        proj_output = pyproj.Proj(init="CH:1903_LV03")

    # extract the data
    out_dict = {}
    f = h5py.File(filename, 'r')
    for key, value in f['wind_out'].items():
        out_dict[key] = np.array(value).squeeze()[::skip_amount]
    # Convert time stamps back to uint64 microseconds
    out_dict['time'] = (out_dict['time'] * 1e6).astype('int64')

    out_dict['alt_amsl'] = out_dict['alt']

    out_dict['x'], out_dict['y'], out_dict['alt'] = \
        pyproj.transform(proj_logfile, proj_output, out_dict['lon'], out_dict['lat'], out_dict['alt_amsl'])

    return out_dict


def extract_wind_data(filename, use_estimate):
    '''
    Get the data from a log file. Loading data from
    a ulog or hdf5 file is supported.
    
    Parameters
    ----------
    filename: str
        Path to the data file
    use_estimate : bool
        In case of loading the data from a ulog file this determines if the wind estimate is used (True) or the wind calculated from the wind triangle (False)

    Returns
    -------
    ulog_data: dict
        Dictionary with the wind data
    '''
    # import the wind data
    t_start = time.time()
    file_ending = filename.split('.')[-1]
    if file_ending == 'ulg':
        print('Importing wind data from ulog file...', end='', flush=True)
        ulog_data = get_log_data(filename)
        wind_data = {
            'time': ulog_data['gp_time'],
            'lat': ulog_data['lat'],
            'lon': ulog_data['lon'],
            'alt_amsl': ulog_data['alt_amsl'],
            'x': ulog_data['x'],
            'y': ulog_data['y'],
            'alt': ulog_data['alt'],
            'time_gps': None,
            }
        if (use_estimate):
            wind_data['we'] = ulog_data['we_east']
            wind_data['wn'] = ulog_data['we_north']
            wind_data['wd'] = ulog_data['we_down']

        else:
            wind_data['we'] = ulog_data['we']
            wind_data['wn'] = ulog_data['wn']
            wind_data['wd'] = ulog_data['wd']

        del ulog_data

    elif file_ending == 'hdf5':
        print('Importing wind data from hdf5 file...', end='', flush=True)
        hdf5_data = read_filtered_hdf5(filename)

        wind_data = {
            'time': hdf5_data['time'],
            'lat': hdf5_data['lat'],
            'lon': hdf5_data['lon'],
            'alt_amsl': hdf5_data['alt_amsl'],
            'x': hdf5_data['x'],
            'y': hdf5_data['y'],
            'alt': hdf5_data['alt'],
            'we': hdf5_data['wind_e'],
            'wn': hdf5_data['wind_n'],
            'wd': hdf5_data['wind_d'],
            'time_gps': hdf5_data['time_gps'],
            }

        del hdf5_data

    else:
        print('Error: Unknown file type:', file_ending)
        exit()

    print(' done [{:.2f} s]'.format(time.time() - t_start))
    return wind_data


def filter_wind_data(wind_data, filter_size):
    '''
    Filter the wind data with a moving average filter.
    
    Parameters
    ----------
    wind_data: dict
        Dictionary with the wind data
    filter_size : int
        Window size of the filter, has to be uneven

    Returns
    -------
    wind_out: dict
        Dictionary with the filtered wind data
    '''
    if filter_size % 2 == 0:
        raise ValueError('The filter size has to be uneven')

    skip = int((filter_size - 1) * 0.5)
    wind_out = {}
    for key in wind_data:
        if not wind_data[key] is None:
            wind_out[key] = wind_data[key][skip:-skip]

    wind_out['we_raw'] = wind_out['we']
    wind_out['wn_raw'] = wind_out['wn']
    wind_out['wd_raw'] = wind_out['wd']

    wind_out['we'] = np.convolve(
        wind_data['we'], np.ones(filter_size) / filter_size, mode='valid'
        )
    wind_out['wn'] = np.convolve(
        wind_data['wn'], np.ones(filter_size) / filter_size, mode='valid'
        )
    wind_out['wd'] = np.convolve(
        wind_data['wd'], np.ones(filter_size) / filter_size, mode='valid'
        )

    return wind_out
