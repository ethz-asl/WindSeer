import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import pyproj
from pyulog.core import ULog
import os
import nn_wind_prediction.cosmo as cosmo
from nn_wind_prediction.utils.yaml_tools import COSMOParameters, UlogParameters
import argparse
from datetime import datetime
from get_mapgeo_terrain import get_terrain


def quaternion_rotation_matrix(q):
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    R = np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk + qj*qr)],
                  [2*(qi*qj + qk*qr), 1-2*(qi**2+qk**2),  2*(qk*qj-qi*qr)],
                  [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1-2*(qi**2+qj**2)]])
    return R


def slerp(v0, v1, t_array):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    # From Wikipedia: https://en.wikipedia.org/wiki/Slerp
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
    return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :])


def get_quat(data, index):
    return(np.array([data['q[0]'][index], data['q[1]'][index], data['q[2]'][index], data['q[3]'][index]]))


def get_log_data(logfile, proj_logfile, proj_output, skip_amount=10):
    log_data = ULog(logfile).data_list

    print('Message types found:')
    all_names = [log_data[i].name for i in range(len(log_data))]
    for d in log_data:
        message_size = sum([ULog.get_field_size(f.type_str) for f in d.field_data])
        num_data_points = len(d.data['timestamp'])
        name_id = "{:} ({:}, {:})".format(d.name, d.multi_id, message_size)
        print(" {:<40} {:7d} {:10d}".format(name_id, num_data_points,
                                            message_size * num_data_points))

    # Create dictionary because otherwise unsorted
    target_fields = ['wind_estimate', 'vehicle_global_position', 'sensor_hall', 'sensor_hall_01', 'airspeed',
                     'vehicle_attitude', 'vehicle_gps_position']
    data_dict = {}
    for field in target_fields:
        try:
            data_dict[field] = log_data[all_names.index(field)].data
        except ValueError:
            print('WARN: Requested field {0} not found in log_file.'.format(field))

    ulog_data = {}

    ulog_data['gp_time'] = data_dict['vehicle_global_position']['timestamp'][::skip_amount]
    ulog_data['lat'] = data_dict['vehicle_global_position']['lat'][::skip_amount]
    ulog_data['lon'] = data_dict['vehicle_global_position']['lon'][::skip_amount]
    ulog_data['alt_amsl'] = data_dict['vehicle_global_position']['alt'][::skip_amount]
    ulog_data['x'], ulog_data['y'], ulog_data['alt'] = \
        pyproj.transform(proj_logfile, proj_output, ulog_data['lon'], ulog_data['lat'], ulog_data['alt_amsl'])

    ulog_data['vel_d'] = data_dict['vehicle_global_position']['vel_d'][::skip_amount]
    ulog_data['vel_e'] = data_dict['vehicle_global_position']['vel_e'][::skip_amount]
    ulog_data['vel_n'] = data_dict['vehicle_global_position']['vel_n'][::skip_amount]

    # Get the wind estimates at each location in global_pos
    ulog_data['we_east'] = np.interp(ulog_data['gp_time'], data_dict['wind_estimate']['timestamp'], data_dict['wind_estimate']['windspeed_east'])
    ulog_data['we_north'] = np.interp(ulog_data['gp_time'], data_dict['wind_estimate']['timestamp'], data_dict['wind_estimate']['windspeed_north'])
    ulog_data['we_down'] = np.zeros(ulog_data['we_north'].shape)

    # Raw wind estimates from alpha, beta !THIS IS UNCORRECTED!
    alpha_skip = np.interp(ulog_data['gp_time'], data_dict['sensor_hall']['timestamp'], data_dict['sensor_hall']['mag_T'])*np.pi/180.0
    beta_skip = np.interp(ulog_data['gp_time'], data_dict['sensor_hall_01']['timestamp'], data_dict['sensor_hall_01']['mag_T'])*np.pi/180.0
    V_skip =  np.interp(ulog_data['gp_time'], data_dict['airspeed']['timestamp'], data_dict['airspeed']['true_airspeed_m_s'])

    # Body axis velocities
    u = V_skip*np.cos(alpha_skip)*np.cos(beta_skip)
    v = V_skip*np.sin(beta_skip)
    w = V_skip*np.sin(alpha_skip)*np.cos(beta_skip)

    # I NEED INTERPOLATED QUATERNIONS!!
    # Get interpolated time steps
    n_skip = len(ulog_data['gp_time'])
    wn, we, wd = np.zeros(n_skip), np.zeros(n_skip), np.zeros(n_skip)
    for i, st in enumerate(ulog_data['gp_time']):
        tdex = np.sum(st >= data_dict['vehicle_attitude']['timestamp'])
        if tdex > 0 and tdex <= len(data_dict['vehicle_attitude']['timestamp']):
            tl, tu = data_dict['vehicle_attitude']['timestamp'][tdex-1], data_dict['vehicle_attitude']['timestamp'][tdex]
            t_scale = (st - tl) / (tu - tl)
            qq = slerp(get_quat(data_dict['vehicle_attitude'], tdex-1), get_quat(data_dict['vehicle_attitude'], tdex), [t_scale])
        else:
            qq = get_quat(data_dict['vehicle_attitude'], tdex)
        R = quaternion_rotation_matrix(qq.flat)
        Vg = np.matmul(R, np.array([[u[i]], [v[i]], [w[i]]]))
        wn[i] = -(Vg[0] - ulog_data['vel_n'][i])
        we[i] = -(Vg[1] - ulog_data['vel_e'][i])
        wd[i] = -(Vg[2] - ulog_data['vel_d'][i])

    ulog_data['wn'], ulog_data['we'], ulog_data['wd'] = wn, we, wd

    # Get the UTC time offset from GPS time
    ulog_data['utc_microsec'] = np.interp(ulog_data['gp_time'], data_dict['vehicle_gps_position']['timestamp'],
              data_dict['vehicle_gps_position']['time_utc_usec'])

    return ulog_data


def get_colors(uu, vv, ww, cmap=plt.cm.hsv, Vmin = 0.0, Vmax=None):
    # Color by azimuthal angle
    c = np.sqrt(uu**2 + vv**2 + ww**2)
    # Flatten and normalize
    if Vmax is None:
        Vmax = c.ptp()
    c = (c.ravel() - Vmin) / Vmax
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    return cmap(c)


def plot_wind_3d(pos, wind, x_terr, y_terr, h_terr, cosmo_wind, origin=(0.0, 0.0, 0.0)):
    # Plot the wind vector estimates
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Altitude (m)')

    X, Y = np.meshgrid(x_terr-origin[0], y_terr-origin[1])
    ax.plot_surface(X, Y, h_terr-origin[2], cmap=cm.terrain)

    xx, yy, zz = pos[0]-origin[0], pos[1]-origin[1], pos[2]-origin[2]
    # ax.plot(xx, yy, zz)
    ax.quiver(xx, yy, zz, wind[0], wind[1], wind[2], colors=get_colors(wind[0], wind[1], wind[2], Vmax=Vmax))

    # Plot cosmo wind

    ax.plot(cosmo_wind['x'].flatten()-origin[0], cosmo_wind['y'].flatten()-origin[1], cosmo_wind['hsurf'].flatten()-origin[2], 'k.')
    ones_vec = np.ones(cosmo_wind['wind_x'].shape[0])
    for ix in range(2):
        for iy in range(2):
            cwe, cwn, cwu = cosmo_wind['wind_x'][:, ix, iy], cosmo_wind['wind_y'][:, ix, iy], cosmo_wind['wind_z'][:, ix, iy]
            ax.quiver(cosmo_wind['x'][ix, iy]*ones_vec-origin[0], cosmo_wind['y'][ix, iy]*ones_vec-origin[1],
                      cosmo_wind['z'][:, ix, iy] - origin[2], cwe, cwn, cwu, colors=get_colors(cwe, cwn, cwu, Vmax=Vmax))
    norm = matplotlib.colors.Normalize()
    norm.autoscale([0, Vmax])

    sm = matplotlib.cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)
    # ax.set_xlim(xx.min(), xx.max())
    # ax.set_ylim(yy.min(), yy.max())
    # ax.set_zlim(zz.min(), zz.max())
    return fig, ax

def plot_wind_estimates(time, w0, w1, w0_name='W0', w1_name='W1'):
    f2, a2 = plt.subplots(3, 1)
    a2[0].plot(time, w0[0])
    a2[0].plot(time, w1[0])
    a2[0].set_ylabel('$V_E$')
    a2[0].legend([w0_name, w1_name])
    a2[1].plot(time, w0[1])
    a2[1].plot(time, w1[1])
    a2[1].set_ylabel('$V_N$')
    a2[2].plot(time, w0[2])
    a2[2].plot(time, w1[2])
    a2[2].set_ylabel('$V_D$')
    return f2, a2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the wind from a log file and corresponding surrounding COSMO estimates')
    parser.add_argument('-y', '--yaml-file', required=True,
                        help='YAML config file (must contain "cosmo" and "ulog" dictionaries')
    parser.add_argument('-s', '--save-figs', action='store_true', help='Save output figures')
    parser.add_argument('-r', '--resolution', type=int, default=64, help='Extracted block resolution')
    args = parser.parse_args()

    cosmo_args = COSMOParameters(args.yaml_file)
    cosmo_args.print()

    ulog_args = UlogParameters(args.yaml_file)
    ulog_args.print()

    proj_WGS84 = pyproj.Proj(proj='latlong', datum='WGS84')
    proj_EGM96 = pyproj.Proj(init="EPSG:4326", geoidgrids="egm96_15.gtx") # init="EPSG:5773",
    proj_CH_1903_LV03 = pyproj.Proj(init="EPSG:21781")  # https://epsg.io/21781
    proj_CH_1903_LV03_SWISSTOPO = pyproj.Proj(init="CH:1903_LV03")
    proj_CH_1903_LV95 = pyproj.Proj(init="EPSG:2056")
    proj_CH_1903 = pyproj.Proj(init="CH:1903")
    proj_SPHERE = pyproj.Proj(proj='latlong', ellps='sphere', a='6371000')

    ulog_data = get_log_data(ulog_args.params['file'], proj_WGS84, proj_CH_1903_LV03_SWISSTOPO)

    lat0, lon0 = ulog_data['lat'][0], ulog_data['lon'][0]

    # Get cosmo wind
    t0 = datetime.utcfromtimestamp(ulog_data['utc_microsec'][0]/1e6)
    offset_cosmo_time = cosmo_args.get_cosmo_time(t0.hour)
    cosmo_wind = cosmo.extract_cosmo_data(cosmo_args.params['file'], lat0, lon0, offset_cosmo_time,
                                   terrain_file=cosmo_args.params['terrain_file'], cosmo_projection=proj_WGS84,
                                          output_projection=proj_CH_1903_LV03_SWISSTOPO)

    Vmax = np.sqrt((cosmo_wind['wind_x'].flatten() ** 2 + cosmo_wind['wind_y'].flatten() ** 2 + cosmo_wind[
        'wind_z'].flatten() ** 2).max())

    # Get corresponding terrain
    # min_height = min(ulog_data['alt'].min(), h_terr.min())
    block_height = [1100.0/95*63]
    x_terr, y_terr, z_terr, h_terr, full_block = \
        get_terrain(cosmo_args.params['terrain_tiff'], cosmo_wind['x'][[0, 1], [0, 1]], cosmo_wind['y'][[0, 1], [0, 1]],
                    block_height, (args.resolution, args.resolution, args.resolution))

    plane_pos = np.array([ulog_data['x'], ulog_data['y'], ulog_data['alt']])
    w_vanes = np.array([ulog_data['we'], ulog_data['wn'], ulog_data['wd']])
    w_ekfest = np.array([ulog_data['we_east'], ulog_data['we_north'], ulog_data['we_down']])
    fh, ah = plot_wind_3d(plane_pos, w_vanes, x_terr, y_terr, h_terr, cosmo_wind, origin=plane_pos[:,0].flat)
    f2, a2 = plot_wind_estimates(ulog_data['gp_time'], w_ekfest, w_vanes, 'On-board EKF estimate', 'Raw vane estimates')

    # Get corner winds for model inference, offset to actual terrain heights
    terrain_corners = h_terr[::h_terr.shape[0]-1, ::h_terr.shape[1]-1]
    cosmo_corners = cosmo.cosmo_corner_wind(cosmo_wind, z_terr, terrain_height=terrain_corners)

    if args.save_figs:
        bn = os.path.splitext(os.path.basename(ulog_args.params['file']))[0]
        fh.savefig('fig/{0}_wind3d.png'.format(bn), bbox_inches='tight')
        f2.savefig('fig/{0}_wind.png'.format(bn), bbox_inches='tight')
    plt.show(block=False)