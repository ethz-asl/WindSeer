import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
import utm
from pyulog.core import ULog
import os
import nn_wind_prediction.cosmo as cosmo

savefig=False
logfile = 'logs/11_11_57.ulg'
cosmo_file = 'data/riemenstalden/cosmo-1_ethz_fcst_2018112312.nc'
cosmo_time = 0

def rpy(roll, pitch, yaw):
    ra = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), -np.cos(roll), 0],
                   [ 0, 0, 1]])
    rb = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [ 0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])


def quaternion_rotation_matrix(q):
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    R = np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk + qj*qr)],
                  [2*(qi*qj + qk*qr), 1-2*(qi**2+qk**2),  2*(qk*qj-qi*qr)],
                  [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1-2*(qi**2+qj**2)]])
    return R

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

def slerp(v0, v1, t_array):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
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


log_data = ULog(logfile).data_list

print('Message types found:')
for d in log_data:
    message_size = sum([ULog.get_field_size(f.type_str) for f in d.field_data])
    num_data_points = len(d.data['timestamp'])
    name_id = "{:} ({:}, {:})".format(d.name, d.multi_id, message_size)
    print(" {:<40} {:7d} {:10d}".format(name_id, num_data_points,
                                        message_size * num_data_points))

# Create dictionary because otherwise unsorted
target_data = ULog(logfile, 'wind_estimate,vehicle_global_position,sensor_hall,sensor_hall_01,airspeed,vehicle_attitude').data_list
data_dict = {}
for d in target_data:
    data_dict[d.name] = d.data

gp_time = data_dict['vehicle_global_position']['timestamp']
lat, lon = data_dict['vehicle_global_position']['lat'], data_dict['vehicle_global_position']['lon']
x0,y0, zone_num0, zone_letter0 = utm.from_latlon(lat[0], lon[0])
x, y, zone_num, zone_letter = utm.from_latlon(lat, lon, force_zone_number=zone_num0, force_zone_letter=zone_letter0)
alt = data_dict['vehicle_global_position']['alt']
vel_d = data_dict['vehicle_global_position']['vel_d']
vel_e = data_dict['vehicle_global_position']['vel_e']
vel_n = data_dict['vehicle_global_position']['vel_n']

skip_amount = 10
skip_time = gp_time[::skip_amount]
skip_vel_n = vel_n[::skip_amount]
skip_vel_e = vel_e[::skip_amount]
skip_vel_d = vel_d[::skip_amount]
# Get the wind estimates at each location in global_pos
we_east = np.interp(skip_time, data_dict['wind_estimate']['timestamp'], data_dict['wind_estimate']['windspeed_east'])
we_north = np.interp(skip_time, data_dict['wind_estimate']['timestamp'], data_dict['wind_estimate']['windspeed_north'])
we_up = np.zeros(we_north.shape)

# Raw wind estimates from alpha, beta !THIS IS UNCORRECTED!
alpha_skip = np.interp(skip_time, data_dict['sensor_hall']['timestamp'], data_dict['sensor_hall']['mag_T'])*np.pi/180.0
beta_skip = np.interp(skip_time, data_dict['sensor_hall_01']['timestamp'], data_dict['sensor_hall_01']['mag_T'])*np.pi/180.0
V_skip =  np.interp(skip_time, data_dict['airspeed']['timestamp'], data_dict['airspeed']['true_airspeed_m_s'])

# Body axis velocities
u = V_skip*np.cos(alpha_skip)*np.cos(beta_skip)
v = V_skip*np.sin(beta_skip)
w = V_skip*np.sin(alpha_skip)*np.cos(beta_skip)

# I NEED INTERPOLATED QUATERNIONS!!
# Get interpolated time steps
n_skip = len(skip_time)
wn, we, wd = np.zeros(n_skip), np.zeros(n_skip), np.zeros(n_skip)
for i, st in enumerate(skip_time):
    tdex = np.sum(st > data_dict['vehicle_attitude']['timestamp'])
    # if tdex < len(data_dict['vehicle_attitude']['timestamp']):
    #     tu, tl = data_dict['vehicle_attitude']['timestamp'][tdex+1], data_dict['vehicle_attitude']['timestamp'][tdex]
    #     t_scale = (st - tl) / (tu - tl)
    #     qq = slerp(get_quat(data_dict['vehicle_attitude'], tdex), get_quat(data_dict['vehicle_attitude'], tdex+1), [t_scale])
    # else:

    qq = get_quat(data_dict['vehicle_attitude'], tdex)
    R = quaternion_rotation_matrix(qq.flat)
    Vg = np.matmul(R, np.array([[u[i]], [v[i]], [w[i]]]))
    wn[i] = (Vg[0] - skip_vel_n[i])
    we[i] = (Vg[1] - skip_vel_e[i])
    wd[i] = (Vg[2] - skip_vel_d[i])

# For some reason they are negative... check signs
wn, we = -wn, -we

# Get cosmo wind
cosmo_wind = cosmo.extract_cosmo_data(cosmo_file, lat[0], lon[0], cosmo_time,
                               terrain_file='data/riemenstalden/cosmo-1_ethz_ana/cosmo-1_ethz_ana_const.nc')
Vmax = np.sqrt((cosmo_wind['wind_x'].flatten()**2 + cosmo_wind['wind_y'].flatten()**2 + cosmo_wind['wind_z'].flatten()**2).max())

# Plot the wind vector estimates
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot(x, y, alt)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_zlabel('Altitude (m)')

xx, yy, zz = x[::skip_amount]-x[0], y[::skip_amount]-y[0], alt[::skip_amount]
# h_w_est = ax.quiver(xx, yy, zz, we_east, we_north, we_up, colors=get_colors(we_east, we_north, we_up, Vmax=Vmax))
h_w_vanes = ax.quiver(xx, yy, zz, we, wn, -wd, colors=get_colors(we, wn, wd, Vmax=Vmax))

f2, a2 = plt.subplots(3, 1)
h_est = a2[0].plot(skip_time, we_east)
h_vanes = a2[0].plot(skip_time, we)
a2[0].legend(['Pixhawk estimate', 'Vane estimate (raw)'])
a2[0].set_ylabel('$V_E$')
a2[1].plot(skip_time, we_north)
a2[1].plot(skip_time, wn)
a2[1].set_ylabel('$V_N$')
a2[2].plot(skip_time, we_up)
a2[2].plot(skip_time, -wd)
a2[2].set_ylabel('$V_D$')

# Plot cosmo wind

ax.plot(cosmo_wind['x'].flat-x[0], cosmo_wind['y'].flat-y[0], cosmo_wind['hsurf'].flat, 'k.')
ones_vec = np.ones(cosmo_wind['wind_x'].shape[0])
for ix in range(2):
    for iy in range(2):
        cwe, cwn, cwu = cosmo_wind['wind_x'][:, ix, iy], cosmo_wind['wind_y'][:, ix, iy], cosmo_wind['wind_z'][:, ix, iy]
        ax.quiver(cosmo_wind['x'][ix, iy]*ones_vec-x[0], cosmo_wind['y'][ix, iy]*ones_vec-y[0], cosmo_wind['z'][:, ix, iy],
                  cwe, cwn, cwu, colors=get_colors(cwe, cwn, cwu, Vmax=Vmax))
norm = matplotlib.colors.Normalize()
norm.autoscale([0, Vmax])

sm = matplotlib.cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
sm.set_array([])
h_cb = fig.colorbar(sm, ax=ax)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_zlim(zz.min(), zz.max())

if savefig:
    bn = os.path.basename(logfile)
    fig.savefig('fig/{0}_wind3d.png'.format(bn), bbox_inches='tight')
    f2.savefig('fig/{0}_wind.png'.format(bn), bbox_inches='tight')
plt.show()
