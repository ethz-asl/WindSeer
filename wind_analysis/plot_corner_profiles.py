import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator

from analysis_utils.extract_cosmo_data import extract_cosmo_data, cosmo_corner_wind
from analysis_utils.get_mapgeo_terrain import get_terrain
from analysis_utils.plotting_analysis import vector_lims, plot_wind_3d, plot_cosmo_corners, plot_wind_estimates, rec2polar, plot_vertical_profile, plot_lateral_variation
from analysis_utils.ulog_utils import get_log_data, read_filtered_hdf5
import nn_wind_prediction.utils.yaml_tools as yaml_tools


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the wind from a log file and corresponding surrounding COSMO estimates')
    parser.add_argument('yaml_file', help='YAML config file (must contain "cosmo" and "ulog" dictionaries)')
    parser.add_argument('-s', '--save-figs', action='store_true', help='Save output figures')
    parser.add_argument('-r', '--resolution', type=int, default=64, help='Extracted block resolution')
    parser.add_argument('-f', '--filter', type=int, default=0, help='Filter size')
    
    args = parser.parse_args()

    cosmo_args = yaml_tools.COSMOParameters(args.yaml_file)
    cosmo_args.print()

    ulog_args = yaml_tools.UlogParameters(args.yaml_file)
    ulog_args.print()

    ulog_data = get_log_data(ulog_args.params['file'])

    lat0, lon0 = ulog_data['lat'][0], ulog_data['lon'][0]

    # Get cosmo wind
    t0 = datetime.utcfromtimestamp(ulog_data['utc_microsec'][0]/1e6)
    offset_cosmo_time = cosmo_args.get_cosmo_time(t0.hour)
    cosmo_wind = extract_cosmo_data(cosmo_args.params['file'], lat0, lon0, offset_cosmo_time,
                                    terrain_file=cosmo_args.params['terrain_file'])

    # Get corresponding terrain
    # min_height = min(ulog_data['alt'].min(), h_terr.min())
    block_height = [1100.0/95*63]
    # x_terr, y_terr and z_terr are the (regular, monotonic) index arrays for the h_terr and full_block arrays
    # h_terr is the terrain height
    x_terr, y_terr, z_terr, h_terr, _ = \
        get_terrain(cosmo_args.params['terrain_tiff'], cosmo_wind['x'][[0, 1], [0, 1]], cosmo_wind['y'][[0, 1], [0, 1]],
                    block_height, (args.resolution, args.resolution, args.resolution), plot=False)

    # Get corner winds for model inference, offset to actual terrain heights
    terrain_corners = h_terr[::h_terr.shape[0]-1, ::h_terr.shape[1]-1]
    cosmo_corners = cosmo_corner_wind(cosmo_wind, z_terr, terrain_height=terrain_corners,
                                      rotate=cosmo_args.params['rotate']*np.pi/180.0, scale=cosmo_args.params['scale'])

    bn = os.path.splitext(os.path.basename(ulog_args.params['file']))[0]

    plane_pos = np.array([ulog_data['x'], ulog_data['y'], ulog_data['alt']])
    w_vanes = np.array([ulog_data['we'], ulog_data['wn'], ulog_data['wd']])
    w_ekfest = np.array([ulog_data['we_east'], ulog_data['we_north'], ulog_data['we_down']])
    all_winds = [w_vanes, w_ekfest]
    wind_names = ['Raw vane estimates', 'On-board EKF estimate']
    try:
        filtered_wind = read_filtered_hdf5(ulog_args.params['filtered_file'])
        w_filtered = []
        for wind_key in ['wind_e', 'wind_n', 'wind_d']:
            w_filtered.append(np.interp(ulog_data['gp_time'], filtered_wind['time'], filtered_wind[wind_key]))
        w_filtered = np.array(w_filtered)
        all_winds.append(w_filtered)
        wind_names.append('Post-process filtered (HDF5 file)')
    except:
        print('Filtered wind hdf5 not found for ulog {0}'.format(bn))
        w_filtered = None

    plot_time = (ulog_data['gp_time']-ulog_data['gp_time'][0])*1e-6

    # Nearest COSMO corner for vertical profile
    cx = int((plane_pos[0, 0] - x_terr[0])/(x_terr[-1] - x_terr[0]) > 0.5)
    cy = int((plane_pos[1, 0] - y_terr[0])/(y_terr[-1] - y_terr[0]) > 0.5)
    if w_filtered is not None:
        w_plot = w_filtered
    else:
        w_plot = w_vanes

    altitude_data = ulog_data['alt']
    if args.filter > 0:
        if not args.filter % 2 == 0:
            skip = int((args.filter - 1) * 0.5)
            altitude_data = altitude_data[skip:-skip]
            plot_time = plot_time[skip:-skip]

            w_tmp = np.ones((w_plot.shape[0], w_plot.shape[1]-args.filter+1))
            for i in range(w_plot.shape[0]):
                w_tmp[i] = np.convolve(w_plot[i], np.ones(args.filter)/args.filter, mode='valid')

            w_plot = w_tmp

    fp, ap = plot_vertical_profile(z_terr, cosmo_corners[:,:,cy,cx], w_plot, altitude_data, plot_time)
    fp.set_size_inches((4.5, 3))

    cosmo_lims = vector_lims(np.array([cosmo_wind['wind_x'], cosmo_wind['wind_y'], cosmo_wind['wind_z']]), axis=0)
    Vlims = (0.0, cosmo_lims[1])
    fh, ah = plot_wind_3d(plane_pos, w_vanes, x_terr, y_terr, h_terr, cosmo_wind, origin=plane_pos[:,0].flat, Vlims=Vlims, plot_cosmo=True)
    fh.set_size_inches((4, 4))
    ah.set_zlim([-100, 1000])

    if args.save_figs:
        print("Saving figures.")
        fp.savefig('fig/{0}{1}_windProfile.png'.format(cosmo_args.params['prediction_prefix'], bn), bbox_inches='tight')
    plt.show()
