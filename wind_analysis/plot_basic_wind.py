import numpy as np
import ulog_utils
from get_mapgeo_terrain import get_terrain
import plot_utils
import matplotlib.pyplot as plt
import os
import nn_wind_prediction.cosmo as cosmo
import nn_wind_prediction.utils.yaml_tools as yaml_tools
import argparse
from datetime import datetime
from scipy.interpolate import RegularGridInterpolator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the wind from a log file and corresponding surrounding COSMO estimates')
    parser.add_argument('yaml_file', help='YAML config file (must contain "cosmo" and "ulog" dictionaries)')
    parser.add_argument('-s', '--save-figs', action='store_true', help='Save output figures')
    parser.add_argument('-r', '--resolution', type=int, default=64, help='Extracted block resolution')
    parser.add_argument('-p', '--prediction', action='store_true', help='Plot network prediction')
    parser.add_argument('--polar', action='store_true', help='Plot polar coordinates (instead of N,E)')
    args = parser.parse_args()

    cosmo_args = yaml_tools.COSMOParameters(args.yaml_file)
    cosmo_args.print()

    ulog_args = yaml_tools.UlogParameters(args.yaml_file)
    ulog_args.print()

    ulog_data = ulog_utils.get_log_data(ulog_args.params['file'])

    lat0, lon0 = ulog_data['lat'][0], ulog_data['lon'][0]

    # Get cosmo wind
    t0 = datetime.utcfromtimestamp(ulog_data['utc_microsec'][0]/1e6)
    offset_cosmo_time = cosmo_args.get_cosmo_time(t0.hour)
    cosmo_wind = cosmo.extract_cosmo_data(cosmo_args.params['file'], lat0, lon0, offset_cosmo_time,
                                   terrain_file=cosmo_args.params['terrain_file'])

    # Get corresponding terrain
    # min_height = min(ulog_data['alt'].min(), h_terr.min())
    block_height = [1100.0/95*63]
    # x_terr, y_terr and z_terr are the (regular, monotonic) index arrays for the h_terr and full_block arrays
    # h_terr is the terrain height
    x_terr, y_terr, z_terr, h_terr, full_block = \
        get_terrain(cosmo_args.params['terrain_tiff'], cosmo_wind['x'][[0, 1], [0, 1]], cosmo_wind['y'][[0, 1], [0, 1]],
                    block_height, (args.resolution, args.resolution, args.resolution), plot=True)

    # Get corner winds for model inference, offset to actual terrain heights
    terrain_corners = h_terr[::h_terr.shape[0]-1, ::h_terr.shape[1]-1]
    cosmo_corners = cosmo.cosmo_corner_wind(cosmo_wind, z_terr, terrain_height=terrain_corners,
                                            rotate=cosmo_args.params['rotate']*np.pi/180.0, scale=cosmo_args.params['scale'])

    bn = os.path.splitext(os.path.basename(ulog_args.params['file']))[0]

    plane_pos = np.array([ulog_data['x'], ulog_data['y'], ulog_data['alt']])
    w_vanes = np.array([ulog_data['we'], ulog_data['wn'], ulog_data['wd']])
    w_ekfest = np.array([ulog_data['we_east'], ulog_data['we_north'], ulog_data['we_down']])
    all_winds = [w_vanes, w_ekfest]
    wind_names = ['Raw vane estimates', 'On-board EKF estimate']
    try:
        filt_file = os.path.join(ulog_args.params['filtered_dir'], bn+'_filtered.hdf5')
        filtered_wind = ulog_utils.read_filtered_hdf5(filt_file)
        w_filtered = []
        for wind_key in ['wind_e', 'wind_n', 'wind_d']:
            w_filtered.append(np.interp(ulog_data['gp_time'], filtered_wind['time'], filtered_wind[wind_key]))
        w_filtered = np.array(w_filtered)
        all_winds.append(w_filtered)
        wind_names.append('Post-process filtered (HDF5 file)')
    except:
        print('Filtered wind hdf5 not found for ulog {0}'.format(bn))
        w_filtered = None

    vane_lims = plot_utils.vector_lims(w_vanes, axis=0)
    cosmo_lims = plot_utils.vector_lims(np.array([cosmo_wind['wind_x'], cosmo_wind['wind_y'], cosmo_wind['wind_z']]), axis=0)
    Vlims = (0.0, max(vane_lims[1], cosmo_lims[1]))  # min(vane_lims[0], cosmo_lims[0])

    fh, ah = plot_utils.plot_wind_3d(plane_pos, w_vanes, x_terr, y_terr, h_terr, cosmo_wind, origin=plane_pos[:,0].flat, Vlims=Vlims, plot_cosmo=True)
    plot_utils.plot_cosmo_corners(ah, cosmo_corners, x_terr, y_terr, z_terr, origin=plane_pos[:,0].flat, Vlims=Vlims)
    plot_time = (ulog_data['gp_time']-ulog_data['gp_time'][0])*1e-6
    f2, a2 = plot_utils.plot_wind_estimates(plot_time, all_winds, wind_names, polar=args.polar)
    if args.prediction:
        try:
            prediction_array = np.load('data/{0}{1}.npy'.format(cosmo_args.params['prediction_prefix'], bn))
        except:
            print('Prediction file {0} not found.')
        x_terr2 = np.linspace(x_terr[0], x_terr[-1], prediction_array.shape[-1])
        y_terr2 = np.linspace(y_terr[0], y_terr[-1], prediction_array.shape[-2])
        z_terr2 = np.linspace(z_terr[0], z_terr[-1], prediction_array.shape[-3])
        prediction_interp = []
        for pred_dim in prediction_array:
            prediction_interp.append(RegularGridInterpolator((z_terr2, y_terr2, x_terr2), pred_dim))

        # Get all the in bounds points
        inbounds = np.ones(ulog_data['x'].shape, dtype='bool')
        inbounds = np.logical_and.reduce([ulog_data['x'] > x_terr[0], ulog_data['x'] < x_terr[-1], inbounds])
        inbounds = np.logical_and.reduce([ulog_data['y'] > y_terr[0], ulog_data['y'] < y_terr[-1], inbounds])
        inbounds = np.logical_and.reduce([ulog_data['alt'] > z_terr[0], ulog_data['alt'] < z_terr[-1], inbounds])

        pred_t = (ulog_data['gp_time'][inbounds] - ulog_data['gp_time'][0])*1e-6
        points = np.array([ulog_data['alt'][inbounds], ulog_data['y'][inbounds], ulog_data['x'][inbounds]]).T
        pred_wind = [prediction_interp[0](points), prediction_interp[1](points), prediction_interp[2](points)]

        if args.polar:
            pred_V, pred_dir = plot_utils.rec2polar(pred_wind[0], pred_wind[1], wind_bearing=True, deg=True)
            a2[0].plot(pred_t, pred_V, 'r.')
            a2[1].plot(pred_t, pred_dir, 'r.')
        else:
            a2[0].plot(pred_t, pred_wind[0], 'r.')
            a2[1].plot(pred_t, pred_wind[1], 'r.')
        a2[2].plot(pred_t, pred_wind[2], 'r.')

        # Calculate RMSE - resample measured wind onto valid time stamps from

    # Nearest COSMO corner for vertical profile
    cx = int((plane_pos[0, 0] - x_terr[0])/(x_terr[-1] - x_terr[0]) > 0.5)
    cy = int((plane_pos[1, 0] - y_terr[0])/(y_terr[-1] - y_terr[0]) > 0.5)
    if w_filtered is not None:
        w_plot = w_filtered
    else:
        w_plot = w_vanes
    fp, ap = plot_utils.plot_vertical_profile(z_terr, cosmo_corners[:,:,cy,cx], w_plot, ulog_data['alt'], plot_time)
    fp.set_size_inches((5, 8))
    fv, av = plot_utils.plot_lateral_variation(w_plot, plane_pos, plot_time, min_alt=1700, max_alt=None)

    if args.save_figs:
        print("Saving figures.")
        fh.savefig('fig/{0}{1}_wind3d.png'.format(cosmo_args.params['prediction_prefix'], bn), bbox_inches='tight')
        f2.savefig('fig/{0}{1}_wind.png'.format(cosmo_args.params['prediction_prefix'], bn), bbox_inches='tight')
        fp.savefig('fig/{0}{1}_windProfile.png'.format(cosmo_args.params['prediction_prefix'], bn), bbox_inches='tight')
        fv.savefig('fig/{0}{1}_windLateral.png'.format(cosmo_args.params['prediction_prefix'], bn), bbox_inches='tight')
    plt.show(block=False)

## Alternative projections
# proj_EGM96 = pyproj.Proj(init="EPSG:4326", geoidgrids="egm96_15.gtx") # init="EPSG:5773",
# proj_CH_1903_LV03 = pyproj.Proj(init="EPSG:21781")  # https://epsg.io/21781
# proj_CH_1903_LV95 = pyproj.Proj(init="EPSG:2056")
# proj_CH_1903 = pyproj.Proj(init="CH:1903")
# proj_SPHERE = pyproj.Proj(proj='latlong', ellps='sphere', a='6371000')