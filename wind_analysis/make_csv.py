import numpy as np
import ulog_utils
from get_mapgeo_terrain import get_terrain
import os
import nn_wind_prediction.cosmo as cosmo
import nn_wind_prediction.utils.yaml_tools as yaml_tools
import argparse
from datetime import datetime


if __name__ == "__main__":
    description = 'Create a csv file (equivalent to openfoam output file) for network input by\n' + \
                  'reading location data from a flight log (ulog) and recovering wind from\n' + \
                  'corresponding surrounding COSMO estimates.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('yaml_file', help='YAML config file (must contain "cosmo" and "ulog" dictionaries)')
    parser.add_argument('-r', '--resolution', type=int, default=64, help='Extracted block resolution')
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
                    block_height, (args.resolution, args.resolution, args.resolution), plot=args.plot_figs)

    # Get corner winds for model inference, offset to actual terrain heights
    terrain_corners = h_terr[::h_terr.shape[0]-1, ::h_terr.shape[1]-1]
    cosmo_corners = cosmo.cosmo_corner_wind(cosmo_wind, z_terr, terrain_height=terrain_corners,
                                            rotate=cosmo_args.params['rotate']*np.pi/180.0, scale=cosmo_args.params['scale'])
    bn = os.path.splitext(os.path.basename(ulog_args.params['file']))[0]

    try:
        csv_args = yaml_tools.BasicParameters(args.yaml_file, 'csv')
        print('Saving csv to {0}'.format(csv_args.params['file']))
        ulog_utils.build_csv(x_terr, y_terr, z_terr, full_block, cosmo_corners, csv_args.params['file'])
    except:
        print('CSV filename parameter (csv:file) not found in {0}, csv not saved'.format(args.yaml_file))