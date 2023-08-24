import argparse
from datetime import datetime
import numpy as np
import os

import windseer.utils as utils


def make_openfoam_csv(
        cosmo_data_file,
        cosmo_terrain_file,
        geotiff_file,
        logfile,
        csv_file,
        resolution=64,
        plot_figs=False,
        rotate=None,
        scale=None
    ):
    log_data = utils.extract_wind_data(logfile, False)
    lat0, lon0 = log_data['lat'][0], log_data['lon'][0]

    # Get cosmo wind
    t0 = datetime.utcfromtimestamp(log_data['time_gps'][0])
    cosmo_wind = utils.extract_cosmo_data(
        cosmo_data_file, lat0, lon0, t0, terrain_file=cosmo_terrain_file
        )

    # Get corresponding terrain
    # min_height = min(log_data['alt'].min(), h_terr.min())
    block_height = [1100.0 / 95 * 63]
    # x_terr, y_terr and z_terr are the (regular, monotonic) index arrays for the h_terr and full_block arrays
    # h_terr is the terrain height
    x_terr, y_terr, z_terr, h_terr, full_block = \
        utils.get_terrain(geotiff_file, cosmo_wind['x'][[0, 1], [0, 1]], cosmo_wind['y'][[0, 1], [0, 1]],
                    block_height, (resolution, resolution, resolution), plot=plot_figs)

    if rotate is None:
        rotate = 0.0
    if scale is None:
        scale = 1.0

    # Get corner winds for model inference, offset to actual terrain heights
    print('Rotation: {0:0.2f} deg, scale: {1:0.2f}'.format(rotate, scale))
    terrain_corners = h_terr[::h_terr.shape[0] - 1, ::h_terr.shape[1] - 1]
    cosmo_corners = utils.cosmo_corner_wind(
        cosmo_wind,
        z_terr,
        terrain_height=terrain_corners,
        rotate=rotate * np.pi / 180.0,
        scale=scale
        )

    print('Saving csv to {0}'.format(csv_file))
    utils.build_csv(x_terr, y_terr, z_terr, full_block, cosmo_corners, csv_file)


if __name__ == "__main__":
    description = 'Create a csv file (equivalent to openfoam output file) for network input by\n' + \
                  'reading location data from a flight log (ulog) and recovering wind from\n' + \
                  'corresponding surrounding COSMO estimates.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '-c', '--cosmo_data', required=True, help='Cosmo file containing the wind data'
        )
    parser.add_argument(
        '-t',
        '--cosmo_terrain',
        required=True,
        help='Cosmo file containing the terrain data'
        )
    parser.add_argument(
        '-g', '--geotiff', required=True, help='High resolution terrain file'
        )
    parser.add_argument(
        '-l',
        '--logfile',
        required=True,
        help='Log file containing the wind measurements, either ulog or hdf5'
        )
    parser.add_argument(
        '-csv',
        '--csv_filename',
        required=True,
        help='Filename of the csv that is generated'
        )
    parser.add_argument(
        '-r', '--resolution', type=int, default=64, help='Extracted block resolution'
        )
    parser.add_argument(
        '-p', '--plot', action='store_true', help='Plot network prediction'
        )
    args = parser.parse_args()
    make_openfoam_csv(
        args.cosmo_data, args.cosmo_terrain, args.geotiff, args.logfile,
        args.csv_filename, args.resolution, args.plot
        )
