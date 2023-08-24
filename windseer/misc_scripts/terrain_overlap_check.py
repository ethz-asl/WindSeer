import argparse
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pyproj
from osgeo import gdal

import windseer.utils as utils

parser = argparse.ArgumentParser(
    description='Plot the cosmo grid in comparison to the height map'
    )
parser.add_argument('-c', '--cosmo_file', help='Filename of the cosmo file')
parser.add_argument('-t', '--geotiff_file', help='Filename of the geotiff')
parser.add_argument(
    '-b',
    '--plot_blocking',
    action='store_true',
    help='Indicates if plt.show should be blocking'
    )
parser.add_argument(
    '-n_hor', '--n_horizontal', type=int, default=1024, help='Number of vertical cells'
    )
parser.add_argument(
    '-n_ver', '--n_vertical', type=int, default=512, help='Number of vertical cells'
    )

args = parser.parse_args()

# does not really matter
cosmo_time = 0

# Get terrain tif limits
img = gdal.Open(args.geotiff_file)

# t is (x0, delta_x, delta_y, y0, delta_x, delta_y)
w = img.RasterXSize
h = img.RasterYSize
t = img.GetGeoTransform()
xlims, ylims = [t[0] + 10, t[0] + t[1] * w - 10], [t[3] + t[5] * h + 10, t[3] - 10]

x_terr, y_terr, h_terr = utils.get_terrain(
    args.geotiff_file,
    xlims,
    ylims, [0.0, 1.0], (args.n_horizontal, args.n_horizontal, args.n_vertical),
    build_block=False
    )

# Get COSMO terrain:
lon, lat, hsurf, hfl = utils.read_cosmo_nc_file(
    args.cosmo_file, ['lon_1', 'lat_1', 'HSURF', 'HFL']
    )

# convert to output coordinate projection
proj_WGS84 = pyproj.Proj(proj='latlong', datum='WGS84')
proj_CH_1903_LV03_SWISSTOPO = pyproj.Proj(init="CH:1903_LV03")

x_grid, y_grid, h_grid = pyproj.transform(
    proj_WGS84, proj_CH_1903_LV03_SWISSTOPO, lon, lat, hsurf
    )

# Plot the wind vector estimates
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_zlabel('Altitude (m)')

origin = (0.0, 0.0, 0.0)
X, Y = np.meshgrid(x_terr - origin[0], y_terr - origin[1])
hs_terr = ax.plot_surface(X, Y, h_terr - origin[2], cmap=cm.terrain)

x_c, y_c, z_c = [], [], []
for x, y, h in zip(x_grid.flat, y_grid.flat, h_grid.flat):
    if x > xlims[0] and x < xlims[1] and y > ylims[0] and y < ylims[1]:
        x_c.append(x)
        y_c.append(y)
        z_c.append(h)
ax.plot(x_c, y_c, z_c, 'k.', markersize=20)

plt.show(block=args.plot_blocking)
