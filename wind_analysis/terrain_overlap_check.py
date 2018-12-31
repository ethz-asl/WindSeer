import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pyproj
import nn_wind_prediction.cosmo as cosmo
from osgeo import gdal
from get_mapgeo_terrain import get_terrain


cosmo_terrain='data/riemenstalden/cosmo-1_ethz_ana/cosmo-1_ethz_ana_const.nc'
cosmo_time = 1
terrain_tiff = 'terrain/lake_geneva_l.tif'         # riemenstalden_full

# Get terrain tif limits
img = gdal.Open(terrain_tiff)

# t is (x0, delta_x, delta_y, y0, delta_x, delta_y)
w = img.RasterXSize
h = img.RasterYSize
t = img.GetGeoTransform()
xlims, ylims = [t[0]+10, t[0] + t[1]*w-10], [t[3] + t[5]*h+10, t[3]-10]

x_terr, y_terr, h_terr = get_terrain(terrain_tiff, xlims, ylims, [0.0, 1.0], (1024, 1024, 512), build_binary_block=False)

# Get COSMO terrain:
lon, lat, hsurf, hfl = cosmo.read_cosmo_nc_file(cosmo_terrain, ['lon_1', 'lat_1', 'HSURF', 'HFL'])

# convert to output coordinate projection
proj_WGS84 = pyproj.Proj(proj='latlong', datum='WGS84')
proj_EGM96 = pyproj.Proj(init="EPSG:4326", geoidgrids="egm96_15.gtx") # init="EPSG:5773",
proj_CH_1903_LV03 = pyproj.Proj(init="EPSG:21781")  # https://epsg.io/21781
proj_CH_1903_LV03_SWISSTOPO = pyproj.Proj(init="CH:1903_LV03")

x_grid, y_grid, h_grid = pyproj.transform(proj_WGS84, proj_CH_1903_LV03_SWISSTOPO, lon, lat, hsurf)

# Plot the wind vector estimates
fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot(x, y, alt)
ax.set_xlabel('Easting (m)')
ax.set_ylabel('Northing (m)')
ax.set_zlabel('Altitude (m)')

origin = (0.0, 0.0, 0.0)
X, Y = np.meshgrid(x_terr-origin[0], y_terr-origin[1])
hs_terr = ax.plot_surface(X, Y, h_terr-origin[2], cmap=cm.terrain)

x_c, y_c, z_c = [], [], []
for x, y, h in zip(x_grid.flat, y_grid.flat, h_grid.flat):
    if x > xlims[0] and x < xlims[1] and y > ylims[0] and y < ylims[1]:
        x_c.append(x)
        y_c.append(y)
        z_c.append(h)
ax.plot(x_c, y_c, z_c, 'k.')
plt.show(block=False)