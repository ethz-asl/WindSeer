#!/usr/bin/env python
from __future__ import print_function

try:
    import numpy as np
except ImportError:
    print("'numpy' is not installed. Use the command below to install it:")
    print("     sudo apt-get install python-numpy")
    exit()

try:
    from osgeo import gdal
except ImportError:
    print("'pygdal' is not installed. Use the command below to install it:")
    print("     pip3 install pygdal==2.3.2.4")
    exit()

try:
    from scipy.interpolate import RectBivariateSpline
except ImportError:
    print("'scipy' is not installed. Use the command below to install it:")
    print("     pip3 install scipy")
    exit()


def plot_terrain_patch(x_img, y_img, z_img, x_p, y_p, z_p):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    f, a = plt.subplots(1, 2, sharex=True, sharey=True)
    a[1].imshow(z_p, origin='lower', extent=[x_p[0], x_p[-1], y_p[0], y_p[-1]])

    a[0].imshow(z_img, origin='lower', extent=[x_img[0], x_img[-1], y_img[0], y_img[-1]])
    a[0].add_patch(Rectangle((x_p[0], y_p[0]), (x_p[-1]-x_p[0]), (y_p[-1]-y_p[0]), edgecolor='r', alpha=1, facecolor='none'))


def get_terrain(terrain_geotiff, x, y, z=None, resolution=(64,64,64), build_binary_block=True, plot=False):
    """
    Extract a terrain patch from a geotiff, and sample to specified resolution
    Inputs:
        terrain_geotiff:    geotiff file location (string)
        x:                  x extent of target patch (x_lo, x_hi)
        y:                  y extent of target patch (y_lo, y_hi)
        z:    If z is:
                  None: use the complete terrain height range
                  Single value: use the minimum terrain height to min+ the value (i.e. specify block height)
                  Two values: use the specified height range
        resolution:         output target resolution (rx, ry, rz)
        build_binary_block: flag to output full_block

    Outputs:
          x_, y_, z_terr:    regular, monotonic index arrays for the h_terr and full_block arrays
          h_terr:            terrain height array (actual height values in m, resolution[0] x resolution[1]
          full_block:        binary terrain array (size is resolution input argument)
                            True is terrain, False is air, dimension order is [z, y, x]
    """
    img = gdal.Open(terrain_geotiff)

    # image dimensions
    w = img.RasterXSize
    h = img.RasterYSize

    band = img.GetRasterBand(1)

    # t is (x0, delta_x, delta_y, y0, delta_x, delta_y)
    t = img.GetGeoTransform()
    assert t[2] == t[4] == 0.0, "Input image is not aligned, only aligned images allowed"

    x_img = t[0] + t[1]/2.0 + t[1]*np.arange(w)
    y_img = t[3] + t[5]/2.0 + t[5]*np.arange(h)
    z_img = band.ReadAsArray()

    # We might need to flip (if the steps are negative, usually y is because it's like image coords)
    if t[1] < 0:
        z_img = np.fliplr(z_img)
        x_img = x_img[::-1]
    if t[5] < 0:
        z_img = np.flipud(z_img)
        y_img = y_img[::-1]

    interp_spline = RectBivariateSpline(y_img, x_img, z_img)
    x_target = x[0] + (x[1] - x[0])/(resolution[0]-1) * np.arange(resolution[0])
    y_target = y[0] + (y[1] - y[0])/(resolution[1]-1) * np.arange(resolution[1])
    z_out = interp_spline(y_target, x_target)

    if plot:
        plot_terrain_patch(x_img, y_img, z_img, x_target, y_target, z_out)

    if build_binary_block:
        # Build binary indicator array
        if z is None:
            z = [z_out.min(), z_out.ax()]
        elif len(z) == 1:
            z = [z_out.min(), z_out.min()+z[0]]
        z_target = z[0] + (z[1] - z[0])/(resolution[2]-1) * np.arange(resolution[2])
        full_block = np.zeros(resolution, dtype='bool')     # full_block is dimension order [z, y, x], z_out is [y, x]
        for yi in range(z_out.shape[0]):
            for xi in range(z_out.shape[1]):
                height_dex = np.sum(z_target < z_out[yi,xi])
                full_block[:height_dex, yi, xi] = True

        return x_target, y_target, z_target, z_out, full_block

    else:
        return x_target, y_target, z_out
