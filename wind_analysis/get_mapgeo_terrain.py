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


def get_terrain(terrain_geotiff, x, y, z, resolution=(64,64,64)):

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
    x_target = x[0] + (x[1] - x[0])/(resolution[0]-1)*np.arange(resolution[0])
    y_target = y[0] + (y[1] - y[0])/(resolution[1]-1) * np.arange(resolution[1])
    z_out = interp_spline(y_target, x_target)

    # Build binary indicator array
    z_target = z[0] + (z[1] - z[0])/(resolution[2]-1) * np.arange(resolution[2])
    full_block = np.zeros(resolution, dtype='bool')
    for i in range(z_out.shape[0]):
        for j in range(z_out.shape[1]):
            height_dex = np.sum(z_target < z_out[i,j])
            full_block[i,j,:height_dex] = True

    return x_target, y_target, z_target, z_out, full_block
