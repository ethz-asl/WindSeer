#!/usr/bin/env python
from __future__ import print_function

try:
    import numpy as np
except ImportError:
    print("'numpy' is not installed. Use the command below to install it:")
    print("     sudo apt-get install python-numpy")

try:
    from osgeo import gdal
except ImportError:
    print("'pygdal' is not installed. Use the command below to install it:")
    print("     pip3 install pygdal")

try:
    from scipy.interpolate import RectBivariateSpline
    from scipy import ndimage
except ImportError:
    print("'scipy' is not installed. Use the command below to install it:")
    print("     pip3 install scipy")


def plot_terrain_patch(x_img, y_img, z_img, x_p, y_p, z_p):
    '''
    Plot the terrain and the extracted patch extent.

    Parameters
    ----------
    x_img : np.array
        X-coordinates of the terrain
    y_img : np.array
        Y-coordinates of the terrain
    z_img : np.array
        Z-coordinates of the terrain
    x_p : np.array
        Boundaries of the extent along the x-axis
    y_p : np.array
        Boundaries of the extent along the y-axis
    z_p : np.array
        Boundaries of the extent along the z-axis
    '''
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    f, a = plt.subplots(1, 2, sharex=True, sharey=True)
    a[1].imshow(z_p, origin='lower', extent=[x_p[0], x_p[-1], y_p[0], y_p[-1]])

    a[0].imshow(
        z_img, origin='lower', extent=[x_img[0], x_img[-1], y_img[0], y_img[-1]]
        )
    a[0].add_patch(
        Rectangle((x_p[0], y_p[0]), (x_p[-1] - x_p[0]), (y_p[-1] - y_p[0]),
                  edgecolor='r',
                  alpha=1,
                  facecolor='none')
        )


def get_terrain(
        terrain_geotiff,
        x,
        y,
        z=None,
        resolution=(64, 64, 64),
        build_block=True,
        plot=False,
        distance_field=False,
        horizontal_overflow=0,
        return_is_wind=False
    ):
    """
    Extract a terrain patch from a geotiff, and sample to specified resolution

    Parameters
    ----------
    terrain_geotiff : str
        Geotiff file location
    x : np.array or list
        X extent of the target patch (x_lo, x_hi)
    y : np.array or list
        Y extent of the target patch (y_lo, y_hi)
    z : np.array or list or None, default : None
        None: use the complete terrain height range
        Single value: use the minimum terrain height to min+ the value (i.e. specify block height)
        Two values: use the specified height range
    resolution : tuple of int, default : (64,64,64)
        Output target resolution
    build_block : bool, default : True
        Flag to output the volumetric full_block
    plot : bool, default : False
        Plot the extracted terrain
    distance_field : bool, default : False
        Return the full_block as a distance field instead of binary map
    horizontal_overflow : int, default : 0
        Extract a larger portion of the map, build the distance field and then extract the actual extent.
        It is forced to be an integer and the unit is in number of cells.
    return_is_wind : bool, default : False
        In case of a binary full_block the cells will be true where there is wind, else terrain cells are true


    Returns
    -------
    x_terr : np.array
        Regular, monotonic index arrays of the x-axis for the h_terr and full_block arrays
    y_terr : np.array
        Regular, monotonic index arrays of the y-axis for the h_terr and full_block arrays
    z_terr : np.array (optional)
        Regular, monotonic index arrays of the z-axis for the h_terr and full_block arrays
        Only returned if build_block is True.
    h_terr : np.array
        Terrain height array (actual height values in m)
    full_block : np.array (optional)
        Terrain array (size is resolution input argument) dimension order is [z, y, x].
        Depending on the distance_field argument a binary array or a distance field array.
    """
    img = gdal.Open(terrain_geotiff)

    # image dimensions
    w = img.RasterXSize
    h = img.RasterYSize

    band = img.GetRasterBand(1)

    # t is (x0, delta_x, delta_y, y0, delta_x, delta_y)
    t = img.GetGeoTransform()
    assert t[2] == t[4] == 0.0, "Input image is not aligned, only aligned images allowed"

    x_img = t[0] + t[1] / 2.0 + t[1] * np.arange(w)
    y_img = t[3] + t[5] / 2.0 + t[5] * np.arange(h)
    z_img = band.ReadAsArray()

    # We might need to flip (if the steps are negative, usually y is because it's like image coords)
    if t[1] < 0:
        z_img = np.fliplr(z_img)
        x_img = x_img[::-1]
    if t[5] < 0:
        z_img = np.flipud(z_img)
        y_img = y_img[::-1]

    dx = (x[1] - x[0]) / (resolution[0] - 1)
    dy = (y[1] - y[0]) / (resolution[1] - 1)

    horizontal_overflow = int(horizontal_overflow)
    bounds_x = x.copy()
    bounds_y = y.copy()
    if distance_field and horizontal_overflow > 0:
        bounds_x[0] -= horizontal_overflow * dx
        bounds_x[1] += horizontal_overflow * dx
        bounds_y[0] -= horizontal_overflow * dy
        bounds_y[1] += horizontal_overflow * dy
    else:
        horizontal_overflow = 0

    # extract the relevant image data to speed up the spline building
    # especially important if we have a high quality terrain
    extended_bounds_x = [bounds_x[0] - 2 * np.abs(t[1]), bounds_x[1] + 2 * np.abs(t[1])]
    extended_bounds_y = [bounds_y[0] - 2 * np.abs(t[5]), bounds_y[1] + 2 * np.abs(t[5])]
    idx_to_keep_x = np.logical_and(
        x_img < extended_bounds_x[1], x_img > extended_bounds_x[0]
        )
    idx_to_keep_y = np.logical_and(
        y_img < extended_bounds_y[1], y_img > extended_bounds_y[0]
        )

    interp_spline = RectBivariateSpline(
        y_img[idx_to_keep_y], x_img[idx_to_keep_x], z_img[idx_to_keep_y][:,
                                                                         idx_to_keep_x]
        )

    x_target = bounds_x[0] + dx * np.arange(resolution[0] + 2 * horizontal_overflow)
    y_target = bounds_y[0] + dy * np.arange(resolution[1] + 2 * horizontal_overflow)
    z_out = interp_spline(y_target, x_target)

    if plot:
        plot_terrain_patch(x_img, y_img, z_img, x_target, y_target, z_out)

    if build_block:
        # Build binary indicator array
        if z is None:
            z = [z_out.min(), z_out.max()]
        elif len(z) == 1:
            z = [z_out.min(), z_out.min() + z[0]]

        dz = (z[1] - z[0]) / (resolution[2] - 1)
        bounds_z = z.copy()

        # full_block is dimension order [z, y, x], z_out is [y, x]
        if distance_field:
            offset_cells_z = int(np.max([np.ceil((bounds_z[0] - z_out.min()) / dz), 0]))
            bounds_z[0] -= offset_cells_z * dz
            full_block = np.zeros(
                (resolution[0] + offset_cells_z, z_out.shape[0], z_out.shape[1]),
                dtype='bool'
                )
        else:
            offset_cells_z = 0
            full_block = np.zeros(resolution, dtype='bool')

        z_target = bounds_z[0] + dz * np.arange(resolution[2] + offset_cells_z)

        for yi in range(z_out.shape[0]):
            for xi in range(z_out.shape[1]):
                height_dex = np.sum(z_target < z_out[yi, xi])
                full_block[:height_dex, yi, xi] = True

        if distance_field:
            is_wind = np.logical_not(full_block).astype('float')
            full_block = ndimage.distance_transform_edt(is_wind)
            if horizontal_overflow > 0:
                full_block = full_block[offset_cells_z:,
                                        horizontal_overflow:-horizontal_overflow,
                                        horizontal_overflow:-horizontal_overflow]
            else:
                full_block = full_block[offset_cells_z:]

        elif return_is_wind:
            full_block = np.logical_not(full_block).astype('float')

        return x_target, y_target, z_target, z_out, full_block

    else:
        return x_target, y_target, z_out
