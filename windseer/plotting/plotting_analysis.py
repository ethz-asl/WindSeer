import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

plt.rc('font', **{'family': 'serif', 'sans-serif': ['Computer Modern Roman']})
plt.rc('text', usetex=True)


def vector_lims(uvw, axis=0):
    '''
    Get the limits of the values in a vector along a certain axis.

    Parameters
    ----------
    uvw : np.array
        Data vector
    axis : int, default : 0
        Data along which the limits are calculated

    Returns
    -------
    limits : np.array
        Array with the lower and upper bound
    '''
    vflat = (uvw * uvw).sum(axis=axis).flatten()
    return np.sqrt([vflat.min(), vflat.max()])


def rec2polar(vx, vy, wind_bearing=False, deg=False):
    '''
    Convert the wind into polar coordinates

    Parameters
    ----------
    vx : np.array
        Wind in x-direction
    vy : np.array
        Wind in y-direction
    wind_bearing : bool, default : False
        If True the heading is the direction where the wind is coming from, else the vector direction
    deg : bool, default : False
        If True the heading is returned in degrees, else in radians

    Returns
    -------
    limits : np.array
        Array with the lower and upper bound
    '''
    mag = np.sqrt(vx**2 + vy**2)
    if wind_bearing:
        dir = np.pi / 2 - np.arctan2(vy, vx)
    else:
        dir = np.arctan2(vy, vx)
    if deg:
        return mag, 360.0 / np.pi * dir
    else:
        return mag, dir


def get_colors(uvw, cmap=plt.cm.hsv, Vlims=None):
    '''
    Compute the colors of the entries in a vector according to a colormap

    Parameters
    ----------
    uvw : np.array
        Data vector
    cmap : colormap, default : plt.cm.hsv
        Colormap
    Vlims : None or np.array, default : None
        If not None sets the boundaries for the colormap

    Returns
    -------
    colors : np.array
        Array with colors for each entry in the input data vector
    '''
    # Color by magnitude
    c = np.sqrt((uvw * uvw).sum(axis=0))
    # Flatten and normalize
    if Vlims is None:
        Vlims = (c.min(), c.max())
    c = (c.ravel() - Vlims[0]) / Vlims[1]
    # There is a problem where zero-length vectors are removed, offsetting all the correspnding colours
    # Repeat for each body line and two head lines
    c = c[c != 0.0]
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    return cmap(c)


def plot_wind_3d(
        pos,
        wind,
        x_terr,
        y_terr,
        h_terr,
        cosmo_wind=None,
        origin=(0.0, 0.0, 0.0),
        wskip=5,
        Vlims=None,
        plot_cosmo=False
    ):
    '''
    Plot the logged wind, cosmo wind, and the terrain.

    Parameters
    ----------
    pos : np.array
        Position of wind measurements location
    wind : np.array
        Measured wind
    x_terr : np.array
        X-coordinates of the terrain
    y_terr : np.array
        Y-coordinates of the terrain
    h_terr : np.array
        Terrain height array
    cosmo_wind : dict or None, default : None
        Dictionary with the cosmo wind data
    origin : np.array or tuple, default : (0.0, 0.0, 0.0)
        Set the origin
    wskip : int, default : 5
        Only plot every nth wind measurement
    Vlims : None or np.array, default : None
        If not None sets the boundaries for the colormap, if None the boundaries are computed
    plot_cosmo : bool, default : False
        If true the cosmo data is plotted

    Returns
    -------
    fig: Figure
        Figure handle
    ax : Axes3DSubplot
        Axes handle
    '''
    # Plot the wind vector estimates
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Altitude (m)')

    X, Y = np.meshgrid(x_terr - origin[0], y_terr - origin[1])
    ax.plot_surface(X, Y, h_terr - origin[2], cmap=cm.terrain)

    if plot_cosmo:
        cwind = np.array([
            cosmo_wind['wind_x'], cosmo_wind['wind_y'], cosmo_wind['wind_z']
            ])
        if Vlims is None:
            wind_lims = vector_lims(wind)
            cw_lims = vector_lims(cwind)
            Vlims = (min(wind_lims[0], cw_lims[0]), max(wind_lims[1], cw_lims[1]))

    xx, yy, zz = pos[0] - origin[0], pos[1] - origin[1], pos[2] - origin[2]
    ax.plot(xx, yy, zz, 'k.', ms=1)
    wind_skip = wind[:, ::wskip]
    ax.quiver(
        xx[::wskip],
        yy[::wskip],
        zz[::wskip],
        wind_skip[0],
        wind_skip[1],
        wind_skip[2],
        colors=get_colors(wind_skip, Vlims=Vlims),
        length=5.0
        )

    # Plot cosmo wind
    if plot_cosmo:
        ax = plot_cosmo_corners(ax, cosmo_wind, origin, Vlims)

    norm = matplotlib.colors.Normalize()

    norm.autoscale(Vlims)

    sm = matplotlib.cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
    sm.set_array([])
    hc = fig.colorbar(sm, ax=ax)
    hc.set_label('Wind speed (m/s)')
    return fig, ax


def plot_cosmo_corners(ax, cosmo_wind, origin=(0.0, 0.0, 0.0), Vlims=None):
    '''
    Plot the wind of the cosmo corners

    Parameters
    ----------
    cosmo_wind : dict
        Dictionary with the cosmo wind data
    origin : np.array or tuple, default : (0.0, 0.0, 0.0)
        Set the origin
    Vlims : None or np.array, default : None
        If not None sets the boundaries for the colormap, if None the boundaries are computed

    Returns
    -------
    ax : Axes3DSubplot
        Axes handle
    '''
    cwind = np.array([cosmo_wind['wind_x'], cosmo_wind['wind_y'], cosmo_wind['wind_z']])
    if Vlims is None:
        Vlims = vector_lims(cwind)

    ones_vec = np.ones(cwind.shape[1])
    for ix in range(2):
        for iy in range(2):
            cw = cwind[:, :, ix, iy]
            ax.quiver(
                cosmo_wind['x'][ix, iy] * ones_vec - origin[0],
                cosmo_wind['y'][ix, iy] * ones_vec - origin[1],
                cosmo_wind['z'][:, ix, iy] - origin[2],
                cw[0],
                cw[1],
                cw[2],
                colors=get_colors(cw, Vlims=Vlims),
                length=5.0
                )

    return ax


def plot_vertical_profile(z_terr, cosmo_corner, wind_est, alt, t):
    '''
    Plot and compare the measured wind to the cosmo wind for a vertical profile

    Parameters
    ----------
    z_terr : np.array
        Terrain height
    cosmo_corner : np.array
        Cosmo corner wind prediction
    wind_est : np.array
        Estimated wind along the flight path
    alt : np.array
        Altitude of the wind estimates
    t : np.array
        Time of the wind estimates

    Returns
    -------
    fig: Figure
        Figure handle
    ax : Axes3DSubplot
        Axes handle
    '''
    fig, ax = plt.subplots(1, 2)
    # magnitude
    vv = np.sqrt(cosmo_corner[0, :]**2 + cosmo_corner[1, :]**2)
    ht, = ax[0].plot(vv, z_terr)
    v3 = np.sqrt(wind_est[0, :]**2 + wind_est[1, :]**2)
    h2 = ax[0].scatter(v3, alt, c=t, s=15)
    try:
        ax[0].grid(visible=True, which='both')
    except:
        ax[0].grid(b=True, which='both')
    ax[0].set_xlabel('Wind speed [m/s]')
    ax[0].set_ylabel('Alt, m')
    ax[0].set_ylim(np.floor(alt.min() / 100) * 100, np.ceil(alt.max() / 100) * 100)

    # direction
    dir_cosmo = np.degrees(np.arctan2(-cosmo_corner[0, :], -cosmo_corner[1, :])) % 360
    dir_wind = np.degrees(np.arctan2(-wind_est[0, :], -wind_est[1, :])) % 360
    h_dc, = ax[1].plot(dir_cosmo, z_terr)
    h_dm = ax[1].scatter(dir_wind, alt, c=t, s=15)
    try:
        ax[1].grid(visible=True, which='both')
    except:
        ax[1].grid(b=True, which='both')
    ax[1].set_xlabel('Wind Direction [deg]')
    ax[1].set_ylim(np.floor(alt.min() / 100) * 100, np.ceil(alt.max() / 100) * 100)
    hl_2 = ax[1].legend([h_dc, h_dm], ['COSMO profile', 'UAV data'])

    hc = fig.colorbar(h_dm, ax=ax[1])
    hc.set_label('Mission time (s)')
    return fig, ax


def plot_lateral_variation(
        wind_est, pos, t, min_alt=None, max_alt=None, fig=None, ax=None, scale=100.0
    ):
    '''
    Plot a top down view of the wind estimates and potentially filter for a certain altitude band.

    Parameters
    ----------
    wind_est : np.array
        Estimated wind along the flight path
    pos : np.array
        Position of the wind estimates
    t : np.array
        Time of the wind estimates
    min_alt  : None or float, default : None
        Filter for a minimum altitude
    max_alt  : None or float, default : None
        Filter for a maximum altitude
    fig  : Figure or float, default : None
        If none a new figure is created, else the provided figure handle is used
    ax  : Axes3DSubplot or float, default : None
        If none a new axes handle is created, else the provided handle is used

    Returns
    -------
    fig: Figure
        Figure handle
    ax : Axes3DSubplot
        Axes handle
    '''
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    dex = np.ones(pos[0].shape, dtype='bool')
    if min_alt is not None:
        dex = np.logical_and(dex, pos[2] > min_alt)
    if max_alt is not None:
        dex = np.logical_and(dex, pos[2] < max_alt)
    h2 = ax.quiver(
        pos[0][dex],
        pos[1][dex],
        wind_est[0][dex],
        wind_est[1][dex],
        t[dex],
        scale=scale
        )
    try:
        ax.grid(visible=True, which='both')
    except:
        ax.grid(b=True, which='both')
    ax.plot(pos[0][0], pos[1][0], 'g^')
    ax.plot(pos[0][-1], pos[1][-1], 'ro')
    ax.set_xlabel('$x$ (East), m')
    ax.set_ylabel('$y$ (North), m')
    hc = fig.colorbar(h2, ax=ax)
    hc.set_label('Mission time (s)')
    return fig, ax


def plot_wind_estimates(time, wind_array, wind_names=None, polar=False):
    '''
    Plot the wind estimates

    Parameters
    ----------
    time : np.array
        Time of the wind estimates
    wind_array : list of np.array
        Estimated wind along the flight path, multiple wind estimates are supported
    wind_names : list of str or None, default : None
        Labels for the different wind estimates
    polar : bool, default : False
        Plot the wind in polar coordinates instead of Cartesian wind components

    Returns
    -------
    fig: Figure
        Figure handle
    ax : Axes3DSubplot
        Axes handle
    '''
    if wind_names is None:
        wind_names = ['W{0:d}'.format(n) for n in range(len(wind_array))]
    f2, a2 = plt.subplots(3, 1)
    if polar:
        for wind in wind_array:
            mag, dir = rec2polar(wind[0], wind[1], wind_bearing=True, deg=True)
            a2[0].plot(time, mag)
            a2[1].plot(time, dir)
        a2[0].set_ylabel('$|V|$')
        a2[1].set_ylabel('$\Psi (deg)$')
    else:
        for wind in wind_array:
            a2[0].plot(time, wind[0])
            a2[1].plot(time, wind[1])
        a2[0].set_ylabel('$V_E$')
        a2[1].set_ylabel('$V_N$')
    a2[0].legend(wind_names)
    for wind in wind_array:
        a2[2].plot(time, wind[2])
    a2[2].set_ylabel('$V_D$')
    return f2, a2
