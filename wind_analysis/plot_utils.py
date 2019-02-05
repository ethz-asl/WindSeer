import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)


def vector_lims(uvw, axis=0):
    vflat = (uvw * uvw).sum(axis=axis).flatten()
    return np.sqrt([vflat.min(), vflat.max()])


def get_colors(uvw, cmap=plt.cm.hsv, Vlims=None):
    # Color by magnitude
    c = np.sqrt((uvw*uvw).sum(axis=0))
    # Flatten and normalize
    if Vlims is None:
        Vlims = (c.min(), c.max())
    c = (c.ravel() - Vlims[0]) / Vlims[1]
    # Repeat for each body line and two head lines
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    return cmap(c)


def plot_wind_3d(pos, wind, x_terr, y_terr, h_terr, cosmo_wind, origin=(0.0, 0.0, 0.0), wskip=5, Vlims=None):
    # Plot the wind vector estimates
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    ax.set_zlabel('Altitude (m)')

    X, Y = np.meshgrid(x_terr-origin[0], y_terr-origin[1])
    ax.plot_surface(X, Y, h_terr-origin[2], cmap=cm.terrain)

    cwind = np.array([cosmo_wind['wind_x'], cosmo_wind['wind_y'], cosmo_wind['wind_z']])
    if Vlims is None:
        wind_lims = vector_lims(wind)
        cw_lims = vector_lims(cwind)
        Vlims = (min(wind_lims[0], cw_lims[0]), max(wind_lims[1], cw_lims[1]))

    xx, yy, zz = pos[0]-origin[0], pos[1]-origin[1], pos[2]-origin[2]
    ax.plot(xx, yy, zz, 'k.', ms=1)
    wind_skip = wind[:, ::wskip]
    ax.quiver(xx[::wskip], yy[::wskip], zz[::wskip], wind_skip[0], wind_skip[1], wind_skip[2],
              colors=get_colors(wind_skip, Vlims=Vlims), length=5.0)

    # Plot cosmo wind
    # ax.plot(cosmo_wind['x'].flatten()-origin[0], cosmo_wind['y'].flatten()-origin[1], cosmo_wind['hsurf'].flatten()-origin[2], 'k.')
    ones_vec = np.ones(cwind.shape[1])
    for ix in range(2):
        for iy in range(2):
            cw = cwind[:,:,ix,iy]
            ax.quiver(cosmo_wind['x'][ix, iy]*ones_vec-origin[0], cosmo_wind['y'][ix, iy]*ones_vec-origin[1],
                      cosmo_wind['z'][:, ix, iy] - origin[2], cw[0], cw[1], cw[2], colors=get_colors(cw, Vlims=Vlims), length=5.0)

    norm = matplotlib.colors.Normalize()

    norm.autoscale(Vlims)

    sm = matplotlib.cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)
    # ax.set_xlim(xx.min(), xx.max())
    # ax.set_ylim(yy.min(), yy.max())
    # ax.set_zlim(zz.min(), zz.max())
    return fig, ax


def plot_cosmo_corners(ax, cosmo_corners, x_terr, y_terr, z_terr, origin=(0.0, 0.0, 0.0), Vlims=None):
    if Vlims is None:
        Vlims = vector_lims(cosmo_corners)

    ones_vec = np.ones(z_terr.shape)
    for yi in [0, -1]:
        for xi in [0, -1]:
            cw = np.array([cosmo_corners[0, :, yi, xi], cosmo_corners[1, :, yi, xi], cosmo_corners[2, :, yi, xi]])
            ax.quiver(x_terr[xi]*ones_vec - origin[0], y_terr[yi]*ones_vec - origin[1], z_terr - origin[2],
                      cw[0], cw[1], cw[2], colors=get_colors(cw, Vlims=Vlims), length=5.0)


def plot_vertical_profile(z_terr, cosmo_corner, wind_est, alt, t, fig=None, ax=None):
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    vv = np.sqrt(cosmo_corner[0,:]**2+ cosmo_corner[1,:]**2)
    ht, = ax.plot(vv, z_terr)
    v3 = np.sqrt(wind_est[0,:]**2+ wind_est[1,:]**2)
    h2 = ax.scatter(v3, alt, c=t, s=15)
    ax.grid(b=True, which='both')
    ax.set_xlabel('Wind speed (m/s)')
    ax.set_ylabel('Alt, m')
    ax.set_ylim(np.floor(alt.min()/100)*100, np.ceil(alt.max()/100)*100)
    hl = ax.legend([ht, h2], ['COSMO profile', 'UAV data'])
    hc = fig.colorbar(h2, ax=ax)
    hc.set_label('Mission time (s)')
    return fig, ax


def plot_lateral_variation(wind_est, pos, t, min_alt=None, max_alt=None, fig=None, ax=None, scale=100.0):
    if ax is None or fig is None:
        fig, ax = plt.subplots()
    dex = np.ones(pos[0].shape, dtype='bool')
    if min_alt is not None:
        dex = np.logical_and(dex, pos[2] > min_alt)
    if max_alt is not None:
        dex = np.logical_and(dex, pos[2] < max_alt)
    h2 = ax.quiver(pos[0][dex], pos[1][dex], wind_est[0][dex], wind_est[1][dex], t[dex], scale=scale)
    ax.grid(b=True, which='both')
    ax.plot(pos[0][0], pos[1][0], 'g^')
    ax.plot(pos[0][-1], pos[1][-1], 'ro')
    ax.set_xlabel('$x$ (East), m')
    ax.set_ylabel('$y$ (North), m')
    hc = fig.colorbar(h2, ax=ax)
    hc.set_label('Mission time (s)')
    return fig, ax


def rec2polar(vx, vy, wind_bearing=False, deg=False):
    mag = np.sqrt(vx**2 + vy**2)
    if wind_bearing:
        dir = np.pi/2 - np.arctan2(vy, vx)
    else:
        dir = np.arctan2(vy, vx)
    if deg:
        return mag, 360.0/np.pi*dir
    else:
        return mag, dir


def U_abl(z, z_0=0.5, U_ref=10.0, Z_ref=10.0, kappa=0.4, z_ground=0.0, U_star=None):
    if U_star is None:
        U_star = kappa*U_ref/(np.log((Z_ref+z_0)/z_0))
    U_star/kappa*np.log((z-z_ground+z_0)/z_0)
    return U_star/kappa*np.log((z-z_ground+z_0)/z_0)


def plot_wind_estimates(time, w0, w1, w0_name='W0', w1_name='W1', polar=False):
    f2, a2 = plt.subplots(3, 1)
    if polar:
        mag0, dir0 = rec2polar(w0[0], w0[1], wind_bearing=True, deg=True)
        mag1, dir1 = rec2polar(w1[0], w1[1], wind_bearing=True, deg=True)
        a2[0].plot(time, mag0)
        a2[0].plot(time, mag1)
        a2[0].set_ylabel('$|V|$')
        a2[0].legend([w0_name, w1_name])
        a2[1].plot(time, dir0)
        a2[1].plot(time, dir1)
        a2[1].set_ylabel('$\Psi (deg)$')
    else:
        a2[0].plot(time, w0[0])
        a2[0].plot(time, w1[0])
        a2[0].set_ylabel('$V_E$')
        a2[0].legend([w0_name, w1_name])
        a2[1].plot(time, w0[1])
        a2[1].plot(time, w1[1])
        a2[1].set_ylabel('$V_N$')
    a2[2].plot(time, w0[2])
    a2[2].plot(time, w1[2])
    a2[2].set_ylabel('$V_D$')
    return f2, a2
