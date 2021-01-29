import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from nn_wind_prediction.utils import PlotUtils
import nn_wind_prediction.utils as nn_utils

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
    # There is a problem where zero-length vectors are removed, offsetting all the correspnding colours
    # Repeat for each body line and two head lines
    c = c[c != 0.0]
    c = np.concatenate((c, np.repeat(c, 2)))
    # Colormap
    return cmap(c)


def plot_wind_3d(pos, wind, x_terr, y_terr, h_terr, cosmo_wind, origin=(0.0, 0.0, 0.0), wskip=5, Vlims=None, plot_cosmo=True):
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
    if plot_cosmo:
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
    hc = fig.colorbar(sm, ax=ax)
    hc.set_label('Wind speed (m/s)')
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

def plot_optimizer_results(results):
        # visualize the results
    import matplotlib.pyplot as plt

    # plot the losses
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    for i, loss in enumerate(results['losses']):
        x = np.arange(len(loss))
        plt.plot(x, loss, label = results['optimizers'][i]['name'])
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend(loc='best')

    # plot the parameter and the gradients
    max_figs_per_figure = 4
    num_params = results['parameter'][0].shape[2]
    num_fig = int(np.ceil(num_params / max_figs_per_figure))

    for co in range(results['gradients'][0].shape[1]):
        for i in range(num_fig):
            num_plots = min(max_figs_per_figure, num_params - i * max_figs_per_figure)

            fig, ah = plt.subplots(2, num_plots, squeeze=False)

            if results['gradients'][0].shape[1] > 1:
                fig.suptitle('Parameter Corner ' + str(co))
            else:
                fig.suptitle('Parameter for all Corners')

            for j in range(len(results['gradients'])):
                for k in range(num_plots):
                    x = np.arange(len(results['parameter'][j][:, co, i * max_figs_per_figure + k]))
                    ah[0][k].plot(x, results['parameter'][j][:, co, i * max_figs_per_figure + k].numpy(), label = results['optimizers'][j]['name'])
                    ah[1][k].plot(x, results['gradients'][j][:, co, i * max_figs_per_figure + k].numpy(), label = results['optimizers'][j]['name'])
                    ah[1][k].set_xlabel('Iteration')
                    ah[0][k].set_title('Parameter ' + str(i * max_figs_per_figure + k))

            ah[0][0].set_ylabel('Parameter Value')
            ah[1][0].set_ylabel('Gradients')
            plt.legend(loc='best')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # plot the corner profiles
    corner_counter = 0
    fig, ah = plt.subplots(results['input'][1:].shape[0], 4)
    fig.suptitle('Velocity Profiles')
    for i in [0, -1]:
        for j in [0, -1]:
            corner_input = results['input'][1: , :, i, j].cpu().detach().numpy()
            height = np.arange(corner_input.shape[1])

            if results['ground_truth'] is not None:
                corner_gt = results['ground_truth'][:, :, i, j].cpu().detach().numpy()

            xlabels = ['ux', 'uy', 'uz']

            for k in range(corner_input.shape[0]):
                ah[k][corner_counter].plot(corner_input[k], height, label = 'prediction')
                if results['ground_truth'] is not None:
                    ah[k][corner_counter].plot(corner_gt[k], height, label = 'ground truth')

                ah[k][corner_counter].set_xlabel(xlabels[k] + ' corner ' + str(corner_counter))
                ah[k][0].set_ylabel('Height [cells]')

            corner_counter += 1

    if results['ground_truth'] is not None:
        plt.legend(loc='best')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if results['mask'] is not None:
        results['mask'] = results['mask'].squeeze()

    nn_utils.plot_prediction(results['label_channels'],
                             prediction = results['prediction'],
                             label = results['ground_truth'],
                             provided_input_channels = results['input_channels'],
                             input = results['input'],
                             terrain = results['terrain'].squeeze(),
                             measurements = results['measurement'],
                             measurements_mask = results['mask'])

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


def plot_wind_estimates(time, wind_array, wind_names=None, polar=False):
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


def plot_prediction_observations(input, label, terrain, save, add_sparse_mask_row, masked_input):
    i2 = input
    instance = PlotUtils('prediction', ['ux', 'uy', 'uz'],
                         ['ux', 'uy', 'uz'], i2, label, terrain, design=1, masked_input=masked_input)
    fig, ax = instance.plot_prediction(label_name='Observed wind', save=save, add_sparse_mask_row=add_sparse_mask_row)
    return fig, ax
