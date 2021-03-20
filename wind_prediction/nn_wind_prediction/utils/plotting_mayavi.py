try:
    from mayavi import mlab
    mayavi_available = True
except ImportError:
    print('mayavi could not get imported, disabling plotting with mayavi. Refer to the README for install instructions')
    mayavi_available = False

import numpy as np
import scipy

def mlab_plot_terrain(terrain, mode=0, uniform_color=False):
    '''
    Plot the terrain into the current figure
    '''
    if mayavi_available:
        # convert to numpy and permute axis such that the order is: [x,y,z]
        terrain_np = terrain.cpu().squeeze().permute(2,1,0).numpy()

        Z = (terrain_np == 0).sum(axis=2).astype(np.float)

        keep_idx = (terrain_np == 0)[:, :, 0]
        Z[~keep_idx] = np.NaN

        if mode == 0:
            indices = np.indices(terrain.cpu().squeeze().numpy().shape)
            X = indices[0, :, :, 0]
            Y = indices[1, :, :, 0]

            X = np.concatenate([X, np.flip(X, axis=0), X[0, None, :]], axis=0)
            Y = np.concatenate([Y, np.flip(Y, axis=0), Y[0, None, :]], axis=0)
            Z = np.concatenate([Z, np.flip(Z * 0.0 + 1.0, axis=0), Z[0, None, :]], axis=0)

            X = np.concatenate([X[:, 0, None], X, X[:,-1, None]], axis=1)
            Y = np.concatenate([Y[:, 0, None], Y, Y[:,-1, None]], axis=1)
            Z = np.concatenate([Z[:, 0, None] * 0.0  + 1.0, Z, Z[:,-1, None] * 0.0  + 1.0], axis=1)

            if uniform_color:
                mlab.mesh(X, Y, Z, representation='surface', mode='cube', color=(160.0/255.0 ,82.0/255.0 ,45.0/255.0))

            else:
                mlab.mesh(X, Y, Z, representation='surface', mode='cube')

        elif mode == 1:
            if uniform_color:
                mlab.barchart(Z, color=(160.0/255.0 ,82.0/255.0 ,45.0/255.0))
            else:
                mlab.barchart(Z)

        else:
            raise ValueError('Unknown terrain plotting mode')

def mlab_plot_measurements(measurements, mask, terrain, terrain_mode=0, terrain_uniform_color=False, blocking=True):
    '''
    Visualize the measurements using mayavi
    The inputs are assumed to be torch tensors.
    '''
    if mayavi_available:
        measurements_np = measurements.cpu().squeeze().numpy()
        mask_np = terrain.cpu().squeeze().numpy()
        measurement_idx = mask.squeeze().nonzero().cpu().numpy()

        mlab.figure()
        mlab_plot_terrain(terrain, terrain_mode, terrain_uniform_color)

        wind_vel = measurements_np[:,measurement_idx[:, 0], measurement_idx[:, 1], measurement_idx[:, 2]]

        mlab.quiver3d(measurement_idx[:, 2], measurement_idx[:, 1], measurement_idx[:, 0], wind_vel[0], wind_vel[1], wind_vel[2])

        if blocking:
            mlab.show()
