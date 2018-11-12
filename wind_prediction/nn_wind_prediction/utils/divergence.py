import numpy as np
import torch

def divergence(x, ds, terrain):
    '''
    The first dimension of x is the three velocities, then x-, y-, and z-direction.
    ds specifies the grid spacing of x.
    '''
    terrain_local = terrain.clone()

    # convert terrain to a binary terrain
    terrain_local.sign_()

    # generate terrain mask
    terrain_mask = torch.zeros(terrain_local.shape)
    terrain_mask[1:-1,1:-1,:-1] = terrain_local[1:-1,1:-1,:-1] * terrain_local[:-2,1:-1,:-1] * terrain_local[2:,1:-1,:-1] * terrain_local[1:-1,:-2,:-1] * terrain_local[1:-1,2:,:-1] * terrain_local[1:-1,1:-1,1:]

    # compute the gradient with uniform grid size
    u_x = (x[0,:,:,1:] - x[0,:,:,:-1]) / float(ds[0])
    v_y = (x[1,:,1:,:] - x[1,:,:-1,:]) / float(ds[1])
    w_z = (x[2,1:,:,:] - x[2,:-1,:,:]) / float(ds[2])

    div = u_x[:-1,:-1,:] + v_y[:-1,:,:-1] + w_z[:,:-1,:-1]
    x_shape = x.shape

    div = torch.cat([div, torch.zeros(1, x.shape[2]-1, x.shape[3]-1)], dim=0)
    div = torch.cat([div, torch.zeros(x.shape[1], 1, x.shape[3]-1)], dim=1)
    div = torch.cat([div, torch.zeros(x.shape[1], x.shape[2], 1)], dim=2)

#     u_x, u_y, u_z = np.gradient(x[0,:,:,:])
#     v_x, v_y, v_z = np.gradient(x[1,:,:,:])
#     w_x, w_y, w_z = np.gradient(x[2,:,:,:])
#
#     # compute the divergence with the correct scaled gradients
#     div = u_x / ds[0] + v_y / ds[1] + w_z / ds[2]
    return div * terrain_mask
