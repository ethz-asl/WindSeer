import numpy as np

def divergence(x, ds):
    '''
    The first dimension of x is the three velocities, then x-, y-, and z-direction.
    ds specifies the grid spacing of x.
    '''

    # compute the gradient with uniform grid size
    u_x, u_y, u_z = np.gradient(x[0,:,:,:])
    v_x, v_y, v_z = np.gradient(x[1,:,:,:])
    w_x, w_y, w_z = np.gradient(x[2,:,:,:])

    # compute the divergence with the correct scaled gradients
    div = u_x / ds[0] + v_y / ds[1] + w_z / ds[2]
    return div
