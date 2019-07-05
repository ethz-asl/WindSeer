import torch.nn as nn
import torch

def derive(input_tensor, deriv_axis, ds=1, terrain=None, no_nans=True):
    '''
    This function computes the derivative of a scalar field, with regard to a specified axis direction

    Input params:
        input_tensor: 4D tensor [samples, Z, Y, X]
        deriv_axis: axis along which to compute the derivative
        ds: grid size of deriv_axis, default is unit.
        terrain: matrix of same shape as input_tensor which provides the terrain input, if passed the derive function
                    will ignore cells right next to the terrain.
        no_nans: if True all nans in the output due to the terrain mask will be removed

    Output:
        deriv_tensor: 4D tensor containing the derived field wrt to deriv_axis for each sample.
    '''
    # account for terrain, the cells near the terrain will be excluded
    if terrain is not None:
        is_terrain = 1 - terrain.sign_()
        input_tensor = input_tensor.clone()
        input_tensor[is_terrain.byte()] = torch.tensor(float('nan'))

    # Permute axes depending on deriv_axs, no permutation required for X
    if deriv_axis == 1:  # Z
        input_tensor = input_tensor.permute(0, 3, 2, 1)

    elif deriv_axis == 2:  # Y
        input_tensor = input_tensor.permute(0, 1, 3, 2)

    elif deriv_axis != 3:
        raise ValueError('The derivation axis must be 1 [Z], 2 [Y] or 3 [X]')

    # setting device
    device = input_tensor.device

    # setting up convolution equivalent to centered first order finite differences
    deriv_conv = nn.Conv2d(in_channels=input_tensor.shape[1], out_channels=input_tensor.shape[1],
                           kernel_size=[1, 3], stride=1, padding=[0, 0], groups=input_tensor.shape[1], bias=False)
    kernel = deriv_conv.weight.data
    kernel[:, :, :, 0] = -1
    kernel[:, :, :, 1] = 0
    kernel[:, :, :, 2] = 1
    deriv_conv.weight.data = kernel.to(device)
    deriv_conv.weight.requires_grad = False

    # define padding
    m = nn.ReplicationPad2d((1, 1, 0, 0))

    # apply convolution
    deriv_tensor = deriv_conv(m(input_tensor)) / (2 * ds)

    # multiply by 2 on edges (finite diff non-centered on edges, similar to np.gradient)
    deriv_tensor[:, :, :, 0] *= 2
    deriv_tensor[:, :, :, -1] *= 2

    # permute output of convolution back
    if deriv_axis == 1:
        deriv_tensor = deriv_tensor.permute(0, 3, 2, 1)

    elif deriv_axis == 2:
        deriv_tensor = deriv_tensor.permute(0, 1, 3, 2)

    # remove nans
    if no_nans:
        deriv_tensor[torch.isnan(deriv_tensor)] = 0.0

    return deriv_tensor

def curl(input_tensor, grid_size, terrain=None):
    '''
    This function computes the curl of a vector field, with regard to a specified grid size

    Input params:
        input_tensor: 5D tensor [samples, input_field(phix_in, phiy_in, phiz_in), Z, Y, X]
        grid_size: list/array which contains the grid sizes in X, Y and Z
        terrain: 5D tensor containing terrain channel

    Output:
        curled_tensor: 5D tensor [samples, curled_field(u, v, w), Z, Y, X]
    '''
    phix_y = derive(input_tensor[:, 0, :, :, :], 2, grid_size[1], terrain, False)
    phix_z = derive(input_tensor[:, 0, :, :, :], 1, grid_size[2], terrain, False)

    phiy_x = derive(input_tensor[:, 1, :, :, :], 3, grid_size[0], terrain, False)
    phiy_z = derive(input_tensor[:, 1, :, :, :], 1, grid_size[2], terrain, False)

    phiz_x = derive(input_tensor[:, 2, :, :, :], 3, grid_size[0], terrain, False)
    phiz_y = derive(input_tensor[:, 2, :, :, :], 2, grid_size[1], terrain, False)

    # curled vector field components
    u = phiz_y - phiy_z
    v = phix_z - phiz_x
    w = phiy_x - phix_y

    # concat all components (2d dimension)
    curled_tensor = torch.cat((u.unsqueeze(1), v.unsqueeze(1), w.unsqueeze(1)), 1)
    curled_tensor[torch.isnan(curled_tensor)] = 0.0
    return curled_tensor


def gradient(input_tensor, grid_size, terrain=None):
    '''
    This function computes the gradient (2nd order tensor) of a vector at each XYZ position, with regard to a specified grid size

    Input params:
        input_tensor: 5D tensor [samples, input_field(u_in, v_in, w_in), Z, Y, X]
        grid_size: list/array which contains the grid sizes in X, Y and Z
        terrain: 5D tensor containing terrain channel

    Output:
        gradient_tensor: 5D tensor [samples, gradient_components(9 in total), Z, Y, X]
    '''
    u_x = derive(input_tensor[:, 0, :, :, :], 3, grid_size[0], terrain).unsqueeze(1)
    u_y = derive(input_tensor[:, 0, :, :, :], 2, grid_size[1], terrain).unsqueeze(1)
    u_z = derive(input_tensor[:, 0, :, :, :], 1, grid_size[2], terrain).unsqueeze(1)

    v_x = derive(input_tensor[:, 1, :, :, :], 3, grid_size[0], terrain).unsqueeze(1)
    v_y = derive(input_tensor[:, 1, :, :, :], 2, grid_size[1], terrain).unsqueeze(1)
    v_z = derive(input_tensor[:, 1, :, :, :], 1, grid_size[2], terrain).unsqueeze(1)

    w_x = derive(input_tensor[:, 2, :, :, :], 3, grid_size[0], terrain).unsqueeze(1)
    w_y = derive(input_tensor[:, 2, :, :, :], 2, grid_size[1], terrain).unsqueeze(1)
    w_z = derive(input_tensor[:, 2, :, :, :], 1, grid_size[2], terrain).unsqueeze(1)

    gradient_tensor = torch.cat((u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z), 1)
    return gradient_tensor

def divergence(input_tensor, grid_size, terrain=None):
    '''
    This function computes the divergence of a vector at each XYZ position, with regard to a specified grid size

    Input params:
        input_tensor: 5D tensor [samples, input_field(u_in, v_in, w_in), Z, Y, X]
        grid_size: list/array which contains the grid sizes in X, Y and Z
        terrain: 5D tensor containing terrain channel

    Output:
        divergence_tensor: 4D tensor of divergence of input field [samples, Z, Y, X]
    '''
    u_x = derive(input_tensor[:, 0, :, :, :], 3, grid_size[0], terrain,False)
    v_y = derive(input_tensor[:, 1, :, :, :], 2, grid_size[1], terrain, False)
    w_z = derive(input_tensor[:, 2, :, :, :], 1, grid_size[2], terrain, False)

    divergence_tensor = u_x + v_y + w_z
    divergence_tensor[torch.isnan(divergence_tensor)] = 0.0
    return divergence_tensor