import torch.nn as nn
import torch

def derive(input_tensor, deriv_axis, ds=1):
    '''
    This function computes the derivative of a scalar field, with regard to a specified axis direction

    Input params:
        input_tensor: 4D tensor (samples x Depth x Height x Width)
        deriv_axis: axis along which to compute the derivative
        ds: grid size of deriv_axis, default is unit.

    Output:
        deriv_tensor: 4D tensor containing the derived field wrt to deriv_axis for each sample
                Implementation of first order centered finite differences. 0 padding for first and last element
    '''
    # Permute axes depending on deriv_axs
    if deriv_axis == 1:
        input_tensor = input_tensor.permute(0, 3, 2, 1)

    elif deriv_axis == 2:
        input_tensor = input_tensor.permute(0, 1, 3, 2)

    elif deriv_axis == 3:
        input_tensor = input_tensor.permute(0, 1, 2, 3)

    else:
        raise ValueError('The derivation axis must be 1, 2 or 3')

    # setting device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    deriv_tensor[:, :, :, 0] = deriv_tensor[:, :, :, 0].clone() * 2
    deriv_tensor[:, :, :, -1] = deriv_tensor[:, :, :, -1].clone() * 2

    # permute output of convolution back
    if deriv_axis == 1:
        deriv_tensor = deriv_tensor.permute(0, 3, 2, 1)

    elif deriv_axis == 2:
        deriv_tensor = deriv_tensor.permute(0, 1, 3, 2)

    return deriv_tensor

def curl(input_tensor, ds=1):
    '''
    This function computes the curl of a vector field, with regard to a specified grid size

    Input params:
        input_tensor: 5D tensor (samples, input_field[phix_in, phiy_in, phiz_in]), X, Y, Z)
        ds: grid size, default is unit.

    Output:
        curled_tensor: 5D tensor (samples, curled_field[u, v, w]), X, Y, Z)
    '''
    phix_y = derive(input_tensor[:, 0, :, :, :], 2, ds)
    phix_z = derive(input_tensor[:, 0, :, :, :], 3, ds)

    phiy_x = derive(input_tensor[:, 1, :, :, :], 1, ds)
    phiy_z = derive(input_tensor[:, 1, :, :, :], 3, ds)

    phiz_x = derive(input_tensor[:, 2, :, :, :], 1, ds)
    phiz_y = derive(input_tensor[:, 2, :, :, :], 2, ds)

    # curled vector field components
    u = phiz_y - phiy_z
    v = phix_z - phiz_x
    w = phiy_x - phix_y

    # concat all components as well as terrain (first channel)
    curled_tensor = torch.cat((u.unsqueeze(1), v.unsqueeze(1), w.unsqueeze(1)), 1)
    return curled_tensor