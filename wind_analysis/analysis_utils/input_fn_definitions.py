import torch

def get_boundary_layer(n_cells, start_cell, roughness, device):
    '''
    Logarithmic boundary layer profile according to:
    https://www.openfoam.com/documentation/guides/latest/doc/guide-bcs-inlet-atm-atmBoundaryLayer.html

    Roughness is in the number of cells (i.e. normalized by cell height)
    '''
    profile = torch.zeros(n_cells).to(device)

    num_cells = n_cells - start_cell
    profile[start_cell:] = torch.log((torch.arange(num_cells).to(device) + roughness) / roughness)

    return profile


def spline_interpolation(x, y, xs):
    '''
    Cubic hermite spline interpolation (https://en.wikipedia.org/wiki/Cubic_Hermite_spline)
    x, y denote the input control points, xs are the positions where the function is interpolated
    '''
    if x.max() < xs.max():
        raise ValueError('Requested xs exceeds provided x')

    if x.min() > xs.min():
        raise ValueError('Requested xs lower than provided x')

    # sort x
    x_sorted, indices = torch.sort(x)
    y_sorted = y[indices]

    # compute the finite differences
    m = (y_sorted[1:] - y_sorted[:-1])/(x_sorted[1:] - x_sorted[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) * 0.5, m[[-1]]])

    # normalize the sampling coordinates
    indices = torch.searchsorted(x[1:], xs)
    dx = (x_sorted[indices + 1] - x_sorted[indices])
    normalized_x = (xs - x_sorted[indices]) / dx

    # build the Hermite basis functions
    coefficients = torch.tensor([[1, 0, -3, 2],
                                 [0, 1, -2, 1],
                                 [0, 0, 3, -2],
                                 [0, 0, -1, 1]]).to(y.dtype).to(y.device)
    t = [None for _ in range(4)]
    t[0] = torch.tensor([1.0]).to(y.dtype).to(y.device)
    for i in range(1, 4):
        t[i] = t[i-1] * normalized_x

    h_fn = [sum(coefficients[i, j]*t[j] for j in range(4)) for i in range(4)]

    return h_fn[0] * y_sorted[indices] + h_fn[1] * m[indices] * dx + h_fn[2] * y_sorted[indices + 1] + h_fn[3] * m[indices + 1] * dx

def get_spline_profile(n_z, start_idx, params, device):
    '''
    Generate an input profile for one corner using the input parameter as control points for a spline
    interpolation.

    The control points are linearly distributed along the vertical axis. The first control point is located at
    the start_idx and the last one at the top cell.
    '''
    profile = torch.zeros(params.shape[0], n_z).to(device)

    x_sample = torch.arange(n_z - start_idx).to(device).to(params.dtype)

    start = 0.0
    end = n_z - start_idx - 1.0
    eps = 0.001
    step_size = (end - start) / (params.shape[1] - 1.0)

    # adding a small eps as adviced in the docs
    x_input = torch.arange(start, end + eps, step_size).to(params.device)

    for i in range(params.shape[0]):
        profile[i, start_idx:] = spline_interpolation(x_input, params[i], x_sample)

    return profile

def bp_corners(params, terrain, interpolator, config):
    '''
    This function returns an input that is an interpolation from the boundary
    layers on each vertical corner. All the corners can be scaled and rotated.
    Optionally the aerodynamic roughness length can be optimized as well if the
    second dimension of the parameter vector has length 3.
    If the first dimension of the params vector has size 1 then for all
    corners the parameter are used, else individual parameter for each
    corner are used.

    params is a tensor with shape: [N, P], where N is either 1 or 4 and P either 2 or 3
    '''

    assert len(terrain.shape) is 5, 'The terrain must be a 5D tensor [batch, channel, nz, ny, nx]'

    assert terrain.shape[0] == 1, 'Only single batch operations are supported'

    assert terrain.shape[1] == 1, 'The terrain vector must contain only one channel'

    assert len(params.shape) == 2, 'The shape of the parameter vector must have exactly two dimensions'

    assert list(params.shape) in [[1, 2], [1, 3], [4, 2], [4, 3]], 'The shape of the parameter vector must be either [1, 2], [1, 3], [4, 2], or [4, 3]'


    device = terrain.device
    input =  torch.zeros_like(terrain).repeat(1, interpolator.num_channels, 1, 1, 1).clone()
    n_z = terrain.squeeze().shape[0]

    corner_idx = 0
    for i in [0, -1]:
        for j in [0, -1]:
            start_idx = (terrain[:, :, :, i, j] == 0).sum()
            if len(params[corner_idx]) > 2:
                roughness = params[corner_idx, 2]
            else:
                roughness = config['roughness']

            # roughness normalized by the actual height of the column
            normalized_roughness = roughness / (max(float(n_z - start_idx), 1.0))

            profile = get_boundary_layer(n_z, start_idx, normalized_roughness, device) * params[corner_idx, 0]

            input[0, 0, :, i, j] = torch.cos(params[corner_idx, 1]) * profile
            input[0, 1, :, i, j] = torch.sin(params[corner_idx, 1]) * profile

            if params.shape[0] > 1:
                corner_idx += 1

    input = input.to(device)

    return interpolator.edge_interpolation(input[0]).unsqueeze(0)

def splines_corner(params, terrain, interpolator, config):
    '''
    Inflow profile where the parameter are the control points for a
    spline interpolation.

    The control points are linearly distributed along the vertical axis.
    The first control point is located at the first non-terrain cell and the
    last one at the top cell.

    params is a tensor with shape: [4, num_channels, n_control_points]
    '''
    assert len(terrain.shape) is 5, 'The terrain must be a 5D tensor [batch, channel, nz, ny, nx]'

    assert terrain.shape[0] == 1, 'Only single batch operations are supported'

    assert terrain.shape[1] == 1, 'The terrain vector must contain only one channel'

    device = terrain.device
    input =  torch.zeros_like(terrain).repeat(1, interpolator.num_channels, 1, 1, 1).clone()
    n_z = terrain.squeeze().shape[0]

    corner_idx = 0
    for i in [0, -1]:
        for j in [0, -1]:
            start_idx = (terrain[:, :, :, i, j] == 0).sum()

            input[0, :params[corner_idx].shape[0], :, i, j] = get_spline_profile(n_z, start_idx, params[corner_idx], device)

            corner_idx += 1

    input = input.to(device)

    return interpolator.edge_interpolation(input[0]).unsqueeze(0)
