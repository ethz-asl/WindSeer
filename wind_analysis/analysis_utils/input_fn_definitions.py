import torch

def get_boundary_layer(n_cells, start_cell, roughness, device):
    profile = torch.zeros(n_cells).to(device)

    num_cells = n_cells - start_cell
    profile[start_cell:] = torch.log((torch.arange(num_cells).to(device) + roughness) / roughness)

    return profile

def bp_corners_1(params, terrain, interpolator, config):
    '''
    This function returns an input that is an interpolation from the boundary
    layers on each vertical corner. All the corners can be scaled and rotated
    by a shared scale and rotation angle.
    
    param[0]: scale []
    param[1]: rotation [rad]
    '''
    if len(terrain.shape) != 5:
        raise ValueError('The terrain must be a 5D tensor (batch, channel, nz, ny, nx]')

    if terrain.shape[0] != 1:
        raise ValueError('Only single batch operations are supported')

    if terrain.shape[1] != 1:
        raise ValueError('The terrain vector must contain only one channel')

    device = terrain.device
    input =  torch.zeros_like(terrain).repeat(1, interpolator.num_channels, 1, 1, 1).clone()
    n_z = terrain.squeeze().shape[0]

    for i in [0, -1]:
        for j in [0, -1]:
            start_idx = (terrain[:, :, :, i, j] == 0).sum()            
            roughness = config['roughness'] / (max(float(n_z - start_idx), 1.0))

            profile = get_boundary_layer(n_z, start_idx, roughness, device) * params[0]
            
            input[0, 0, :, i, j] = torch.cos(params[1]) * profile
            input[0, 1, :, i, j] = torch.sin(params[1]) * profile

    input = input.to(device)

    return interpolator.edge_interpolation(input[0]).unsqueeze(0)

def bp_corners_4(params, terrain, interpolator, config):
    '''
    This function returns an input that is an interpolation from the boundary
    layers on each vertical corner. All the corners can be scaled and rotated
    by a separate scale and rotation angle.
    
    param[0]: scale corner 0 []
    param[1]: scale corner 1 []
    param[2]: scale corner 2 []
    param[3]: scale corner 3 []
    param[4]: rotation corner 0 [rad]
    param[5]: rotation corner 1 [rad]
    param[6]: rotation corner 2 [rad]
    param[7]: rotation corner 3 [rad]
    '''
    if len(terrain.shape) != 5:
        raise ValueError('The terrain must be a 5D tensor (batch, channel, nz, ny, nx]')

    if terrain.shape[0] != 1:
        raise ValueError('Only single batch operations are supported')

    if terrain.shape[1] != 1:
        raise ValueError('The terrain vector must contain only one channel')

    device = terrain.device
    input =  torch.zeros_like(terrain).repeat(1, interpolator.num_channels, 1, 1, 1).clone()
    n_z = terrain.squeeze().shape[0]

    corner_idx = 0
    for i in [0, -1]:
        for j in [0, -1]:
            start_idx = (terrain[:, :, :, i, j] == 0).sum()            
            roughness = config['roughness'] / (max(float(n_z - start_idx), 1.0))

            profile = get_boundary_layer(n_z, start_idx, roughness, device) * params[corner_idx]

            input[0, 0, :, i, j] = torch.cos(params[corner_idx + 4]) * profile
            input[0, 1, :, i, j] = torch.sin(params[corner_idx + 4]) * profile

            corner_idx += 1

    input = input.to(device)

    return interpolator.edge_interpolation(input[0]).unsqueeze(0)

def bp_corners_1_roughness(params, terrain, interpolator, config):
    '''
    This function returns an input that is an interpolation from the boundary
    layers on each vertical corner. All the corners can be scaled and rotated
    by a shared scale and rotation angle. The roughness parameter of the
    boundary layer profile is adjusted as well.

    param[0]: scale []
    param[1]: rotation [rad]
    param[2]: roughness [cells]
    '''
    if len(terrain.shape) != 5:
        raise ValueError('The terrain must be a 5D tensor (batch, channel, nz, ny, nx]')

    if terrain.shape[0] != 1:
        raise ValueError('Only single batch operations are supported')

    if terrain.shape[1] != 1:
        raise ValueError('The terrain vector must contain only one channel')

    device = terrain.device
    input =  torch.zeros_like(terrain).repeat(1, interpolator.num_channels, 1, 1, 1).clone()
    n_z = terrain.squeeze().shape[0]

    for i in [0, -1]:
        for j in [0, -1]:
            start_idx = (terrain[:, :, :, i, j] == 0).sum()
            roughness = params[2] / (max(float(n_z - start_idx), 1.0))

            profile = get_boundary_layer(n_z, start_idx, roughness, device) * params[0]

            input[0, 0, :, i, j] = torch.cos(params[1]) * profile
            input[0, 1, :, i, j] = torch.sin(params[1]) * profile

    input = input.to(device)

    return interpolator.edge_interpolation(input[0]).unsqueeze(0)
