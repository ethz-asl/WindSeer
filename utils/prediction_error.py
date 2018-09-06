import torch

def compute_prediction_error(label, prediction, u_hor_scale, u_ver_scale):
    abs_error = (label - prediction).abs()

    d3 = (len(list(abs_error.size())) > 4)

    # convert the velocities to m/s
    abs_error[:,0,:] *= u_hor_scale
    if d3:
        abs_error[:,1,:,:,:] *= u_hor_scale
        abs_error[:,2,:,:,:] *= u_ver_scale
    else:
        abs_error[:,1,:,:] *= u_ver_scale

    num_wind = abs_error[:,0,:].nonzero().shape[0]

    # get terrain height
    nonzero_idx = label[:,0,:].nonzero()

    if d3:
        terrain = torch.ones(abs_error.shape[3], abs_error.shape[4], dtype=torch.long) * (abs_error.shape[2]-1)
        for i in range(nonzero_idx.shape[0]):
            if (nonzero_idx[i, 1] < terrain[nonzero_idx[i, 2].item(), nonzero_idx[i, 3].item()]):
                terrain[nonzero_idx[i, 2].item(), nonzero_idx[i, 3].item()] = nonzero_idx[i, 1].item()
    else:
        terrain = torch.ones(abs_error.shape[3], dtype=torch.long) * (abs_error.shape[2]-1)

        for i in range(nonzero_idx.shape[0]):
            if (nonzero_idx[i, 1] < terrain[nonzero_idx[i, 2].item()]):
                terrain[nonzero_idx[i, 2].item()] = nonzero_idx[i, 1].item()

    low_error_x = 0.0
    low_error_y = 0.0
    low_error_z = 0.0
    high_error_x = 0.0
    high_error_y = 0.0
    high_error_z = 0.0
    max_low_x = 0.0
    max_low_y = 0.0
    max_low_z = 0.0
    max_high_x = 0.0
    max_high_y = 0.0
    max_high_z = 0.0

    if d3:
        for i in range(abs_error.shape[3]):
            for j in range(abs_error.shape[4]):
                low_error_x += abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j].sum()
                low_error_y += abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j].sum()
                low_error_z += abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j].sum()
                high_error_x += abs_error[:,0,terrain[i,j]+4:, i, j].sum()
                high_error_y += abs_error[:,1,terrain[i,j]+4:, i, j].sum()
                high_error_z += abs_error[:,2,terrain[i,j]+4:, i, j].sum()
                max_low_x = max(max_low_x, abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j].max().item())
                max_low_y = max(max_low_y, abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j].max().item())
                max_low_z = max(max_low_z, abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j].max().item())
                max_high_x = max(max_high_x, abs_error[:,1,terrain[i,j]+4:, i, j].max().item())
                max_high_y = max(max_high_y, abs_error[:,1,terrain[i,j]+4:, i, j].max().item())
                max_high_z = max(max_high_z, abs_error[:,2,terrain[i,j]+4:, i, j].max().item())

        low_error_x /= abs_error.shape[3] * abs_error.shape[4] * 4
        low_error_y /= abs_error.shape[3] * abs_error.shape[4] * 4
        low_error_z /= abs_error.shape[3] * abs_error.shape[4] * 4
        high_error_x /= num_wind - abs_error.shape[3] * abs_error.shape[4] * 4
        high_error_y /= num_wind - abs_error.shape[3] * abs_error.shape[4] * 4
        high_error_z /= num_wind - abs_error.shape[3] * abs_error.shape[4] * 4

        avg_abs_error = torch.sqrt(abs_error[:,0,:,:]**2 + abs_error[:,1,:,:]**2 + abs_error[:,2,:,:]**2).sum() / num_wind
        avg_abs_error_x = abs_error[:,0,:,:].sum() / num_wind
        avg_abs_error_y = abs_error[:,1,:,:].sum() / num_wind
        avg_abs_error_z = abs_error[:,2,:,:].sum() / num_wind
    else:
        for i in range(abs_error.shape[3]):
            low_error_x += abs_error[:,0,terrain[i]:terrain[i]+4, i].sum()
            low_error_z += abs_error[:,1,terrain[i]:terrain[i]+4, i].sum()
            high_error_x += abs_error[:,0,terrain[i]+4:, i].sum()
            high_error_z += abs_error[:,1,terrain[i]+4:, i].sum()
            max_low_x = max(max_low_x, abs_error[:,0,terrain[i]:terrain[i]+4, i].max().item())
            max_low_z = max(max_low_z, abs_error[:,1,terrain[i]:terrain[i]+4, i].max().item())
            max_high_x = max(max_high_x, abs_error[:,1,terrain[i]+4:, i].max().item())
            max_high_z = max(max_high_z, abs_error[:,1,terrain[i]+4:, i].max().item())

        low_error_x /= abs_error.shape[3] * 4
        low_error_z /= abs_error.shape[3] * 4
        high_error_x /= num_wind - abs_error.shape[3] * 4
        high_error_z /= num_wind - abs_error.shape[3] * 4
        low_error_y = torch.tensor(-1)
        high_error_y = torch.tensor(-1)
        max_low_y = -1
        max_high_y = -1

        avg_abs_error = torch.sqrt(abs_error[:,0,:,:]**2 + abs_error[:,1,:,:]**2).sum() / num_wind
        avg_abs_error_x = abs_error[:,0,:,:].sum() / num_wind
        avg_abs_error_y = torch.tensor(-1)
        avg_abs_error_z = abs_error[:,1,:,:].sum() / num_wind

    error_stats = {
        'avg_abs_error': avg_abs_error.item(),
        'avg_abs_error_x': avg_abs_error_x.item(),
        'avg_abs_error_y': avg_abs_error_y.item(),
        'avg_abs_error_z': avg_abs_error_z.item(),
        'low_error_x': low_error_x.item(),
        'low_error_y': low_error_y.item(),
        'low_error_z': low_error_z.item(),
        'high_error_x': high_error_x.item(),
        'high_error_y': high_error_y.item(),
        'high_error_z': high_error_z.item(),
        'max_low_x': max_low_x,
        'max_low_y': max_low_y,
        'max_low_z': max_low_z,
        'max_high_x': max_high_x,
        'max_high_y': max_high_y,
        'max_high_z': max_high_z,
        }

    return error_stats
