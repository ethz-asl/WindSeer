import torch

def compute_prediction_error(label, prediction, u_hor_scale, u_ver_scale, uncertainty_predicted):
    if uncertainty_predicted:
        abs_error = (label - prediction[:,-1,:]).abs()
    else:
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

    low_error_hor = 0.0
    low_error_vert = 0.0
    low_error_tot = 0.0
    high_error_hor = 0.0
    high_error_vert = 0.0
    high_error_tot = 0.0
    max_low_hor = 0.0
    max_low_vert = 0.0
    max_low_tot = 0.0
    max_high_hor = 0.0
    max_high_vert = 0.0
    max_high_tot = 0.0

    if d3:
        for i in range(abs_error.shape[3]):
            for j in range(abs_error.shape[4]):
                low_error_hor += (abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] +
                                  abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j]).sqrt().sum()
                low_error_vert += abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j].sum()
                low_error_tot += (abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] +
                                  abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j] +
                                  abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j]).sqrt().sum()
                high_error_hor += (abs_error[:,0,terrain[i,j]+4:, i, j] * abs_error[:,0,terrain[i,j]+4:, i, j] +
                                   abs_error[:,1,terrain[i,j]+4:, i, j] * abs_error[:,1,terrain[i,j]+4:, i, j]).sqrt().sum()
                high_error_vert += abs_error[:,2,terrain[i,j]+4:, i, j].sum()
                high_error_tot += (abs_error[:,0,terrain[i,j]+4:, i, j] * abs_error[:,0,terrain[i,j]+4:, i, j] +
                                   abs_error[:,1,terrain[i,j]+4:, i, j] * abs_error[:,1,terrain[i,j]+4:, i, j] +
                                   abs_error[:,2,terrain[i,j]+4:, i, j] * abs_error[:,2,terrain[i,j]+4:, i, j]).sqrt().sum()
                max_low_hor = max(max_low_hor,
                                  (abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] +
                                  abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j]).sqrt().max().item())
                max_low_vert = max(max_low_vert, abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j].max().item())
                max_low_tot = max(max_low_tot,
                                  (abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,0,terrain[i,j]:terrain[i,j]+4, i, j] +
                                  abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,1,terrain[i,j]:terrain[i,j]+4, i, j] +
                                  abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j] * abs_error[:,2,terrain[i,j]:terrain[i,j]+4, i, j]).sqrt().max().item())
                max_high_hor = max(max_high_hor,
                                   (abs_error[:,0,terrain[i,j]+4:, i, j] * abs_error[:,0,terrain[i,j]+4:, i, j] +
                                   abs_error[:,1,terrain[i,j]+4:, i, j] * abs_error[:,1,terrain[i,j]+4:, i, j]).sqrt().max().item())
                max_high_vert = max(max_high_vert, abs_error[:,2,terrain[i,j]+4:, i, j].max().item())
                max_high_tot = max(max_high_tot,
                                   (abs_error[:,0,terrain[i,j]+4:, i, j] * abs_error[:,0,terrain[i,j]+4:, i, j] +
                                   abs_error[:,1,terrain[i,j]+4:, i, j] * abs_error[:,1,terrain[i,j]+4:, i, j] +
                                   abs_error[:,2,terrain[i,j]+4:, i, j] * abs_error[:,2,terrain[i,j]+4:, i, j]).sqrt().max().item())

        low_error_hor /= abs_error.shape[3] * abs_error.shape[4] * 4
        low_error_vert /= abs_error.shape[3] * abs_error.shape[4] * 4
        low_error_tot /= abs_error.shape[3] * abs_error.shape[4] * 4
        high_error_hor /= num_wind - abs_error.shape[3] * abs_error.shape[4] * 4
        high_error_vert /= num_wind - abs_error.shape[3] * abs_error.shape[4] * 4
        high_error_tot /= num_wind - abs_error.shape[3] * abs_error.shape[4] * 4

        avg_abs_error = torch.sqrt(abs_error[:,0,:,:]**2 + abs_error[:,1,:,:]**2 + abs_error[:,2,:,:]**2).sum() / num_wind
        avg_abs_error_x = abs_error[:,0,:,:].sum() / num_wind
        avg_abs_error_y = abs_error[:,1,:,:].sum() / num_wind
        avg_abs_error_z = abs_error[:,2,:,:].sum() / num_wind
    else:
        for i in range(abs_error.shape[3]):
            low_error_hor += abs_error[:,0,terrain[i]:terrain[i]+4, i].sum()
            low_error_vert += abs_error[:,1,terrain[i]:terrain[i]+4, i].sum()
            low_error_tot += (abs_error[:,0,terrain[i]:terrain[i]+4, i] * abs_error[:,0,terrain[i]:terrain[i]+4, i] +
                              abs_error[:,1,terrain[i]:terrain[i]+4, i] * abs_error[:,1,terrain[i]:terrain[i]+4, i]).sqrt().sum()
            high_error_hor += abs_error[:,0,terrain[i]+4:, i].sum()
            high_error_vert += abs_error[:,1,terrain[i]+4:, i].sum()
            high_error_tot += (abs_error[:,0,terrain[i]+4:, i] * abs_error[:,0,terrain[i]+4:, i] +
                               abs_error[:,1,terrain[i]+4:, i] * abs_error[:,1,terrain[i]+4:, i]).sqrt().sum()
            max_low_hor = max(max_low_hor, abs_error[:,0,terrain[i]:terrain[i]+4, i].max().item())
            max_low_vert = max(max_low_vert, abs_error[:,1,terrain[i]:terrain[i]+4, i].max().item())
            max_low_tot = max(max_low_tot,
                              (abs_error[:,0,terrain[i]:terrain[i]+4, i] * abs_error[:,0,terrain[i]:terrain[i]+4, i] +
                              abs_error[:,1,terrain[i]:terrain[i]+4, i] * abs_error[:,1,terrain[i]:terrain[i]+4, i]).sqrt().max().item())
            max_high_hor = max(max_high_x, abs_error[:,1,terrain[i]+4:, i].max().item())
            max_high_vert = max(max_high_z, abs_error[:,1,terrain[i]+4:, i].max().item())
            max_high_tot = max(max_high_tot,
                               (abs_error[:,0,terrain[i]+4:, i] * abs_error[:,0,terrain[i]+4:, i] +
                               abs_error[:,1,terrain[i]+4:, i] * abs_error[:,1,terrain[i]+4:, i]).sqrt().max().item())

        low_error_hor /= abs_error.shape[3] * 4
        low_error_vert /= abs_error.shape[3] * 4
        low_error_tot /= abs_error.shape[3] * 4
        high_error_hor /= num_wind - abs_error.shape[3] * 4
        high_error_vert /= num_wind - abs_error.shape[3] * 4
        high_error_tot /= num_wind - abs_error.shape[3] * 4

        avg_abs_error = torch.sqrt(abs_error[:,0,:,:]**2 + abs_error[:,1,:,:]**2).sum() / num_wind
        avg_abs_error_x = abs_error[:,0,:,:].sum() / num_wind
        avg_abs_error_y = torch.tensor(-1)
        avg_abs_error_z = abs_error[:,1,:,:].sum() / num_wind

    error_stats = {
        'avg_abs_error': avg_abs_error.item(),
        'avg_abs_error_x': avg_abs_error_x.item(),
        'avg_abs_error_y': avg_abs_error_y.item(),
        'avg_abs_error_z': avg_abs_error_z.item(),
        'low_error_hor': low_error_hor.item(),
        'low_error_vert': low_error_vert.item(),
        'low_error_tot': low_error_tot.item(),
        'high_error_hor': high_error_hor.item(),
        'high_error_vert': high_error_vert.item(),
        'high_error_tot': high_error_tot.item(),
        'max_low_hor': max_low_hor,
        'max_low_vert': max_low_vert,
        'max_low_tot': max_low_tot,
        'max_high_hor': max_high_hor,
        'max_high_vert': max_high_vert,
        'max_high_tot': max_high_tot,
        }

    return error_stats
