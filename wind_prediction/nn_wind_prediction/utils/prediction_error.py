import numpy as np
from scipy import ndimage
import torch

threshold_low = 7.0 # this corresponds to roughly 80 m

def compute_prediction_error(label, prediction, terrain, uncertainty_predicted, device, turbulence):
    if uncertainty_predicted:
        abs_error = (label - prediction[:label.shape[0]]).abs()
    else:
        abs_error = (label - prediction).abs()

    d3 = (len(list(abs_error.size())) > 3)

    # convert the boolean terrain into a distance field if required
    if (terrain.max().item() == 1.0) and (torch.mul(torch.gt(terrain, torch.zeros_like(terrain)), torch.gt(torch.ones_like(terrain), terrain)).sum() == 0):
        terrain = torch.from_numpy(ndimage.distance_transform_edt(terrain.cpu().numpy()).astype(np.float32)).to(device)

    # create the masks
    mask_low = torch.mul((terrain <= threshold_low), (terrain > 0.0))
    mask_high = (terrain > threshold_low)
    mask_wind = (terrain > 0.0)

    # extract the properties
    if d3:
        error_tot = abs_error[:3].norm(dim=0)
        error_hor = abs_error[:2].norm(dim=0)
        error_ver = abs_error[2]
        if turbulence:
            error_turb = abs_error[3]

    else:
        error_tot = abs_error[:2].norm(dim=0)
        error_hor = abs_error[0].norm(dim=0)
        error_ver = abs_error[1]
        if turbulence:
            error_turb = abs_error[2]

    # error properties over the full domain
    all_tot_mean = torch.masked_select(error_tot, mask_wind).mean()
    all_tot_max = torch.masked_select(error_tot, mask_wind).max()
    all_tot_median = torch.masked_select(error_tot, mask_wind).median()

    all_hor_mean = torch.masked_select(error_hor, mask_wind).mean()
    all_hor_max = torch.masked_select(error_hor, mask_wind).max()
    all_hor_median = torch.masked_select(error_hor, mask_wind).median()

    all_ver_mean = torch.masked_select(error_ver, mask_wind).mean()
    all_ver_max = torch.masked_select(error_ver, mask_wind).max()
    all_ver_median = torch.masked_select(error_ver, mask_wind).median()

    if turbulence:
        all_turb_mean = torch.masked_select(error_turb, mask_wind).mean()
        all_turb_max = torch.masked_select(error_turb, mask_wind).max()
        all_turb_median = torch.masked_select(error_turb, mask_wind).median()
    else:
        all_turb_mean = torch.tensor(-1)
        all_turb_max = torch.tensor(-1)
        all_turb_median = torch.tensor(-1)

    # error properties close to the ground
    low_tot_mean = torch.masked_select(error_tot, mask_low).mean()
    low_tot_max = torch.masked_select(error_tot, mask_low).max()
    low_tot_median = torch.masked_select(error_tot, mask_low).median()

    low_hor_mean = torch.masked_select(error_hor, mask_low).mean()
    low_hor_max = torch.masked_select(error_hor, mask_low).max()
    low_hor_median = torch.masked_select(error_hor, mask_low).median()

    low_ver_mean = torch.masked_select(error_ver, mask_low).mean()
    low_ver_max = torch.masked_select(error_ver, mask_low).max()
    low_ver_median = torch.masked_select(error_ver, mask_low).median()

    if turbulence:
        low_turb_mean = torch.masked_select(error_turb, mask_low).mean()
        low_turb_max = torch.masked_select(error_turb, mask_low).max()
        low_turb_median = torch.masked_select(error_turb, mask_low).median()
    else:
        low_turb_mean = torch.tensor(-1)
        low_turb_max = torch.tensor(-1)
        low_turb_median = torch.tensor(-1)

    # error properties high above the ground
    high_tot_mean = torch.masked_select(error_tot, mask_high).mean()
    high_tot_max = torch.masked_select(error_tot, mask_high).max()
    high_tot_median = torch.masked_select(error_tot, mask_high).median()

    high_hor_mean = torch.masked_select(error_hor, mask_high).mean()
    high_hor_max = torch.masked_select(error_hor, mask_high).max()
    high_hor_median = torch.masked_select(error_hor, mask_high).median()

    high_ver_mean = torch.masked_select(error_ver, mask_high).mean()
    high_ver_max = torch.masked_select(error_ver, mask_high).max()
    high_ver_median = torch.masked_select(error_ver, mask_high).median()

    if turbulence:
        high_turb_mean = torch.masked_select(error_turb, mask_high).mean()
        high_turb_max = torch.masked_select(error_turb, mask_high).max()
        high_turb_median = torch.masked_select(error_turb, mask_high).median()
    else:
        high_turb_mean = torch.tensor(-1)
        high_turb_max = torch.tensor(-1)
        high_turb_median = torch.tensor(-1)

    # pack the values
    error_stats = {
        'all_tot_mean': all_tot_mean.item(),
        'all_tot_max': all_tot_max.item(),
        'all_tot_median': all_tot_median.item(),
        'all_hor_mean': all_hor_mean.item(),
        'all_hor_max': all_hor_max.item(),
        'all_hor_median': all_hor_median.item(),
        'all_ver_mean': all_ver_mean.item(),
        'all_ver_max': all_ver_max.item(),
        'all_ver_median': all_ver_median.item(),
        'all_turb_mean': all_turb_mean.item(),
        'all_turb_max': all_turb_max.item(),
        'all_turb_median': all_turb_median.item(),

        'low_tot_mean': low_tot_mean.item(),
        'low_tot_max': low_tot_max.item(),
        'low_tot_median': low_tot_median.item(),
        'low_hor_mean': low_hor_mean.item(),
        'low_hor_max': low_hor_max.item(),
        'low_hor_median': low_hor_median.item(),
        'low_ver_mean': low_ver_mean.item(),
        'low_ver_max': low_ver_max.item(),
        'low_ver_median': low_ver_median.item(),
        'low_turb_mean': low_turb_mean.item(),
        'low_turb_max': low_turb_max.item(),
        'low_turb_median': low_turb_median.item(),

        'high_tot_mean': high_tot_mean.item(),
        'high_tot_max': high_tot_max.item(),
        'high_tot_median': high_tot_median.item(),
        'high_hor_mean': high_hor_mean.item(),
        'high_hor_max': high_hor_max.item(),
        'high_hor_median': high_hor_median.item(),
        'high_ver_mean': high_ver_mean.item(),
        'high_ver_max': high_ver_max.item(),
        'high_ver_median': high_ver_median.item(),
        'high_turb_mean': high_turb_mean.item(),
        'high_turb_max': high_turb_max.item(),
        'high_turb_median': high_turb_median.item(),
        }

    return error_stats
