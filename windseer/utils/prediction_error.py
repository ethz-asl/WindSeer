import numpy as np
from scipy import ndimage
import torch

threshold_low = 7.0  # this corresponds to roughly 80 m for the unscaled terrain
eps = 1e-1


def compute_prediction_error(label, prediction, terrain, device, turbulence):
    '''
    Compute the prediction errors for one single sample.
    Assumes that ux, uy, uz, and optionally the TKE are predicted.

    Parameters
    ----------
    label : torch.Tensor
        Label tensor
    prediction : torch.Tensor
        Prediction tensor
    terrain : torch.Tensor
        Terrain tensor
    device : torch.Device
        Device where the computations are executed
    turbulence : bool
        Flag indicating if the turbulence was predicted

    Returns
    -------
    error_stats : dict
        Dictionary with the individual error statistics
    '''
    abs_error = (label - prediction).abs()

    # convert the boolean terrain into a distance field if required
    if ((terrain.max().item() == 1.0) and (
        torch.mul(
            torch.gt(terrain, torch.zeros_like(terrain)),
            torch.gt(torch.ones_like(terrain), terrain)
            ).sum() == 0
        )):
        terrain = torch.from_numpy(
            ndimage.distance_transform_edt(terrain.sign().cpu().numpy()
                                           ).astype(np.float32)
            ).to(device)

    mask_low = torch.mul((terrain <= threshold_low), (terrain > 0.0))
    mask_high = (terrain > threshold_low)
    mask_wind = (terrain > 0.0)

    properties_dict = {
        'error_tot': abs_error[:3].norm(dim=0),
        'error_hor': abs_error[:2].norm(dim=0),
        'error_ver': abs_error[2],
        'vel_tot': label[:3].norm(dim=0).clamp(min=eps),
        'vel_hor': label[:2].norm(dim=0).clamp(min=eps),
        'vel_ver': label[2].clamp(min=eps),
        }
    channels = ['tot', 'hor', 'ver']

    if turbulence:
        properties_dict['error_turb'] = abs_error[3]
        properties_dict['vel_turb'] = label[3].clamp(min=eps)
        channels += ['turb']

    error_stats = {}
    for domain, mask in zip(['all_', 'low_', 'high_'],
                            [mask_wind, mask_low, mask_high]):
        sum_mask = mask.sum()

        for ch in channels:
            if sum_mask > 0.0:
                error_stats[domain + ch + '_mean'] = torch.masked_select(
                    properties_dict['error_' + ch], mask
                    ).mean().cpu().item()
                error_stats[domain + ch + '_max'] = torch.masked_select(
                    properties_dict['error_' + ch], mask
                    ).max().cpu().item()
                error_stats[domain + ch + '_median'] = torch.masked_select(
                    properties_dict['error_' + ch], mask
                    ).max().cpu().item()

                rel_error = torch.masked_select(
                    properties_dict['error_' + ch], mask
                    ) / torch.masked_select(properties_dict['vel_' + ch], mask)
                error_stats[domain + ch + '_mean_rel'] = rel_error.mean().cpu().item()
                error_stats[domain + ch + '_max_rel'] = rel_error.max().cpu().item()
                error_stats[domain + ch +
                            '_median_rel'] = rel_error.median().cpu().item()
            else:
                for rel in ['', '_rel']:
                    error_stats[domain + ch + '_mean' + rel] = float('NaN')
                    error_stats[domain + ch + '_max' + rel] = float('NaN')
                    error_stats[domain + ch + '_median' + rel] = float('NaN')

        if not 'turb' in channels:
            for rel in ['', '_rel']:
                error_stats[domain + 'turb_mean' + rel] = float('NaN')
                error_stats[domain + 'turb_max' + rel] = float('NaN')
                error_stats[domain + 'turb_median' + rel] = float('NaN')

    return error_stats
