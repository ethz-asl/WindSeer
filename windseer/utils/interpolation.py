import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch


def interpolate_flight_path(wind_prediction, grid_dimensions, flight_path):
    '''
    Interpolate the wind and turbulence (if available) prediction of the regular grid along the flight path.

    Parameters
    ----------
    wind_prediction : torch.Tensor
        Tensor with the wind prediction
    grid_dimensions : dict
        Dictionary with the grid dimensions (extent, resolutions, ...)
    flight_path : dict
        Dictionary with the flight path positions

    Returns
    -------
    interpolated_estimates : dict
        Dictionary with the interpolated wind and turbulence predictions
    '''
    if len(wind_prediction.squeeze().shape) != 4:
        raise ValueError(
            'The prediction is assumed to be a 4D tensor (channels, z, y, x)'
            )

    wind_prediction_permuted = wind_prediction.detach().cpu().squeeze().permute(
        0, 3, 2, 1
        ).numpy()

    if not (
        wind_prediction_permuted.shape[0] == 3 or wind_prediction_permuted.shape[0] == 4
        ):
        raise ValueError(
            'The prediction is assumed to have 3/4 channels (ux, uy, uz)/(ux, uy, uz, turb)'
            )

    x = np.arange(grid_dimensions['n_cells'])
    y = np.arange(grid_dimensions['n_cells'])
    z = np.arange(grid_dimensions['n_cells'])

    interpolators = [
        RegularGridInterpolator((x, y, z),
                                wind_prediction_permuted[0],
                                bounds_error=False,
                                fill_value=None),
        RegularGridInterpolator((x, y, z),
                                wind_prediction_permuted[1],
                                bounds_error=False,
                                fill_value=None),
        RegularGridInterpolator((x, y, z),
                                wind_prediction_permuted[2],
                                bounds_error=False,
                                fill_value=None)
        ]
    if wind_prediction_permuted.shape[0] == 4:
        interpolators.append(
            RegularGridInterpolator((x, y, z),
                                    wind_prediction_permuted[3],
                                    bounds_error=False,
                                    fill_value=None)
            )

    resolutions = [(grid_dimensions['x_max'] - grid_dimensions['x_min']) /
                   grid_dimensions['n_cells'],
                   (grid_dimensions['y_max'] - grid_dimensions['y_min']) /
                   grid_dimensions['n_cells'],
                   (grid_dimensions['z_max'] - grid_dimensions['z_min']) /
                   grid_dimensions['n_cells']]

    points = np.array([flight_path['x'], flight_path['y'],
                       flight_path['z']]).transpose()
    points -= np.array([
        grid_dimensions['x_min'], grid_dimensions['y_min'], grid_dimensions['z_min']
        ])
    points -= 0.5 * np.array(resolutions)
    points /= resolutions

    interpolated_estimates = {
        'we_pred': interpolators[0](points),
        'wn_pred': interpolators[1](points),
        'wu_pred': interpolators[2](points)
        }

    if wind_prediction_permuted.shape[0] == 4:
        interpolated_estimates['turb_pred'] = interpolators[3](points)

    return interpolated_estimates
