import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch

def interpolate_flight_path(wind_prediction, grid_dimensions, flight_path):
    if len(wind_prediction.squeeze().shape) != 4:
        raise ValueError('The prediction is assumed to be a 4D tensor (channels, z, y, x)')

    wind_prediction_permuted = wind_prediction.detach().cpu().squeeze().permute(0,3,2,1).numpy()

    if wind_prediction_permuted.shape[0] != 3:
        raise ValueError('The prediction is assumed to have 3 channels (ux, uy, uz)')

    x = np.arange(grid_dimensions['n_cells'])
    y = np.arange(grid_dimensions['n_cells'])
    z = np.arange(grid_dimensions['n_cells'])

    interpolators = [RegularGridInterpolator((x, y, z), wind_prediction_permuted[0]),
                     RegularGridInterpolator((x, y, z), wind_prediction_permuted[1]),
                     RegularGridInterpolator((x, y, z), wind_prediction_permuted[2])]

    resolutions = [(grid_dimensions['x_max'] - grid_dimensions['x_min']) / grid_dimensions['n_cells'],
                   (grid_dimensions['y_max'] - grid_dimensions['y_min']) / grid_dimensions['n_cells'],
                   (grid_dimensions['z_max'] - grid_dimensions['z_min']) / grid_dimensions['n_cells']]

    points = np.array([flight_path['x'], flight_path['y'], flight_path['z']]).transpose()
    points -= np.array([grid_dimensions['x_min'], grid_dimensions['y_min'], grid_dimensions['z_min']])
    points -= 0.5 * np.array(resolutions)
    points /= resolutions

    interpolated_estimates = {'we_pred': interpolators[0](points),
                              'wn_pred': interpolators[1](points),
                              'wu_pred': interpolators[2](points)}

    return interpolated_estimates
