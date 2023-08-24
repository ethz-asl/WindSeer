#!/usr/bin/env python
'''
Testcases for the flight path interpolation
'''

import windseer.utils as utils

import numpy as np
import torch
import unittest


class TestInterpolation(unittest.TestCase):

    def test_interpolate_flight_path(self):
        wind_prediction = torch.randn(3, 2, 2, 2)
        grid_dimensions = {
            'n_cells': 2,
            'x_min': 0.0,
            'y_min': 0.0,
            'z_min': 0.0,
            'x_max': 1.0,
            'y_max': 1.0,
            'z_max': 1.0,
            }
        flight_path = {
            'x': np.array([0.5, 0.25, 0.25]),
            'y': np.array([0.25, 0.5, 0.25]),
            'z': np.array([0.25, 0.25, 0.5])
            }
        interpolated = utils.interpolate_flight_path(
            wind_prediction, grid_dimensions, flight_path
            )

        self.assertTrue(
            np.allclose(
                interpolated['we_pred'][0],
                0.5 * (wind_prediction[0, 0, 0, 0] + wind_prediction[0, 0, 0, 1])
                )
            )
        self.assertTrue(
            np.allclose(
                interpolated['wn_pred'][0],
                0.5 * (wind_prediction[1, 0, 0, 0] + wind_prediction[1, 0, 0, 1])
                )
            )
        self.assertTrue(
            np.allclose(
                interpolated['wu_pred'][0],
                0.5 * (wind_prediction[2, 0, 0, 0] + wind_prediction[2, 0, 0, 1])
                )
            )

        self.assertTrue(
            np.allclose(
                interpolated['we_pred'][1],
                0.5 * (wind_prediction[0, 0, 0, 0] + wind_prediction[0, 0, 1, 0])
                )
            )
        self.assertTrue(
            np.allclose(
                interpolated['wn_pred'][1],
                0.5 * (wind_prediction[1, 0, 0, 0] + wind_prediction[1, 0, 1, 0])
                )
            )
        self.assertTrue(
            np.allclose(
                interpolated['wu_pred'][1],
                0.5 * (wind_prediction[2, 0, 0, 0] + wind_prediction[2, 0, 1, 0])
                )
            )

        self.assertTrue(
            np.allclose(
                interpolated['we_pred'][2],
                0.5 * (wind_prediction[0, 0, 0, 0] + wind_prediction[0, 1, 0, 0])
                )
            )
        self.assertTrue(
            np.allclose(
                interpolated['wn_pred'][2],
                0.5 * (wind_prediction[1, 0, 0, 0] + wind_prediction[1, 1, 0, 0])
                )
            )
        self.assertTrue(
            np.allclose(
                interpolated['wu_pred'][2],
                0.5 * (wind_prediction[2, 0, 0, 0] + wind_prediction[2, 1, 0, 0])
                )
            )


if __name__ == '__main__':
    unittest.main()
