#!/usr/bin/env python
'''
Testcases for the measurements binning
'''

import windseer.utils as utils

import numpy as np
import torch
import unittest

class TestBinLogData(unittest.TestCase):
    def setUp(self):
        self.grid_dimensions = {
            'n_cells': 2,
            'x_min': 0.0,
            'y_min': 0.0,
            'z_min': 0.0,
            'x_max': 10.0,
            'y_max': 10.0,
            'z_max': 10.0,
            'n_cells': 2,
            }

        self.wind_data = {
            'x': np.array([2.5, 2.5, 6, 7.5]),
            'y': np.array([2.5, 7.5, 2.5, 2.5]),
            'alt': np.array([2.5, 7.5, 7.5, 7.5]),
            'we': np.array([1, 2, 3, 7]),
            'wn': np.array([7, 3, 2, -4]),
            'wd': np.array([-2, 2, 2, 2]),
            'time': np.array([1, 2, 3, 4]) * 1e6,
            'time_gps': np.array([11, 12, 13, 14]),
            }

    def run_test(self, method, gt_wind1, gt_wind2):
        # all data
        wind, variance, mask, prediction = utils.bin_log_data(
            self.wind_data,
            self.grid_dimensions,
            method=method,
            t_start=None,
            t_end=None,
            use_gps_time=False
            )

        gt_mask = torch.tensor([[[1., 0.],
                                 [0., 0.]],
                                 [[0., 1.],
                                 [1., 0.]]])

        self.assertTrue(torch.allclose(gt_wind1, wind, atol=1e-4))
        self.assertTrue(torch.equal(gt_mask, mask))

        # only middle data
        wind, variance, mask, prediction = utils.bin_log_data(
            self.wind_data,
            self.grid_dimensions,
            method='binning',
            t_start=1.5,
            t_end=3.5,
            use_gps_time=False
            )

        gt_mask = torch.tensor([[[0., 0.],
                                 [0., 0.]],
                                 [[0., 1.],
                                 [1., 0.]]])

        self.assertTrue(torch.equal(gt_wind2, wind))
        self.assertTrue(torch.equal(gt_mask, mask))

        # using gps time
        wind_gps, variance, mask_gps, prediction = utils.bin_log_data(
            self.wind_data,
            self.grid_dimensions,
            method='binning',
            t_start=11.5,
            t_end=13.5,
            use_gps_time=True
            )

        self.assertTrue(torch.equal(wind, wind_gps))
        self.assertTrue(torch.equal(mask, mask_gps))

    def test_binning(self):
        gt_wind1 = torch.zeros(3,2,2,2)
        gt_wind1[:,0,0,0] = torch.tensor([1., 7., 2.])
        gt_wind1[:,1,1,0] = torch.tensor([2., 3., -2.])
        gt_wind1[:,1,0,1] = torch.tensor([5., -1., -2.])

        gt_wind2 = torch.zeros(3,2,2,2)
        gt_wind2[:,1,1,0] = torch.tensor([2., 3., -2.])
        gt_wind2[:,1,0,1] = torch.tensor([3., 2., -2.])

        self.run_test('binning', gt_wind1, gt_wind2)

    def test_interpolation(self):
        gt_wind1 = torch.zeros(3,2,2,2)
        gt_wind1[:,0,0,0] = torch.tensor([1., 7., 2.])
        gt_wind1[:,1,1,0] = torch.tensor([2., 3., -2.])
        gt_wind1[:,1,0,1] = torch.tensor([7., -4., -2.])

        gt_wind2 = torch.zeros(3,2,2,2)
        gt_wind2[:,1,1,0] = torch.tensor([2., 3., -2.])
        gt_wind2[:,1,0,1] = torch.tensor([3., 2., -2.])

        self.run_test('interpolation', gt_wind1, gt_wind2)

    def test_gpr(self):
        gt_wind1 = torch.zeros(3,2,2,2)
        gt_wind1[:,0,0,0] = torch.tensor([1.45, 6., 1.43])
        gt_wind1[:,1,1,0] = torch.tensor([2.25, 2.80, -1.8119])
        gt_wind1[:,1,0,1] = torch.tensor([6.25, -2.8, -1.9034])

        gt_wind2 = torch.zeros(3,2,2,2)
        gt_wind2[:,1,1,0] = torch.tensor([2., 3., -2.])
        gt_wind2[:,1,0,1] = torch.tensor([3., 2., -2.])

        self.run_test('gpr', gt_wind1, gt_wind2)

if __name__ == '__main__':
    unittest.main()
