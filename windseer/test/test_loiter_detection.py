#!/usr/bin/env python
'''
Testcases for the ulog utilities
'''

import windseer.utils as utils
import windseer

import numpy as np
import os
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_file_hdf5 = os.path.join(testdata_folder, 'test_hdf5.hdf5')


class TestLoiterDetection(unittest.TestCase):

    def test_handle_hdf5(self):
        wind_data = utils.extract_wind_data(test_file_hdf5, True)

        config = {
            'target_radius': 100,
            'radius_tolerance': 12,
            'max_climb_rate': 1.5,
            'error_tolerance': 12,
            'max_altitude_change': 18,
            'loiter_threshold': 1.0,
            'min_window_time': 20,
            'step': 1,
            'plot_results': False,
            }

        loiters = utils.detect_loiters(wind_data, config)

        self.assertTrue(len(loiters) == 5)

        for ltr in loiters:
            self.assertTrue(
                abs(ltr['R'] - config['target_radius']) < config['radius_tolerance']
                )
            self.assertTrue(
                (ltr['idx_stop'] - ltr['idx_start']) > config['min_window_time'] * 50
                )


if __name__ == '__main__':
    unittest.main()
