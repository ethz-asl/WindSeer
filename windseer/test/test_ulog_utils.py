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
test_file_data = os.path.join(testdata_folder, 'test_cosmo.nc')
test_file_terrain = os.path.join(testdata_folder, 'test_cosmo_terrain.nc')
test_file_ulog = os.path.join(testdata_folder, 'test_ulog.ulg')
test_file_hdf5 = os.path.join(testdata_folder, 'test_hdf5.hdf5')


class TestUlogUtils(unittest.TestCase):

    def test_handle_ulog(self):
        wind_data = utils.extract_wind_data(test_file_ulog, False)

        shape = wind_data['time'].shape
        for key in [
            'time', 'lat', 'lon', 'alt_amsl', 'x', 'y', 'alt', 'time_gps', 'we', 'wn',
            'wd'
            ]:
            self.assertTrue(key in wind_data.keys())
            if key != 'time_gps':
                self.assertEqual(wind_data[key].shape, shape)

        wind_data = utils.extract_wind_data(test_file_ulog, True)

        shape = wind_data['time'].shape
        for key in [
            'time', 'lat', 'lon', 'alt_amsl', 'x', 'y', 'alt', 'time_gps', 'we', 'wn',
            'wd'
            ]:
            self.assertTrue(key in wind_data.keys())
            if key != 'time_gps':
                self.assertEqual(wind_data[key].shape, shape)

    def test_handle_hdf5(self):
        wind_data = utils.extract_wind_data(test_file_hdf5, True)

        shape = wind_data['time'].shape
        for key in [
            'time', 'lat', 'lon', 'alt_amsl', 'x', 'y', 'alt', 'time_gps', 'we', 'wn',
            'wd'
            ]:
            self.assertTrue(key in wind_data.keys())
            self.assertEqual(wind_data[key].shape, shape)

    def test_filter_wind_data(self):
        wind_in = {
            'we': np.array([0., 1, 2, 3, 4, 5, 6]),
            'wn': np.array([1., 1, 1, 1, 1, 1, 1]),
            'wd': np.array([0., 0, 0, 3, 3, 3, 6])
            }

        wind_out = utils.filter_wind_data(wind_in, 3)

        self.assertTrue(np.array_equal(wind_in['we'][1:-1], wind_out['we_raw']))
        self.assertTrue(np.array_equal(wind_in['wn'][1:-1], wind_out['wn_raw']))
        self.assertTrue(np.array_equal(wind_in['wd'][1:-1], wind_out['wd_raw']))

        self.assertTrue(np.allclose(wind_in['we'][1:-1], wind_out['we']))
        self.assertTrue(np.allclose(wind_in['wn'][1:-1], wind_out['wn']))
        self.assertTrue(np.allclose(np.array([0, 1, 2, 3, 4]), wind_out['wd']))


if __name__ == '__main__':
    unittest.main()
