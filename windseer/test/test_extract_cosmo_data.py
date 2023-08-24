#!/usr/bin/env python
'''
Testcases for the neural network prediction
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


class TestLoadModel(unittest.TestCase):
    x_local = 683290.31
    y_local = 247923.97
    lat_requested = 47.373878
    lon_requested = 8.545094

    def test_get_cosmo_cell(self):
        cell = utils.get_cosmo_cell(
            test_file_terrain, self.lat_requested, self.lon_requested, 1000, 1100, 700
            )

        self.assertTrue(self.x_local > cell['x_min'])
        self.assertTrue(self.x_local < cell['x_max'])
        self.assertTrue(self.y_local > cell['y_min'])
        self.assertTrue(self.y_local < cell['y_max'])
        gt_cell = {
            'x_min': 683042.5985994573,
            'x_max': 684142.5985994573,
            'y_min': 247239.62005789945,
            'y_max': 248339.62005789945,
            'z_min': 1000,
            'z_max': 1700
            }
        self.assertEqual(cell, gt_cell)

    def test_extract_cosmo_data(self):
        cosmo_data = utils.extract_cosmo_data(
            test_file_data, self.lat_requested, self.lon_requested, 0, test_file_terrain
            )

        self.assertTrue(cosmo_data['valid'])
        self.assertEqual(
            list(cosmo_data.keys()), [
                'valid', 'lat', 'lon', 'x', 'y', 'wind_x', 'wind_y', 'wind_z', 'hsurf',
                'z'
                ]
            )
        self.assertTrue(
            np.allclose(
                cosmo_data['lat'],
                np.array([[47.37076, 47.37095], [47.38076, 47.380947]],
                         dtype=np.float32)
                )
            )
        self.assertTrue(
            np.allclose(
                cosmo_data['lon'],
                np.array([[8.538125, 8.552902], [8.537849, 8.552629]], dtype=np.float32)
                )
            )
        self.assertTrue(
            np.allclose(
                cosmo_data['x'],
                np.array([[683042.59859946, 684158.31869453],
                          [683006.20668277, 684121.94038624]],
                         dtype=np.float32)
                )
            )
        self.assertTrue(
            np.allclose(
                cosmo_data['y'],
                np.array([[247239.6200579, 247276.1466035],
                          [248350.85077544, 248387.3763286]],
                         dtype=np.float32)
                )
            )
        self.assertTrue(
            np.allclose(
                cosmo_data['hsurf'],
                np.array([[412.34570312, 469.09570312], [447.59570312, 520.59570312]],
                         dtype=np.float32)
                )
            )
        self.assertEqual(cosmo_data['wind_x'].shape, (31, 2, 2))
        self.assertEqual(cosmo_data['wind_y'].shape, (31, 2, 2))
        self.assertEqual(cosmo_data['wind_z'].shape, (31, 2, 2))
        self.assertEqual(cosmo_data['z'].shape, (31, 2, 2))

    def test_cosmo_corner_wind(self):
        cosmo_data = utils.extract_cosmo_data(
            test_file_data, self.lat_requested, self.lon_requested, 0, test_file_terrain
            )

        wind = utils.cosmo_corner_wind(cosmo_data, np.array([500, 1000]))

        gt_wind = np.array([[[[0.15214282, -0.20453681], [-0.44260006, 0.]],
                             [[-9.03256081, -9.53809709], [-9.74994277, -9.84905954]]],
                            [[[-2.5741844, -2.13986921], [-2.39365438, 0.]],
                             [[5.41880898, 5.87208184], [6.20041312, 6.91702701]]],
                            [[[-0.04319926, -0.09577166], [-0.05619838, 0.]],
                             [[-0.09297289, -0.03331454], [-0.12702667, 0.13384758]]]])
        self.assertTrue(np.allclose(wind, gt_wind))


if __name__ == '__main__':
    unittest.main()
