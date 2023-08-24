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
test_file = os.path.join(testdata_folder, 'test_geotiff.tif')


class TestGetMapgeoTerrain(unittest.TestCase):

    def test_get_terrain(self):
        # case 1: Bool, True inside the terrain
        x_target, y_target, z_target, z_out, full_block = utils.get_terrain(
            test_file, [569000, 570000], [219000, 220000],
            z=None,
            resolution=(64, 64, 64),
            build_block=True,
            plot=True,
            distance_field=False,
            horizontal_overflow=0,
            return_is_wind=False
            )

        self.assertTrue(569000 <= x_target.min())
        self.assertTrue(570000 >= x_target.max())
        self.assertTrue(219000 <= y_target.min())
        self.assertTrue(220000 >= y_target.max())
        self.assertTrue(full_block.max() <= 1)
        self.assertEqual(full_block.dtype, bool)

        # case 2: Bool, True outside the terrain
        x_target, y_target, z_target, z_out, full_block_is_wind = utils.get_terrain(
            test_file, [569000, 570000], [219000, 220000],
            z=None,
            resolution=(64, 64, 64),
            build_block=True,
            plot=False,
            distance_field=False,
            horizontal_overflow=0,
            return_is_wind=True
            )

        self.assertTrue(569000 <= x_target.min())
        self.assertTrue(570000 >= x_target.max())
        self.assertTrue(219000 <= y_target.min())
        self.assertTrue(220000 >= y_target.max())
        self.assertTrue(full_block_is_wind.max() <= 1)
        self.assertTrue(np.all(full_block_is_wind != full_block))

        # case 3: Distance field
        x_target, y_target, z_target, z_out, full_block_dist = utils.get_terrain(
            test_file, [569000, 570000], [219000, 220000],
            z=None,
            resolution=(64, 64, 64),
            build_block=True,
            plot=False,
            distance_field=True,
            horizontal_overflow=0,
            return_is_wind=True
            )

        self.assertTrue(569000 <= x_target.min())
        self.assertTrue(570000 >= x_target.max())
        self.assertTrue(219000 <= y_target.min())
        self.assertTrue(220000 >= y_target.max())
        self.assertTrue(full_block_dist.max() > 1)
        self.assertEqual(full_block_dist.dtype, float)

        # case 3: Distance field with overflow
        x_target, y_target, z_target, z_out, full_block_dist = utils.get_terrain(
            test_file, [569000, 570000], [219000, 220000],
            z=None,
            resolution=(64, 64, 64),
            build_block=True,
            plot=False,
            distance_field=True,
            horizontal_overflow=10,
            return_is_wind=True
            )

        self.assertTrue(569000 > x_target.min())
        self.assertTrue(570000 < x_target.max())
        self.assertTrue(219000 > y_target.min())
        self.assertTrue(220000 < y_target.max())
        self.assertTrue(full_block_dist.max() > 1)
        self.assertEqual(full_block_dist.dtype, float)
        self.assertEqual(full_block_dist.shape, (64, 64, 64))


if __name__ == '__main__':
    unittest.main()
