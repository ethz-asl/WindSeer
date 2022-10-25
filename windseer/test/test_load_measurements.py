#!/usr/bin/env python
'''
Testcases for loading measurements
'''

import windseer.data as data
import windseer.utils as utils
import windseer

import os
import torch
import unittest

windseer_path = os.path.dirname(windseer.__file__)
model_config_filename = os.path.join(
    windseer_path, 'test', 'testdata', 'model', 'params.yaml'
    )
config_filename = os.path.join(
    os.path.split(windseer_path)[0], 'configs', 'example_sparse.yaml'
    )
dataset_filename = os.path.join(windseer_path, 'test', 'testdata', 'test_dataset.hdf5')
ulog_filename = os.path.join(windseer_path, 'test', 'testdata', 'test_ulog.ulg')
hdf5_filename = os.path.join(windseer_path, 'test', 'testdata', 'test_hdf5.hdf5')
cosmo_filename = os.path.join(
    windseer_path, 'test', 'testdata', 'test_cosmo_terrain.nc'
    )
terrain_filename = os.path.join(windseer_path, 'test', 'testdata', 'test_geotiff.tif')


class TestLoadMeasurements(unittest.TestCase):

    def test_load_measurements(self):
        config = utils.BasicParameters(config_filename)
        model_config = utils.WindseerParams(model_config_filename)

        # load data from the hdf5 dataset
        config.params['measurements']['type'] = 'cfd'
        config.params['measurements']['cfd']['filename'] = dataset_filename
        measurement, terrain, ground_truth, mask, scale, wind_data, grid_dimensions = data.load_measurements(
            config.params['measurements'], model_config.data
            )

        self.assertEqual(type(measurement), torch.Tensor)
        self.assertEqual(type(terrain), torch.Tensor)
        self.assertEqual(type(ground_truth), torch.Tensor)
        self.assertEqual(type(mask), torch.Tensor)
        self.assertEqual(scale, None)
        self.assertEqual(wind_data, None)
        self.assertEqual(grid_dimensions, None)

        config.params['measurements']['type'] = 'log'
        config.params['measurements']['log']['filename'] = ulog_filename
        config.params['measurements']['log']['geotiff_file'] = terrain_filename
        measurement, terrain, ground_truth, mask, scale, wind_data, grid_dimensions = data.load_measurements(
            config.params['measurements'], model_config.data
            )

        self.assertEqual(type(measurement), torch.Tensor)
        self.assertEqual(type(terrain), torch.Tensor)
        self.assertEqual(ground_truth, None)
        self.assertEqual(type(mask), torch.Tensor)
        self.assertEqual(scale, None)
        self.assertEqual(type(wind_data), dict)
        self.assertEqual(type(grid_dimensions), dict)

        config.params['measurements']['type'] = 'log'
        config.params['measurements']['log']['filename'] = hdf5_filename
        config.params['measurements']['log']['cosmo_file'] = cosmo_filename
        config.params['measurements']['log']['geotiff_file'] = terrain_filename
        config.params['measurements']['log']['use_cosmo_grid'] = True
        measurement, terrain, ground_truth, mask, scale, wind_data, grid_dimensions = data.load_measurements(
            config.params['measurements'], model_config.data
            )

        self.assertEqual(type(measurement), torch.Tensor)
        self.assertEqual(type(terrain), torch.Tensor)
        self.assertEqual(ground_truth, None)
        self.assertEqual(type(mask), torch.Tensor)
        self.assertEqual(scale, None)
        self.assertEqual(type(wind_data), dict)
        self.assertEqual(type(grid_dimensions), dict)


if __name__ == '__main__':
    unittest.main()
