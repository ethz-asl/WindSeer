#!/usr/bin/env python
'''
Testcases for the yaml tools
'''

import windseer.utils as utils
import windseer

import os
import unittest

windseer_path = os.path.dirname(windseer.__file__)
config_filename = os.path.join(
    os.path.split(windseer_path)[0], 'configs', 'example.yaml'
    )
config_wind_filename = os.path.join(
    os.path.split(windseer_path)[0], 'configs', 'example_wind.yaml'
    )


class TestYamltools(unittest.TestCase):

    def test_WindseerParams(self):
        config = utils.WindseerParams(config_filename)

        config.save(windseer_path)
        save_name = os.path.join(windseer_path, 'params.yaml')

        config_saved = utils.WindseerParams(save_name)
        os.remove(save_name)

        self.assertEqual(config_saved.model, config.model)
        self.assertEqual(config_saved.model_kwargs(), config.model_kwargs())
        self.assertEqual(config_saved.data, config.data)
        self.assertEqual(config_saved.Dataset_kwargs(), config.Dataset_kwargs())
        self.assertEqual(config_saved.run, config.run)
        self.assertEqual(config_saved.loss, config.loss)

    def test_BasicParameters(self):
        config = utils.BasicParameters(config_filename)
        params = config.params

        self.assertTrue(True)

    def test_COSMOParameters(self):
        config = utils.COSMOParameters(config_wind_filename)

        config.save(windseer_path)
        save_name = os.path.join(windseer_path, 'cosmo.yaml')

        config_saved = utils.COSMOParameters(save_name)
        os.remove(save_name)

        self.assertEqual(config_saved.params, config.params)

    def test_UlogParameters(self):
        config = utils.UlogParameters(config_wind_filename)

        config.save(windseer_path)
        save_name = os.path.join(windseer_path, 'ulog.yaml')

        config_saved = utils.UlogParameters(save_name)
        os.remove(save_name)

        self.assertEqual(config_saved.params, config.params)


if __name__ == '__main__':
    unittest.main()
