#!/usr/bin/env python
'''
Testcases for the sparse evaluation
'''

import windseer.evaluation as eval
import windseer.utils as utils
import windseer

import numpy as np
import os
import torch
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_file_hdf5 = os.path.join(testdata_folder, 'test_hdf5.hdf5')
model_dir = os.path.join(testdata_folder, 'model')
terrain_filename = os.path.join(windseer_path, 'test', 'testdata', 'test_geotiff.tif')
config_filename = os.path.join(
    os.path.split(windseer_path)[0], 'configs', 'example_sparse.yaml'
    )


class TestSparseEvaluation(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net, params = utils.load_model(model_dir, 'latest', None, self.device, True)
        self.net = net
        self.params = params

        # baseline params
        self.eval_config_baseline = utils.BasicParameters(config_filename)
        self.eval_config_baseline.params['evaluation']['benchmark'] = True
        self.eval_config_baseline.params['evaluation']['mode'] = 0
        self.eval_config_baseline.params['evaluation']['dt_input'] = 500
        self.eval_config_baseline.params['evaluation']['dt_pred'] = 500
        self.eval_config_baseline.params['evaluation']['compute_baseline'] = True
        self.eval_config_baseline.params['evaluation']['show_plots'] = True
        self.eval_config_baseline.params['evaluation']['early_averaging'] = True
        self.eval_config_baseline.params['evaluation']['validation_file'] = [
            test_file_hdf5
            ]
        self.eval_config_baseline.params['measurements']['type'] = 'log'
        self.eval_config_baseline.params['measurements']['log']['filename'
                                                                ] = test_file_hdf5
        self.eval_config_baseline.params['measurements']['log']['geotiff_file'
                                                                ] = terrain_filename
        self.eval_config_baseline = eval.update_eval_config(
            self.eval_config_baseline, None
            )

        # model params
        self.eval_config_model = utils.BasicParameters(config_filename)
        self.eval_config_model.params['evaluation']['benchmark'] = True
        self.eval_config_model.params['evaluation']['mode'] = 0
        self.eval_config_model.params['evaluation']['dt_input'] = 500
        self.eval_config_model.params['evaluation']['dt_pred'] = 500
        self.eval_config_model.params['evaluation']['compute_baseline'] = False
        self.eval_config_model.params['evaluation']['show_plots'] = False
        self.eval_config_model.params['evaluation']['early_averaging'] = False
        self.eval_config_model.params['evaluation']['validation_file'] = [
            test_file_hdf5
            ]
        self.eval_config_model.params['measurements']['type'] = 'log'
        self.eval_config_model.params['measurements']['log']['filename'
                                                             ] = test_file_hdf5
        self.eval_config_model.params['measurements']['log']['geotiff_file'
                                                             ] = terrain_filename
        self.eval_config_model = eval.update_eval_config(
            self.eval_config_model, self.params
            )

        self.data = eval.load_wind_data(self.eval_config_model, True)

        # only keep the two loiters to speed up the script
        self.data['loiters'] = self.data['loiters'][:2]
        self.data['loiters_validation'][0] = self.data['loiters_validation'][0][-2:]

    def test_sparse_evaluation(self):
        for i in range(5):
            self.eval_config_baseline.params['evaluation']['mode'] = i
            self.eval_config_model.params['evaluation']['mode'] = i
            eval.evaluate_flight_log(
                self.data['wind_data'], self.data['scale'], self.data['terrain'],
                self.data['grid_dimensions'], None, self.eval_config_baseline,
                self.device, self.data['wind_data_validation'], False, False
                )
            eval.evaluate_flight_log(
                self.data['wind_data'], self.data['scale'], self.data['terrain'],
                self.data['grid_dimensions'], self.net, self.eval_config_model,
                self.device, self.data['wind_data_validation'], False, False
                )

    def test_loiter_evaluation(self):
        self.eval_config_model.params['evaluation']['benchmark'] = False
        self.eval_config_baseline.params['evaluation']['benchmark'] = False
        eval.loiter_evaluation(
            self.data, None, self.eval_config_baseline, self.device, False
            )
        eval.loiter_evaluation(
            self.data, self.net, self.eval_config_model, self.device, False
            )

        self.eval_config_model.params['evaluation']['benchmark'] = True
        self.eval_config_baseline.params['evaluation']['benchmark'] = True
        eval.loiter_evaluation(
            self.data, None, self.eval_config_baseline, self.device, False
            )
        eval.loiter_evaluation(
            self.data, self.net, self.eval_config_model, self.device, False
            )


if __name__ == '__main__':
    unittest.main()
