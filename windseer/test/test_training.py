#!/usr/bin/env python
'''
Testcases for the neural network training
'''

import windseer.utils as utils
import windseer.nn as nn_custom
import windseer

import os
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_filename = os.path.join(testdata_folder, 'test_dataset.hdf5')

config_filename = os.path.join(
    os.path.split(windseer_path)[0], 'configs', 'example.yaml'
    )


class TestTraining(unittest.TestCase):

    def test_training(self):
        configs = utils.WindseerParams(config_filename)
        configs.data['trainset_name'] = test_filename
        configs.data['validationset_name'] = test_filename
        configs.run['batchsize'] = 1
        configs.run['num_workers'] = 0
        configs.run['plot_every_n_batches'] = 1
        configs.run['save_model_every_n_epoch'] = 1
        configs.run['compute_validation_loss_every_n_epochs'] = 1
        configs.run['n_epochs'] = 3

        # start the actual training
        nn_custom.train_model(configs, '/tmp/model_out', True, False)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
