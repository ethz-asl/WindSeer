#!/usr/bin/env python
'''
Testcases for the neural network prediction
'''

import windseer.utils as utils
import windseer

import os
import torch
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_dataset = os.path.join(testdata_folder, 'test_dataset.hdf5')
model_dir = os.path.join(testdata_folder, 'model')


class TestLoadModel(unittest.TestCase):

    def test_load_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net, params = utils.load_model(model_dir, 'latest', test_dataset, device)

        self.assertTrue(isinstance(net, torch.nn.Module))
        self.assertTrue(isinstance(params, utils.WindseerParams))


if __name__ == '__main__':
    unittest.main()
