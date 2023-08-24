#!/usr/bin/env python
'''
Testcases for the neural network prediction
'''

import windseer.data as data
import windseer.utils as utils
import windseer.nn as nn
import windseer

import h5py
import numpy as np
import os
import torch
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_dataset = os.path.join(testdata_folder, 'test_dataset.hdf5')
model_dir = os.path.join(testdata_folder, 'model')


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net, params = utils.load_model(model_dir, 'latest', test_dataset, self.device)
        self.net = net
        self.params = params

        self.testset = data.HDF5Dataset(
            test_dataset,
            augmentation=False,
            return_grid_size=True,
            **params.Dataset_kwargs()
            )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=1, shuffle=False, num_workers=0
            )

        self.loss_fn = nn.CombinedLoss(**self.params.loss)

    def test_dataset_prediction_error(self):
        prediction_errors, losses, metrics, worst_index, maxloss = nn.compute_prediction_error(
            self.net,
            self.device,
            self.params,
            self.loss_fn,
            self.testset,
            print_output=False
            )
        self.assertTrue(isinstance(prediction_errors, dict))
        self.assertEqual(len(prediction_errors.keys()), 72)
        for key in prediction_errors.keys():
            self.assertEqual(len(prediction_errors[key]), 1)

        self.assertTrue(isinstance(losses, dict))
        self.assertEqual(len(losses.keys()), 5)
        for key in losses.keys():
            self.assertEqual(len(losses[key]), 1)

        self.assertTrue(isinstance(metrics, dict))
        self.assertEqual(len(metrics.keys()), 7)
        for key in metrics.keys():
            self.assertEqual(len(metrics[key]), 1)

        self.assertEqual(worst_index, 0)
        self.assertTrue(isinstance(maxloss, torch.Tensor))

    def test_sample_prediction_error(self):
        num_predictions = 3
        prediction_errors, losses, metrics, worst_index, maxloss = nn.compute_prediction_error(
            self.net,
            self.device,
            self.params,
            self.loss_fn,
            self.testset,
            single_sample=True,
            num_predictions=num_predictions,
            print_output=False
            )

        self.assertTrue(isinstance(prediction_errors, dict))
        self.assertEqual(len(prediction_errors.keys()), 72)
        for key in prediction_errors.keys():
            self.assertEqual(len(prediction_errors[key]), num_predictions)

        self.assertTrue(isinstance(losses, dict))
        self.assertEqual(len(losses.keys()), 5)
        for key in losses.keys():
            self.assertEqual(len(losses[key]), num_predictions)

        self.assertTrue(isinstance(metrics, dict))
        self.assertEqual(len(metrics.keys()), 7)
        for key in metrics.keys():
            self.assertEqual(len(metrics[key]), num_predictions)

    def test_visualize_prediction(self):
        savename = os.path.join(testdata_folder, 'saved_prediction.npy')

        try:
            # run mayavi in the headless mode
            from mayavi import mlab
            mlab.options.offscreen = True
        except:
            pass

        nn.predict_and_visualize(
            self.testset,
            0,
            self.device,
            self.net,
            self.params,
            'all',
            plot_divergence=False,
            loss_fn=torch.nn.MSELoss(),
            savename=savename,
            plottools=True,
            mayavi=True,
            blocking=False
            )

        pred = np.load(savename)

        os.remove(savename)

        self.assertTrue(isinstance(pred, np.ndarray))
        self.assertEqual(pred.shape, (4, 64, 64, 64))

    def test_save_prediction_to_database(self):
        savename = os.path.join(testdata_folder, 'sampled_dataset.hdf5')

        nn.save_prediction_to_database([{
            'net': self.net,
            'params': self.params,
            'name': 'model1'
            }, {
                'net': self.net,
                'params': self.params,
                'name': 'model2'
                }], self.device, self.params, savename, self.testset)

        file_exists = os.path.exists(savename)

        file = h5py.File(savename, 'r')
        group_0 = file[list(file.keys())[0]]

        os.remove(savename)

        self.assertTrue(file_exists)
        self.assertEqual(len(file.keys()), len(self.testset))
        self.assertEqual(len(group_0.keys()), 4)
        self.assertEqual(len(group_0['predictions'].keys()), 4)
        self.assertTrue(isinstance(group_0['terrain'][...], np.ndarray))
        self.assertTrue(
            isinstance(group_0['predictions']['zerowind']['wind'][...], np.ndarray)
            )
        self.assertTrue(
            isinstance(group_0['predictions']['interpolated']['wind'][...], np.ndarray)
            )
        self.assertTrue(
            isinstance(group_0['predictions']['model1']['wind'][...], np.ndarray)
            )


if __name__ == '__main__':
    unittest.main()
