#!/usr/bin/env python
'''
Testcases for the neural network training
'''

import windseer.utils as utils
import windseer

import os
import torch
import unittest


class TestDictutils(unittest.TestCase):

    def test_dict_update(self):
        dict_in = {
            'a': 1,
            'b': {
                'changed': 2
                },
            'changed': 3,
            'c': {
                'unchanged': 3
                },
            'none': None
            }
        dict_update = {
            'b': {
                'changed': 4
                },
            'new_field': 7,
            'newgroup': {
                'val': 0
                },
            'none': {
                'val': 0
                }
            }
        dict_out_gt = {
            'a': 1,
            'b': {
                'changed': 4
                },
            'changed': 3,
            'c': {
                'unchanged': 3
                },
            'new_field': 7,
            'newgroup': {
                'val': 0
                },
            'none': {
                'val': 0
                }
            }

        out = utils.dict_update(dict_in, dict_update)
        self.assertTrue(dict_out_gt == out)

    def test_data_to_device(self):
        dict_in = {'1': torch.zeros(1, 1), '2': {'1': torch.zeros(1, 1)}}

        # a bit of a nonsense test but it at least makes sure the function returns the same dictionary
        device = torch.device('cpu')
        out = utils.data_to_device(dict_in, device)
        self.assertFalse(out['1'].is_cuda)
        self.assertFalse(out['2']['1'].is_cuda)

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            out = utils.data_to_device(dict_in, device)
            self.assertTrue(out['1'].is_cuda)
            self.assertTrue(out['2']['1'].is_cuda)

    def test_tensors_to_dtype(self):
        dict_in = {
            '1': torch.zeros(1, 1, dtype=torch.float),
            '2': {
                '1': torch.zeros(1, 1, dtype=torch.float)
                }
            }

        for dtype in [torch.half, torch.double, torch.int, torch.short, torch.bool]:
            out = utils.tensors_to_dtype(dict_in, dtype)
            self.assertEqual(out['1'].dtype, dtype)
            self.assertEqual(out['2']['1'].dtype, dtype)

    def test_data_unsqueeze(self):
        dict_in = {'1': torch.zeros(2, 2), '2': {'1': torch.zeros(2, 2)}}
        out = utils.data_unsqueeze(dict_in, 0)
        self.assertEqual(list(out['1'].size()), [1, 2, 2])
        self.assertEqual(list(out['2']['1'].size()), [1, 2, 2])

        dict_in = {'1': torch.zeros(2, 2), '2': {'1': torch.zeros(2, 2)}}
        out = utils.data_unsqueeze(dict_in, 1)
        self.assertEqual(list(out['1'].size()), [2, 1, 2])
        self.assertEqual(list(out['2']['1'].size()), [2, 1, 2])

        dict_in = {'1': torch.zeros(2, 2), '2': {'1': torch.zeros(2, 2)}}
        out = utils.data_unsqueeze(dict_in, -1)
        self.assertEqual(list(out['1'].size()), [2, 2, 1])
        self.assertEqual(list(out['2']['1'].size()), [2, 2, 1])


if __name__ == '__main__':
    unittest.main()
