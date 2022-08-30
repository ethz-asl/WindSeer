#!/usr/bin/env python
'''
Testcases for the KwargsParser
'''

from windseer.data.process_datasets import *
from windseer.data import HDF5Dataset
import windseer

import copy
import os
import unittest

windseer_path = os.path.dirname(windseer.__file__)
testdata_folder = os.path.join(windseer_path, 'test', 'testdata')
test_filename = os.path.join(testdata_folder, 'test_dataset.hdf5')
tar_filename = os.path.join(testdata_folder, 'test_csv.tar')
out_filename = os.path.join(testdata_folder, 'out_testdataset.hdf5')

default_config = {
    'filename': out_filename,
    'input_channels': ['terrain', 'ux', 'uy', 'uz'],
    'label_channels': ['ux', 'uy', 'uz', 'turb'],
    'nx': 32,
    'ny': 32,
    'nz': 32,
    'input_mode': 5,
    }


class TestDatasetConversion(unittest.TestCase):

    def test_compression_lzf(self):
        change_dataset_compression(test_filename, out_filename, 1, 1, True)

        dataset = HDF5Dataset(**default_config)
        data = dataset[0]

        file_size_original = os.stat(test_filename).st_size
        file_size_compressed = os.stat(out_filename).st_size

        os.remove(out_filename)

        self.assertIsInstance(data[0], torch.Tensor)
        self.assertTrue(file_size_original > file_size_compressed)

    def test_compression_stride(self):
        change_dataset_compression(test_filename, out_filename, 2, 2, False)

        dataset = HDF5Dataset(**default_config)
        data = dataset[0]

        file_size_original = os.stat(test_filename).st_size
        file_size_compressed = os.stat(out_filename).st_size

        os.remove(out_filename)

        self.assertIsInstance(data[0], torch.Tensor)
        self.assertTrue(file_size_original > file_size_compressed)

    def test_conversion(self):
        convert_dataset(tar_filename, out_filename, 200, 200, False, True, True, False)

        dataset = HDF5Dataset(**default_config)
        data = dataset[0]

        file_size = os.stat(out_filename).st_size

        os.remove(out_filename)

        self.assertTrue(file_size > 0)
        self.assertIsInstance(data[0], torch.Tensor)
        self.assertEqual(len(dataset), 2)

    def test_sample_dataset(self):
        config = copy.deepcopy(default_config)
        config['filename'] = test_filename
        config['return_name'] = True
        config['return_grid_size'] = True
        dataset = HDF5Dataset(**config)

        sample_dataset(dataset, out_filename, 6, False)

        out_dataset = HDF5Dataset(**default_config)
        data = out_dataset[0]

        file_size = os.stat(out_filename).st_size

        os.remove(out_filename)

        self.assertTrue(file_size > 0)
        self.assertIsInstance(data[0], torch.Tensor)
        self.assertEqual(len(out_dataset), 6)


if __name__ == '__main__':
    unittest.main()
