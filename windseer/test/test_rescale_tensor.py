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
config_filename = os.path.join(
    os.path.split(windseer_path)[0], 'configs', 'example.yaml'
    )


class TestRescaleTensor(unittest.TestCase):

    def test_rescale_tensor(self):
        params = utils.WindseerParams(config_filename)
        input = torch.ones(1, 2, 3, 3, 3)
        scale = 0.5

        params.data['ux_scaling'] = 0.1
        params.data['uy_scaling'] = 1.0
        out = utils.rescale_tensor(input, ['ux', 'uy'], scale, params)
        self.assertTrue(
            torch.equal(out,
                        input * torch.Tensor([[[[[0.1]]], [[[1.0]]]]]) * scale)
            )

        params.data['ux_scaling'] = 0.1
        params.data['uy_scaling'] = 1.0
        out = utils.rescale_tensor(input, ['uy', 'ux'], scale, params)
        self.assertTrue(
            torch.equal(out,
                        input * torch.Tensor([[[[[0.1]]], [[[1.0]]]]]) * scale)
            )

        params.data['ux_scaling'] = 1.0
        params.data['uy_scaling'] = 0.1
        out = utils.rescale_tensor(input, ['ux', 'uy'], scale, params)
        self.assertTrue(
            torch.equal(out,
                        input * torch.Tensor([[[[[1.0]]], [[[0.1]]]]]) * scale)
            )

        params.data['p_scaling'] = 0.02
        params.data['turb_scaling'] = 2.0
        out = utils.rescale_tensor(input, ['turb', 'p'], scale, params)
        self.assertTrue(
            torch.equal(
                out,
                input * torch.Tensor([[[[[2.0]]], [[[0.02]]]]]) * scale * scale
                )
            )

        input = torch.ones(1, 3, 3, 3)

        params.data['terrain_scaling'] = 0.02
        out = utils.rescale_tensor(input, ['terrain'], scale, params)
        self.assertTrue(torch.equal(out, input * torch.Tensor([[[[0.02]]]])))

        params.data['epsilon_scaling'] = 5.0
        out = utils.rescale_tensor(input, ['epsilon'], scale, params)
        self.assertTrue(
            torch.equal(out,
                        input * torch.Tensor([[[[5.0]]]]) * scale * scale * scale)
            )

        input = torch.ones(10, 2, 3, 3, 3)
        params.data['ux_scaling'] = 0.1
        params.data['uy_scaling'] = 1.0
        out = utils.rescale_tensor(input, ['ux', 'uy'], scale, params)
        self.assertTrue(
            torch.equal(out,
                        input * torch.Tensor([[[[[0.1]]], [[[1.0]]]]]) * scale)
            )


if __name__ == '__main__':
    unittest.main()
