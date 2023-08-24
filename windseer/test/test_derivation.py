#!/usr/bin/env python
'''
Testcases for the KwargsParser
'''

from windseer.utils.derivation import *

import torch
import unittest

input_tensor = torch.tensor([[1, 2, 3], [1, 2, 3],
                             [6, 5, 4]]).unsqueeze(0).unsqueeze(-1).float()


class TestDerivation(unittest.TestCase):

    def test_derive(self):
        derivation_z = torch.tensor([[[[0.0], [0.0], [0.0]], [[2.5], [1.5], [0.5]],
                                      [[5.0], [3.0], [1.0]]]])

        derivation_y = torch.tensor([[[[1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]],
                                      [[-1.0], [-1.0], [-1.0]]]])

        self.assertTrue(torch.equal(derive(input_tensor, 1), derivation_z))
        self.assertTrue(torch.equal(derive(input_tensor, 2), derivation_y))

    def test_curl(self):
        self.assertTrue(
            torch.equal(
                curl(torch.ones(1, 3, 3, 3, 3), [1, 1, 1]), torch.zeros(1, 3, 3, 3, 3)
                )
            )
        self.assertTrue(
            torch.equal(
                curl(torch.arange(81).reshape(1, 3, 3, 3, 3).float(), [1, 1, 1]),
                torch.ones(1, 3, 3, 3, 3) *
                torch.tensor([[-6, 8, -2]]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                )
            )

    def test_gradient(self):
        self.assertTrue(
            torch.equal(
                gradient(torch.ones(1, 3, 3, 3, 3), [1, 1, 1]),
                torch.zeros(1, 9, 3, 3, 3)
                )
            )
        self.assertTrue(
            torch.equal(
                gradient(torch.arange(81).reshape(1, 3, 3, 3, 3).float(), [1, 1, 1]),
                torch.ones(1, 9, 3, 3, 3) *
                torch.tensor([[1, 3, 9, 1, 3, 9, 1, 3, 9]]
                             ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                )
            )

    def test_divergence(self):
        self.assertTrue(
            torch.equal(
                divergence(torch.ones(1, 3, 3, 3, 3), [1, 1, 1]),
                torch.zeros(1, 3, 3, 3)
                )
            )
        self.assertTrue(
            torch.equal(
                divergence(torch.arange(81).reshape(1, 3, 3, 3, 3).float(), [1, 1, 1]),
                torch.ones(1, 3, 3, 3) * 13.0
                )
            )


if __name__ == '__main__':
    unittest.main()
