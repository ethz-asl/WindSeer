from __future__ import print_function

import argparse
import torch


def compare_model_weights(model1, model2, eps, verbose):
    param1 = torch.load(model1)
    param2 = torch.load(model2)

    # first check if both models have the same keys
    if (param1.keys() != param2.keys()):
        print('The parameter of the two models are not the same')

    for key in param1.keys():
        diff = param1[key] - param2[key]
        if verbose:
            print(key)
            print(
                '\t{}, {}, {}'.format(
                    param1[key].norm().item(), param2[key].norm().item(),
                    diff.norm().item()
                    )
                )
        else:
            if diff.max() > eps:
                print(key, ', norm difference: ', diff.norm().item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to compare two model parameter'
        )
    parser.add_argument(
        '-m1', dest='model1', required=True, help='Path to the first model weights'
        )
    parser.add_argument(
        '-m2', dest='model2', required=True, help='Path to the second model weights'
        )
    parser.add_argument(
        '-v', dest='verbose', action='store_true', help='Verbose output'
        )
    parser.add_argument(
        '-e',
        '--eps',
        dest='eps',
        type=float,
        default=1e-6,
        help='Threshold when comparing parameter'
        )
    args = parser.parse_args()

    compare_model_weights(args.model1, args.model2, args.eps, args.verbose)
