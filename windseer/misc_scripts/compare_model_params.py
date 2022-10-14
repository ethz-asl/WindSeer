from __future__ import print_function

import torch

def compare_model_weights(model1, model2):
    param1 = torch.load(model1)
    param2 = torch.load(model2)
    
    # first check if both models have the same keys
    if (param1.keys() != param2.keys()):
        print('The parameter of the two models are not the same')

    for key in param1.keys():
        print(key)
        diff = param1[key] - param2[key]
        print('\t{}, {}, {}'.format(param1[key].norm().item(), param2[key].norm().item(), diff.norm().item()))


if __name__ == '__main__':
    model1 = 'trained_models/pretrained1_naKd4sF8mK/latest.model'
    model2 = 'trained_models/pretrained3_naKd4sF8mK/latest.model'
    compare_model_weights(model1, model2)