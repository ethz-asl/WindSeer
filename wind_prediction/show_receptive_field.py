#!/usr/bin/env python

from __future__ import print_function

import argparse
import nn_wind_prediction.models as models
import nn_wind_prediction.utils as utils
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Training an EDNN for predicting wind data from terrain')
parser.add_argument('-y', '--yaml-config', required=True, help='YAML config file')

args = parser.parse_args()

run_params = utils.EDNNParameters(args.yaml_config)

# define model
NetworkType = getattr(models, run_params.model['model_type'])

with torch.no_grad():
    # initialize model
    net = NetworkType(**run_params.model_kwargs())
    net.eval()
    net.set_receptive_field_params()

    x = torch.zeros(1,5,128,256,256)
    x[0,:, 0, 0, 0] = 1
    out = net(x)['pred'][0,0].numpy()

    plt.figure()
    plt.imshow(out[64])
    plt.show()
    
    import pdb
    pdb.set_trace()
    print('done')