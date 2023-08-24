#!/usr/bin/env python

from __future__ import print_function

import argparse

import windseer.utils as utils
import windseer.nn as nn_custom

parser = argparse.ArgumentParser(
    description='Training a neural network for predicting wind data from terrain'
    )
parser.add_argument('-y', '--yaml-config', required=True, help='YAML config file')
parser.add_argument(
    '-o', '--output-dir', default='trained_models/', help='Output directory'
    )
parser.add_argument(
    '-w',
    '--writer',
    dest='use_writer',
    default=True,
    action='store_false',
    help='Do not use a SummaryWriter to log the learningcurve'
    )
parser.add_argument(
    '-c',
    '--copy',
    dest='copy_datasets',
    action='store_true',
    help=
    'Copy the dataset files to a local machine (should only be used on the cluster as it cleans up the tempfolder afterwards)'
    )

args = parser.parse_args()

configs = utils.WindseerParams(args.yaml_config)

# start the actual training
nn_custom.train_model(configs, args.output_dir, args.use_writer, args.copy_datasets)
