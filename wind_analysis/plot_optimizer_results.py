import argparse
import numpy as np

import nn_wind_prediction.models as models
import nn_wind_prediction.utils as nn_utils
from analysis_utils import utils
from analysis_utils.plotting_analysis import plot_optimizer_results
from analysis_utils.WindOptimizer import WindOptimizer

parser = argparse.ArgumentParser(description='Visualize the results of the optimizer')
parser.add_argument('file', help='File containing the results')
args = parser.parse_args()

results = np.load(args.file, allow_pickle=True).item()

plot_optimizer_results(results)
