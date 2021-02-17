import argparse
import numpy as np

from analysis_utils.plotting_analysis import plot_optimizer_results

parser = argparse.ArgumentParser(description='Visualize the results of the optimizer')
parser.add_argument('file', help='File containing the results')
args = parser.parse_args()

results = np.load(args.file, allow_pickle=True).item()

plot_optimizer_results(results)
