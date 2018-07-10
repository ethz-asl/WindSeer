import numpy as np
import matplotlib.pyplot as plt
import argparse
import os.path

def read_logs(filelist):
    loaded_data = np.loadtxt(filelist[0])
    X = np.zeros((loaded_data.shape[0], len(filelist)+1))
    X[:, 0:2] = loaded_data
    var_names = [os.path.basename(filelist[0])]
    for i, file in enumerate(filelist[1:]):
        X[:, i+2] = np.loadtxt(file)[:loaded_data.shape[0],1]
        var_names.append(os.path.basename(file))
    return X, var_names


def plot_convergence(X, var_names):
    fh, ah = plt.subplots()
    h_lines = []
    for y in (X.T)[1:]:
        h_lines.extend(ah.plot((X.T)[0], y))
    ah.set_ylabel('Residual')
    ah.legend(h_lines, var_names, loc='best')
    ah.set_yscale('log')
    ah.set_xlabel('Iteration')
    return fh, ah


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot OpenFOAM convergence residuals')
    parser.add_argument('log_files', metavar='FILES', nargs='+', help='log files to plot')
    args = parser.parse_args()

    X, var_names = read_logs(args.log_files)
    fh, ah = plot_convergence(X, var_names)
    plt.show()