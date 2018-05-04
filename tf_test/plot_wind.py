import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WINDNX = 129
WINDNZ = 64
NRECORDS = WINDNX*WINDNZ

def read_wind_csv(infile):
    types = {"p": np.float32,
             "U:0": np.float32,
             "U:1": np.float32,
             "U:2": np.float32,
             "epsilon": np.float32,
             "k": np.float32,
             "nut": np.float32,
             "vtkValidPointMask": np.bool,
             "Points:0": np.float32,
             "Points:1": np.float32,
             "Points:2": np.float32}
    wind_data = pd.read_csv(infile, dtype=types)
    wind_data.drop(['U:1', 'Points:1'], axis=1)     # Get rid of y data
    assert wind_data.shape[0] == NRECORDS

    # We actually want each column to be a 2D array
    return wind_data


def plot_data(wind_data):
    fh, ah = plt.subplots(1, 2)
    fh.set_size_inches([10, 5])




