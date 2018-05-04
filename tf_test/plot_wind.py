import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WINDNX = 129
WINDNZ = 65
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
    # For some reason the rename doesn't work
    # wind_data.rename(
    #     index=str, columns={'U:0': 'Ux', 'U:2': 'Uz', 'vtkValidPointMask': 'is_air', 'Points:0': 'x', 'Points:2': 'z'})
    assert wind_data.shape[0] == NRECORDS

    # We actually want each column to be a 2D array
    return wind_data


def plot_data(wind_data, sp='p', sUx='U:0', sUz='U:2'):
    fh, ah = plt.subplots(2, 1)     # , {'aspect':'equal'})
    fh.set_size_inches([5, 8])

    p = wind_data.get(sp).values.reshape([WINDNZ, WINDNX])
    # x = wind_data.get('Points:0').values.reshape([WINDNZ, WINDNX])
    # z = wind_data.get('Points:2').values.reshape([WINDNZ, WINDNX])
    Ux = wind_data.get(sUx).values.reshape([WINDNZ, WINDNX])
    Uz = wind_data.get(sUz).values.reshape([WINDNZ, WINDNX])

    ah[0].imshow(p, origin='lower')
    ah[1].quiver(Ux, Uz, np.sqrt(Ux**2 + Uz**2))
    ah[1].set_aspect('equal')
    return fh, ah

def build_input_from_output(wind_data):
    # Copy wind across valid locations, build ground occupancy (binary)
    Ux_in = np.zeros(NRECORDS)
    jj = 0
    for i in range(WINDNZ):
        Ux_in[jj:(jj+WINDNX)] = wind_data.get("U:0")[i*WINDNX]
        jj += WINDNX
    Uz_in = np.zeros(NRECORDS)
    input_data = pd.DataFrame({
        'isWind': wind_data.get('vtkValidPointMask').values,
        'Ux': Ux_in,
        'Uz': Uz_in})
    return input_data

wind = read_wind_csv('data/Y+001W150.csv')
fig, ax = plot_data(wind)

wind_in = build_input_from_output(wind)
f_in, a_in = plot_data(wind_in, sp='isWind', sUx='Ux', sUz='Uz')
plt.show(block=False)
