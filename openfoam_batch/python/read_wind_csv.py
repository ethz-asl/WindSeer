from __future__ import print_function
import numpy as np
import pandas as pd
import glob
import os
import argparse

WINDNX = 128
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
    if 'U:0' not in wind_data.keys():
        print('U:0 not in {0}'.format(infile))
        raise IOError
    # wind_data.drop(['U:1', 'Points:1'], axis=1)     # Get rid of y data
    # For some reason the rename doesn't work
    # wind_data.rename(
    #     index=str, columns={'U:0': 'Ux', 'U:2': 'Uz', 'vtkValidPointMask': 'is_air', 'Points:0': 'x', 'Points:2': 'z'})
    assert wind_data.shape[0] == NRECORDS

    # We actually want each column to be a 2D array
    return wind_data


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


def move_junk_data(in_directory, junk_directory, Uthresh=1.0e4, pthresh=5.0e3):
    all_files = glob.glob(in_directory+'/*.csv')
    n_files = len(all_files)
    junked_files = 0
    for i, wind_csv in enumerate(all_files):
        fname = os.path.basename(wind_csv)
        try:
            wind_out = read_wind_csv(wind_csv)
        except IOError:
            junked_files += 1
            print("{0}: File read failed (IOError), moving to junk. Junked ratio {n}/{t}".format(fname, n=junked_files, t=i))
            os.rename(wind_csv, os.path.join(junk_directory, fname))
            continue

        data_max = wind_out.max()
        data_min = wind_out.min()
        if ((data_max['U:0'] > Uthresh) or (data_max['U:2'] > Uthresh) or (data_min['U:0'] < -Uthresh) or (data_min['U:2'] < -Uthresh)
                or (data_max['p'] > pthresh) or (data_max['p'] < -pthresh)
                or (data_max['U:0'] - data_min['U:0'] < 0.1) or (data_max['p'] - data_min['p'] < 0.1)):
            junked_files += 1
            print("{0}: Value outside threshold, moving to junk. Junked ratio {n}/{t}".format(fname, n=junked_files, t=i))
            os.rename(wind_csv, os.path.join(junk_directory, fname))
    print("{0} files processed, {1} sent to junk".format(n_files, junked_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate terrainDict from stl file')
    parser.add_argument('-i', '--input-dir', required=False, default='data/train', help='Input directory of csv files')
    parser.add_argument('-j', '--junk-dir', required=False, default='data/junk', help='Destination directory for junk')
    parser.add_argument('-tU', '--threshold-U', required=False, default=1000.0, type=float, help='')
    parser.add_argument('-tp', '--threshold-p', required=False, default=5000.0, type=float, help='')
    args = parser.parse_args()
    move_junk_data(args.input_dir, args.junk_dir, Uthresh=args.threshold_U, pthresh=args.threshold_p)
