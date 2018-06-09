import tensorflow as tf
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
        print 'U:0 not in {0}'.format(infile)
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


def build_tf_dataset(directory):
    # Load data from csv files:
    all_files = glob.glob(directory+'/Y*.csv')
    n_files = len(all_files)
    train_input = {'isWind': np.zeros([n_files, WINDNZ, WINDNX], dtype=np.float32),
                  'Ux_in': np.zeros([n_files, WINDNZ, WINDNX], dtype=np.float32),
                  'Uz_in': np.zeros([n_files, WINDNZ, WINDNX], dtype=np.float32)}
    train_labels = {'Ux_out': np.zeros([n_files, WINDNZ, WINDNX], dtype=np.float32),
                    'Uz_out': np.zeros([n_files, WINDNZ, WINDNX], dtype=np.float32)}

    for i, wind_csv in enumerate(all_files):
        try:
            wind_out = read_wind_csv(wind_csv)
        except IOError:
            continue

        train_labels['Ux_out'][i, :, :] = wind_out.get('U:0').values.reshape([WINDNZ, WINDNX])
        train_labels['Uz_out'][i, :, :] = wind_out.get('U:2').values.reshape([WINDNZ, WINDNX])

        wind_in = build_input_from_output(wind_out)
        train_input['isWind'][i, :, :] = wind_in.get('isWind').values.reshape([WINDNZ, WINDNX]).astype(np.float32)
        train_input['Ux_in'][i, :, :] = wind_in.get('Ux').values.reshape([WINDNZ, WINDNX])
        train_input['Uz_in'][i, :, :] = wind_in.get('Uz').values.reshape([WINDNZ, WINDNX])

    tf_train_features = tf.data.Dataset.from_tensor_slices(train_input)
    tf_train_labels   = tf.data.Dataset.from_tensor_slices(train_labels)
    return tf.data.Dataset.zip((tf_train_features, tf_train_labels))


def move_junk_data(in_directory, junk_directory, thresh=1.0e4):
    all_files = glob.glob(in_directory+'/*.csv')
    n_files = len(all_files)
    junked_files = 0
    for i, wind_csv in enumerate(all_files):
        fname = os.path.basename(wind_csv)
        try:
            wind_out = read_wind_csv(wind_csv)
        except IOError:
            print "{0}: File read failed (IOError), moving to junk".format(fname)
            os.rename(wind_csv, os.path.join(junk_directory, fname))
            junked_files += 1
            continue

        data_max = wind_out.max()
        data_min = wind_out.min()
        if (data_max['U:0'] > thresh) or (data_max['U:2'] > thresh) or (data_min['U:0'] < -thresh) or (data_min['U:2'] < -thresh):
            print "{0}: Value outside threshold, moving to junk.".format(fname)
            os.rename(wind_csv, os.path.join(junk_directory, fname))
            junked_files += 1
    print "{0} files processed, {1} sent to junk".format(n_files, junked_files)


        # get('U:0')
        # train_labels['Uz_out'][i, :, :] = wind_out.get('U:2').values.reshape([WINDNZ, WINDNX])



def feeder_tf_dataset(directory):
    # Load data from csv files:
    file_list = tf.train.match_filenames_once(directory + '/Y*.csv')
    file_reader = tf.train.string_input_producer(file_list, shuffle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate terrainDict from stl file')
    parser.add_argument('-i', '--input-dir', required=False, default='data/train', help='Input directory of csv files')
    parser.add_argument('-j', '--junk-dir', required=False, default='data/junk', help='Destination directory for junk')
    args = parser.parse_args()
    move_junk_data(args.input_dir, args.junk_dir)