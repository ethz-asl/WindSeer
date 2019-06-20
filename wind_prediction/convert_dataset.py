#!/usr/bin/env python

'''
Convert a dataset of csv files to serialized torch tensors.

TODO: Investigate how to speed up the 3D case
'''

from __future__ import print_function

import argparse
from nn_wind_prediction.data import convert_dataset
import time

def main():
    '''
    Main function which parses the arguments and then calls convert_data
    '''
    parser = argparse.ArgumentParser(description='Script to remove bad data from a database')
    parser.add_argument('-i', dest='infile', required=True, help='input tar file')
    parser.add_argument('-o', dest='outfile', help='output hdf5 file name, if none provided the input file name is used')
    parser.add_argument('-vlim', type=float, default=1000.0, help='limit of the velocity magnitude in one dimension')
    parser.add_argument('-klim', type=float, default=1000.0, help='limit of the turbulent viscosity')
    parser.add_argument('-v', dest='verbose', action='store_true', help='verbose')
    parser.add_argument('-b', dest='boolean_terrain', action='store_true', help='If flag is set the terrain is represented by a boolean variable, else by a distance field.')
    parser.add_argument('-czs', dest='create_zero_samples', action='store_true',help='Indicates if all zero samples should be created and saved for each different terrain')
    args = parser.parse_args()


    if (not args.outfile):
        in_splitted = args.infile.split('/')
        if len(in_splitted) > 1:
            out = ''
            for elem in in_splitted[0:-1]:
                out = out + elem + '/'

            args.outfile = out + in_splitted[-1]

        # remove .tar file extension
        args.outfile = args.outfile[0:-4]

    start_time = time.time()
    convert_dataset(args.infile, args.outfile, args.vlim, args.klim, args.boolean_terrain, args.verbose, args.create_zero_samples)
    print("INFO: Converting the database took %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
