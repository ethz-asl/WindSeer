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
    parser.add_argument('-o', dest='outfile', help='output tar file, if none provided the input file name is prepended with "converted_"')
    parser.add_argument('-vlim', type=float, default=1000.0, help='limit of the velocity magnitude in one dimension')
    parser.add_argument('-klim', type=float, default=1000.0, help='limit of the turbulent viscosity')
    parser.add_argument('-v', dest='verbose', action='store_true', help='verbose')
    parser.add_argument('-c', dest='compress', action='store_true', help='compress the individual tensors')
    parser.add_argument('-b', dest='boolean_terrain', action='store_true', help='If flag is set the terrain is represented by a boolean variable, else by a distance field.')
    parser.add_argument('-a', dest='add_all', action='store_true',help='Add all variables (if false: add only U and k)')
    args = parser.parse_args()

    if (args.outfile == args.infile):
        print('WARNING: The outfile cannot be the same file as the infile, prepending "converted_"')
        args.outfile = None

    if (not args.outfile):
        in_splitted = args.infile.split('/')
        if len(in_splitted) > 1:
            out = ''
            for elem in in_splitted[0:-1]:
                out = out + elem + '/'

            args.outfile = out + 'converted_' + in_splitted[-1]

        else:
            args.outfile = 'converted_' + args.infile

    start_time = time.time()
    convert_dataset(args.infile, args.outfile, args.vlim, args.klim, args.boolean_terrain, args.verbose, args.compress, args.add_all)
    print("INFO: Converting the database took %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
