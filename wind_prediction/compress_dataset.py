#!/usr/bin/env python

'''
Convert a dataset of csv files to serialized torch tensors.

TODO: Investigate how to speed up the 3D case
'''

from __future__ import print_function

import argparse
from nn_wind_prediction.data import compress_dataset
import time

def main():
    '''
    Main function which parses the arguments and then calls convert_data
    '''
    parser = argparse.ArgumentParser(description='Script to remove bad data from a database')
    parser.add_argument('-i', dest='infile', required=True, help='input tar file')
    parser.add_argument('-o', dest='outfile', help='output tar file, if none provided the input file name is prepended with "compressed_"')
    parser.add_argument('-s_hor', default=1, type=int, help='stride in horizontal direction')
    parser.add_argument('-s_ver', default=1, type=int, help='stride in vertical direction')
    parser.add_argument('-ic', dest='input_compressed', action='store_true', help='If true the input file is compressed')
    parser.add_argument('-c', dest='compress', action='store_true', help='compress the individual tensors')
    args = parser.parse_args()

    if (args.outfile == args.infile):
        print('WARNING: The outfile cannot be the same file as the infile, prepending "compressed_"')
        args.outfile = None

    if (not args.outfile):
        in_splitted = args.infile.split('/')
        if len(in_splitted) > 1:
            out = ''
            for elem in in_splitted[0:-1]:
                out = out + elem + '/'

            args.outfile = out + 'compressed_' + in_splitted[-1]

        else:
            args.outfile = 'compressed_' + args.infile

    start_time = time.time()
    compress_dataset(args.infile, args.outfile, args.s_hor, args.s_ver, args.input_compressed, args.compress)
    print("INFO: Compressing the database took %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
