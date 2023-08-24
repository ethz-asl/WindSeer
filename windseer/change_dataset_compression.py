#!/usr/bin/env python
'''
Change the size of a dataset
'''

from __future__ import print_function

import argparse
from windseer.data import change_dataset_compression
import time


def main():
    '''
    Main function which parses the arguments and then calls convert_data
    '''
    parser = argparse.ArgumentParser(
        description=
        'Script to compress a dataset either with lzf or by reducing the data size with striding.'
        )
    parser.add_argument('-i', dest='infile', required=True, help='Input hdf5 file')
    parser.add_argument(
        '-o',
        dest='outfile',
        help=
        'Output hdf5 file, if none is provided "compressed_" is prepended to the filename'
        )
    parser.add_argument(
        '-s_hor', default=1, type=int, help='Stride in horizontal direction'
        )
    parser.add_argument(
        '-s_ver', default=1, type=int, help='Stride in vertical direction'
        )
    parser.add_argument(
        '-c',
        dest='compress',
        action='store_true',
        help='If set the output dataset is compressed, else uncompressed'
        )
    args = parser.parse_args()

    if (args.outfile == args.infile):
        print(
            'WARNING: The outfile cannot be the same file as the infile, prepending "compressed_"'
            )
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
    change_dataset_compression(
        args.infile, args.outfile, args.s_hor, args.s_ver, args.compress
        )
    print("INFO: Compressing the database took %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
