#!/usr/bin/env python

import argparse
from math import trunc
import numpy as np
import pandas as pd
import subprocess
from shutil import copyfile
import sys
import time
import zipfile

def clean_data(infile, outfile, vlim, verbose = False):
    print('INFO: Parsing all the files')
    copyfile(infile, outfile)

    delete_list = []

    with zipfile.ZipFile(outfile, "r") as zip:
        nameslist = zip.namelist()
        num_files = len(nameslist)

        for i, name in enumerate(nameslist):
            f = zip.open(name)

            if f is not None:
                content = pd.read_csv(f, header=0)

                max_0 = np.max(content['U:0'])
                min_0 = np.min(content['U:0'])
                max_2 = np.max(content['U:2'])
                min_2 = np.min(content['U:2'])

                if ((np.abs(max_0) > vlim) or
                    (np.abs(min_0) > vlim) or
                    (np.abs(max_2) > vlim) or
                    (np.abs(min_2) > vlim) or
                    ((max_0 == 0.0) and (max_2 == 0.0) and (min_0 == 0.0) and (min_2 == 0.0))): #empty file

                    delete_list.append(name)
                    if verbose:
                        print('------------------------------------')
                        print('Removing', name)
                        print('Statistics: max U0:', max_0, ', maxU2:', max_2, ', minU0:', min_0, ', minU2:', min_2)
                        print('------------------------------------')

            if ((i % np.ceil(num_files/10.0)) == 0.0):
                print(trunc((i+1)/num_files*100), '%')

    print('100 %')
    print('INFO: Finished parsing all the files, deleting', len(delete_list), 'of them')

    for item in delete_list:
        subprocess.call('zip -d ' + outfile + ' ' + item, shell=True)


def main():
    parser = argparse.ArgumentParser(description='Script to remove bad data from a database')
    parser.add_argument('-i', dest='infile', required=True, help='input zip file')
    parser.add_argument('-o', dest='outfile', help='output zip file, if none provided the input file name is prepended with "clean_"')
    parser.add_argument('-vlim', type=float, help='limit of the velocity magnitude in one dimension')
    parser.add_argument('-v', dest='verbose', action='store_true', help='verbose')

    args = parser.parse_args()
    
    if (args.outfile == args.infile):
        print('WARNING: The outfile cannot be the same file as the infile, prepending "clean_"')
        args.outfile = None

    if (not args.outfile):
        in_splitted = args.infile.split('/')
        if len(in_splitted) > 1:
            out = ''
            for elem in in_splitted[0:-1]:
                out = out + elem + '/'
                
            args.outfile = out + 'clean_' + in_splitted[-1]
                
        else:
            args.outfile = 'clean_' + args.infile

    if (not args.vlim):
        default_val = 30.0
        args.vlim = default_val
        print('INFO: No velocity limit provided, using the default value of', default_val, 'm/s')

    start_time = time.time()
    clean_data(args.infile, args.outfile, args.vlim, args.verbose)
    print("INFO: Cleaning the database took %s seconds" % (time.time() - start_time))



if __name__ == "__main__":
    main()
