#!/usr/bin/env python

'''
Convert a dataset of csv files to serialized torch tensors.

TODO: Investigate how to speed up the 3D case
'''

import argparse
import io
import lz4.frame
from math import trunc
import numpy as np
import os
import pandas as pd
from scipy import ndimage
from subprocess import call
import shutil
import sys
import tarfile
import time
import torch

def save_data(tensor, name, compress):
    if compress:
        bytes = io.BytesIO()
        torch.save(tensor, bytes)
        f = open(name, 'wb')
        f.write(lz4.frame.compress(bytes.getvalue(), compression_level = -20))
        f.close()
    else:
        torch.save(tensor, name)

def compress_data(infile, outfile, s_hor, s_ver, input_compressed, compress):
    # open the file
    tar = tarfile.open(infile, 'r')
    num_files = len(tar.getnames())

    # create temp directory to store all serialized arrays
    if (os.path.isdir("/cluster/scratch/")):
        tempfolder = '/scratch/tmp_' + time.strftime("%Y_%m_%d-%H_%M_%S") + '/'
    else:
        tempfolder = 'tmp_' + time.strftime("%Y_%m_%d-%H_%M_%S") + '/'

    os.makedirs(tempfolder)

    print('INFO: Looping through all the files')
    for i, member in enumerate(tar.getmembers()):
        file = tar.extractfile(member)
        
        if input_compressed:
            data = torch.load(io.BytesIO(lz4.frame.decompress(file.read())))

        else:
            data = torch.load(file)
        
        if (len(list(data.size())) > 3):
            out = data[:,::s_ver,::s_hor, ::s_hor].clone()
        else:
            out = data[:,::s_ver, ::s_hor].clone()

        save_data(out, tempfolder + member.name, compress)

        if ((i % np.ceil(num_files/10.0)) == 0.0):
            print(trunc((i+1)/num_files*100), '%')

    print('INFO: Finished compressing all the files')

    # collecting all files in the tmp folder to a tar
    out_tar = tarfile.open(outfile, 'w')
    for filename in os.listdir(tempfolder):
        out_tar.add(tempfolder + filename, arcname = filename)

    # cleaning up
    out_tar.close()
    tar.close()
    shutil.rmtree(tempfolder)


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
    compress_data(args.infile, args.outfile, args.s_hor, args.s_ver, args.input_compressed, args.compress)
    print("INFO: Compressing the database took %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
