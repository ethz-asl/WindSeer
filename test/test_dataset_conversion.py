#!/usr/bin/env python

'''
Test with a test csv file if the conversion is working
'''

import argparse
from nn_wind_prediction.data import convert_dataset, compress_dataset
import time

def main():
    # convert the csv file to the tensor
    convert_dataset('test_csv.tar', 'converted.tar', 200, 128, 128, 64, 1000, True, True, False, False)

    # reduce the resolution from 128x128x64 to 32x32x32
    compress_dataset('converted.tar', 'compressed.tar', 4, 2, False, False)

    # TODO check if the conversion was correct

if __name__ == "__main__":
    main()
