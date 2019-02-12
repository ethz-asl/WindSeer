#!/usr/bin/env python

import os
import inspect
from shutil import copyfile

try:
    import pyproj
except ImportError:
    print("'pyproj' is not installed. Use the command below to install it:")
    print("     pip install pyproj")
    exit()

# Get the pyproj location
pyproj_dir = os.path.dirname(inspect.getfile(pyproj))
ch_target = os.path.join(pyproj_dir, 'data', 'CH')

# Copy the CH file into the pyproj data directory
if os.path.isfile(ch_target):
    print('{0} file already present!'.format(ch_target))
else:
    print('Attempting to copy CH to {0}'.format(ch_target))
    try:
        copyfile('CH', ch_target)
    except:
        print('Copy failed. Please copy the CH file manually to the specified location.')
        raise

# Check if it worked:
try:
    proj_output = pyproj.Proj(init="CH:1903_LV03")
    print('Loading CH:1903_LV03 projection succeeded!')
except:
    print('Loading projection failed!')
    raise

