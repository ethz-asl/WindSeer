#!/usr/bin/env python

import os
from shutil import copyfile

try:
    import pyproj
except ImportError:
    print("'pyproj' is not installed. Use the command below to install it:")
    print("     pip install pyproj")
    exit()


def copy_with_check(src, dst):
    if os.path.isfile(dst):
        print('{0} file already exists, not copying.'.format(dst))
    else:
        print('Attempting to copy {0} to {1}'.format(src, dst))
        try:
            copyfile(src, dst)
            print('Copied {0} to {1}'.format(src, dst))
        except:
            print('Copy failed. Please copy the file manually to the specified location.')
            raise


# Copy files to pyproj data directory
for file in ['CH', 'chenyx06etrs.gsb']:
    try:
        destination = os.path.join(pyproj.pyproj_datadir, file)
    except AttributError:
        destination = os.path.join(pyproj.datadir.get_data_dir(), file)
    copy_with_check(file, destination)

# Check if it worked:
try:
    proj_ch = pyproj.Proj(init="CH:1903_LV03")
    print('Loading CH:1903_LV03 projection succeeded!')
    proj_wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')
    print('Loading WGS84 projection succeeded!')
    [e], [n], [alt] = pyproj.transform(proj_wgs84, proj_ch, [7.438632451], [46.951082814], [248.913])
    if abs(e - 6e5) < 1.0 and abs(n - 2e5) < 1.0:
        print('Successful conversion from WGS84 -> LV03')
    else:
        print('Conversion failed. e = {0:0.5f}, n = {0:0.5f}')
except:
    print('Loading projection failed!')
    raise

