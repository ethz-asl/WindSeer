# wind_analysis

## Installation

1. This code requires installation of the GDAL (v 2.3.2) library for working with geospatial data (geotiffs)
    ~~~
    wget http://download.osgeo.org/gdal/2.3.2/gdal-2.3.2.tar.gz
    tar -xzf gdal-2.3.2.tar.gz
    cd gdal-2.3.2
    ./configure --prefix=/usr/local
    make -j 8
    sudo make install
    gdalinfo --version
    ~~~

2. Install pygdal library
    `pip install pygdal==2.3.2.4`

3. Other dependencies
    `pip install numpy scipy pyproj`

4. Need to manually add the CH1903 definitions to pyproj. You can try to use this simple script.
    ~~~
    cd intel_wind/wind_analysis/proj_definitions
    python install_ch_defs.py
    ~~~
    If you have permission errors, you basically need to install the `CH` file in `pyproj_definitions` to the `data` subdirectory of the pyproj module. The script above should have returned an error specifying where the file was supposed to go. You can copy the file there manually with the required permissions.
    