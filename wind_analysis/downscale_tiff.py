import argparse
import matplotlib.pyplot as plt
from osgeo import gdal

from analysis_utils import get_mapgeo_terrain

parser = argparse.ArgumentParser(description='Rescale a geotiff')
parser.add_argument('-i', dest='input', required=True, help='Input geotiff')
parser.add_argument('-o', dest='output', required=True, help='Output geotiff')
parser.add_argument('-r', dest='resolution', required=True, type=float, help='Target resolution')
args = parser.parse_args()

gdal.Translate(args.output, args.input, xRes=args.resolution, yRes=args.resolution, resampleAlg="cubicspline")

# original image
plt.figure()
img = gdal.Open(args.input)
t = img.GetGeoTransform()
extent = [t[0] , t[0] + img.RasterXSize * t[1] , t[3] + img.RasterYSize * t[5], t[3]]
band = img.GetRasterBand(1)
data = band.ReadAsArray()
plt.imshow(data,extent=extent)

plt.figure()
img = gdal.Open(args.output)
t = img.GetGeoTransform()
extent = [t[0] , t[0] + img.RasterXSize * t[1] , t[3] + img.RasterYSize * t[5], t[3]]
band = img.GetRasterBand(1)
data = band.ReadAsArray()
plt.imshow(data,extent=extent)

plt.show()
