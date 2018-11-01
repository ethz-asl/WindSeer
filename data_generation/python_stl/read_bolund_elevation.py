import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import csv
from read_grd import read_grd

x, y, Z = read_grd('../data/Bolund.grd')

with open('../data/Measurements/Dir_270.dat', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', skipinitialspace=True)
    # ID invL Samples x[m] y[m] z[m] gl[m] u*[m/s] vel/u* u/u* v/u* w/u* tke/u*^2 uu/u*^2 vv/u*^2 ww/u*^2 u*/u*
    header = reader.next()
    measurements = []
    measurement_ids = []
    for i, line in enumerate(reader):
        measurement_ids.append(line[0])
        measurements.append([float(v) for v in line[1:]])
    measurements = np.array(measurements, dtype=float)


fh, ah = plt.subplots()
ah.matshow(Z.T, origin='lower', interpolation='none', cmap=cm.terrain, extent=[min_x, max_x, min_y, max_y])
ah.xaxis.tick_bottom()
ah.set_xlabel('Easting (m)')
ah.set_ylabel('Northing (m)')

fh2 = plt.figure(figsize=(10, 4))
ah2 = fh2.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ah2.plot_surface(X, Y, Z.T, cmap=cm.terrain, linewidth=0, antialiased=False, rcount=80, ccount=80)
u_star = measurements[:, 6]
u, v, w = u_star*measurements[:, 8], u_star*measurements[:, 9], u_star*measurements[:, 10]
# Get colors
c_array = [v for v in measurements[:, 7]]
for v in measurements[:, 7]:
    c_array.extend([v, v])
c_array = np.array(c_array)
c_array /= c_array.max()
q = ah2.quiver(measurements[:, 2], measurements[:, 3], measurements[:, 4], u, v, w, colors=cm.jet(c_array))


ah2.set_xlim(x[0], x[-1])
ah2.set_ylim(y[0], y[-1])
ah2.set_zlim(0, 50)
ah2.set_xlabel('Easting (m)')
ah2.set_ylabel('Northing (m)')
ah2.set_zlabel('Altitude (m)')

# fh.savefig('../fig/bolund_flat.pdf', bbox_inches='tight')
# fh2.savefig('../fig/bolund.pdf', bbox_inches='tight')
plt.show(block=False)

