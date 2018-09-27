import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
import read_grd

subsample = 1
xy_grade = 5.0
z_grade = 8
in_buff = -0.2      # Positive will shrink by x, negative will enlarge by x ratio of dx
file_dir = '../data/'
infile = file_dir+'Bolund'
x, y, Z = read_grd.read_grd(infile+'.grd')
terrain_mesh = read_grd.create_trimesh(x, y, Z, subsample=subsample)

fh, ah = plt.subplots()
ah.matshow(Z.T, origin='lower', interpolation='none', cmap=cm.terrain, extent=[x[0], x[-1], y[0], y[-1]])
ah.xaxis.tick_bottom()
ah.set_xlabel('Easting (m)')
ah.set_ylabel('Northing (m)')

if terrain_mesh.vectors.shape[0] <= 50000:
    fh2 = plt.figure(figsize=(10, 4))
    ah2 = fh2.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(x, y)
    # ah2.plot_surface(X, Y, Z.T, cmap=cm.terrain, linewidth=0, antialiased=False, rcount=80, ccount=80)
    ah2.add_collection(mplot3d.art3d.Poly3DCollection(terrain_mesh.vectors))
    ah2.set_xlim(x[0], x[-1])
    ah2.set_ylim(y[0], y[-1])
    ah2.set_zlim(0, 50)
    ah2.set_xlabel('Easting (m)')
    ah2.set_ylabel('Northing (m)')
    ah2.set_zlabel('Altitude (m)')

# fh.savefig('../fig/bolund_flat.pdf', bbox_inches='tight')
# fh2.savefig('../fig/bolund.pdf', bbox_inches='tight')
plt.show(block=False)
# terrain_mesh.save(infile+'.stl')
lims = [[x[0], x[-1]], [y[0], y[-1]], [Z.min(), 70.0]]
def build_grade(g, c=0.5):
    return '( ({0} {1} {2}) ({3} {4} {5}) )'.format(c, c, 1.0/g, (1-c), (1-c), g)

bmesh_extras = {'nx': 80, 'ny': 80, 'nz': 50,
                'in_buffer': in_buff, 'gz': z_grade, 'gx': build_grade(xy_grade, c=0.3), 'gy': build_grade(xy_grade)}
read_grd.create_blockMeshDict(file_dir+'blockMeshDict', lims, **bmesh_extras)