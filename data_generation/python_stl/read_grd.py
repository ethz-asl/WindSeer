import numpy as np
import csv
from string import Template
from stl import mesh


def read_grd(infile):
    with open(infile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        type_str = reader.next()
        nx, ny = [int(v) for v in reader.next()]
        min_x, max_x = [float(v) for v in reader.next()]   # west, east
        min_y, max_y = [float(v) for v in reader.next()]    # north, south
        min_z, max_z = [float(v) for v in reader.next()]
        reader.next()
        Z = np.zeros((nx, ny))
        cx, cy = 0, 0
        for line in reader:
            nnx = len(line)
            Z[cx:(cx+nnx), cy] = [float(v) for v in line]
            cx += nnx
            if cx >= nx-1:
                cx = 0
                cy += 1

    x, y = np.linspace(min_x, max_x, nx), np.linspace(min_y, max_y, ny)
    return x, y, Z


def create_trimesh(x, y, z, subsample=1, verbose=True):
    x, y, z = x[::subsample], y[::subsample], z[::subsample, ::subsample]
    nx = len(x)
    ny = len(y)
    vertices = np.zeros([nx*ny, 3], dtype='float')
    triangles = []
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            nv = ny*i+j
            vertices[nv] = [xi, yj, z[i,j]]
            if i < nx-1 and j < ny-1:
                triangles.append([nv, nv+ny, nv+ny+1])
                triangles.append([nv, nv+ny+1, nv+1])
    triangles = np.array(triangles)

    terrian_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(triangles):
        for j in range(3):
            terrian_mesh.vectors[i][j] = vertices[f[j], :]
    if verbose:
        print "Created mesh with {0} vertices, {1} triangles.".format(vertices.shape[0], triangles.shape[0])
    return terrian_mesh


def create_blockMeshDict(outfile, xyz_lims, nx=10, ny=10, nz=10, infile = './blockMeshDict.in',
                         mconvert=1.0, in_buffer=0.0, gx=1, gy=1, gz=1, quiet=False):
    xyz_lims = np.array(xyz_lims)
    dx, dy, dz = [h - l for l, h in xyz_lims]
    lx, hx = xyz_lims[0]+ [in_buffer*dx, -in_buffer*dx]
    ly, hy = xyz_lims[1]+ [in_buffer*dy, -in_buffer*dy]
    lz, hz = xyz_lims[2]    # + [0.001*dz, 0.0]
    sub_dict = {'MINX': '{0:0.4f}'.format(lx), 'MAXX': '{0:0.4f}'.format(hx),
                'MINY': '{0:0.4f}'.format(ly), 'MAXY': '{0:0.4f}'.format(hy),
                'MINZ': '{0:0.4f}'.format(lz), 'MAXZ': '{0:0.4f}'.format(hz),
                'NX': '{0:d}'.format(nx), 'NY': '{0:d}'.format(ny), 'NZ': '{0:d}'.format(nz),
                'MCONVERT': '{0:0.2f}'.format(mconvert), 'GX': gx, 'GY': gy, 'GZ': gz}

    if not quiet:
        print "Creating outfile {0} from {1}".format(outfile, infile)
        print "Mesh limits: x in [{0}, {1}], y in [{2}, {3}], z in [{4}, {5}]".format(lx, hx, ly, hy, lz, hz)

    with open(infile, "r") as fh:
        src = Template(fh.read())
    mesh_dict = src.substitute(sub_dict)

    with open(outfile, "w") as out_fh:
        out_fh.write(mesh_dict)
