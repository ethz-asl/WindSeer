#!/usr/bin/python
import os
import sys
import numpy as np
import argparse
from stl import mesh
from string import Template


def create_terrainDict(outfile, xyz_lims, stl_file, nx=10, ny=10, nz=10, infile='./terrainDict.in',
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
                'MCONVERT': '{0:0.2f}'.format(mconvert), 'GX': gx, 'GY': gy, 'GZ': gz,
                'STL_FILE': '"{0}"'.format(os.path.basename(stl_file))}

    if not quiet:
        print "Creating outfile {0} from {1}".format(outfile, infile)
        print "Mesh limits: x in [{0}, {1}], y in [{2}, {3}], z in [{4}, {5}]".format(lx, hx, ly, hy, lz, hz)

    with open(infile, "r") as fh:
        src = Template(fh.read())
    mesh_dict = src.substitute(sub_dict)

    with open(outfile, "w") as out_fh:
        out_fh.write(mesh_dict)


def generate_terrainDict(stl_file, dict_out, stl_out, infile='terrainDict.in', nx=128, ny=128, nz=128, pad_z=3.0):

    # if os.path.basename(dict_out) is not 'terrainDict':
    #     print "Warning: Specified output \"{0}\" should be a terrainDict file".format(dict_out)

    hill_mesh = mesh.Mesh.from_file(stl_file)

    # Shift origin to one corner
    hill_mesh.translate(-1.0*hill_mesh.min_)
    hill_mesh.update_min()
    hill_mesh.update_max()
    lims = np.zeros((3, 2), dtype='float')
    lims[:, 0] = hill_mesh.min_
    lims[:, 1] = hill_mesh.max_
    lims[2, 1] = lims[2, 0] + pad_z*(lims[2, 1] - lims[2, 0])

    bmesh_extras = {'nx': nx, 'ny': ny, 'nz': nz, 'infile': infile, 'quiet': True}
    create_terrainDict(dict_out, lims, stl_out, **bmesh_extras)
    hill_mesh.save(stl_out)
    return lims


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate terrainDict from stl file')
    parser.add_argument('-si', '--stl-in', required=True, help='Input stl file')
    parser.add_argument('-so', '--stl-out', required=False, default=None, help='Output stl file')
    parser.add_argument('-do', '--dict-out', required=True, help='Output dictionary file')
    parser.add_argument('-di', '--dict-in', default='terrainDict.in', help='Input dictionary file (usually .in)')
    parser.add_argument('-nx', type=int, default=128,
                        help='Number of points in x direction (uniform)')
    parser.add_argument('-ny', type=int, default=128,
                        help='Number of points in y direction (uniform)')
    parser.add_argument('-nz', type=int, default=64,
                        help='Number of points in z direction (uniform)')
    parser.add_argument('-pz', '--pad-z', type=float, default=2.0, help='Multiples of terrain height to add above mesh')
    args = parser.parse_args()

    lims = generate_terrainDict(stl_file=args.stl_in, dict_out=args.dict_out, stl_out=args.stl_out,
                                infile=args.dict_in, nx=args.nx, ny=args.ny, nz=args.nz, pad_z=args.pad_z)
    print '{0:0.2f} {1:0.2f}'.format(lims[1, 0], lims[1, 1])
