#!/usr/bin/python
import os
import numpy as np
import argparse
from read_grd import create_blockMeshDict
from stl import mesh

def generate_blockMeshDict(stl_file, block_mesh, infile='blockMeshDict.in', nx=128, ny=128, nz=128, pad_z=3.0):

    if os.path.basename(block_mesh) is not 'blockMeshDict':
        print "Warning: Specified output \"{0}\" should be a blockMeshDict file".format(block_mesh)

    hill_mesh = mesh.Mesh.from_file(stl_file)

    # Shift origin to one corner
    hill_mesh.translate(-1.0*hill_mesh.min_)
    hill_mesh.update_min()
    hill_mesh.update_max()
    lims = np.zeros((3, 2), dtype='float')
    lims[:, 0] = hill_mesh.min_
    lims[:, 1] = hill_mesh.max_
    lims[2, 1] = lims[2, 0] + pad_z*(lims[2, 1] - lims[2, 0])

    bmesh_extras = {'nx': nx, 'ny': ny, 'nz': nz, 'infile': infile}
    create_blockMeshDict(block_mesh, lims, **bmesh_extras)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate blockMeshDict from stl file')
    parser.add_argument('-s', '--stl', required=True, help='Input stl file')
    parser.add_argument('-o', '--block-mesh-out', required=True, help='Output block mesh file')
    parser.add_argument('-in', '--block-mesh-in', default='blockMeshDict.in', help='Input block mesh file (usually .in)')
    parser.add_argument('-nx', type=int, default=128,
                        help='Number of points in x direction (uniform)')
    parser.add_argument('-ny', type=int, default=128,
                        help='Number of points in y direction (uniform)')
    parser.add_argument('-nz', type=int, default=64,
                        help='Number of points in z direction (uniform)')
    parser.add_argument('-pz', '--pad-z', type=float, default=3.0, help='Multiples of terrain height to add above mesh')
    args = parser.parse_args()

    generate_blockMeshDict(args.stl, args.block_mesh_out, args.block_mesh_in, nx=args.nx, ny=args.ny, nz=args.nz, pad_z=args.pad_z)