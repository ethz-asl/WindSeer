#!/usr/bin/python
from __future__ import print_function
import os
import sys
import numpy as np
import argparse
from stl import mesh
from string import Template
from scipy.optimize import newton


def grading_function(k, n, L, ds):
    return L/ds - (np.power(k, n)-1)/(k-1)


def dgrading_function_dk(k, n, *args, **kwargs):
    return (np.power(k, n-1)*(k*(1-n)+n) - 1)/np.power(k-1, 2)


def r_from_k(k, n):
    return np.power(k, n-1)


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
        print("Creating outfile {0} from {1}".format(outfile, infile))
        print("Mesh limits: x in [{0}, {1}], y in [{2}, {3}], z in [{4}, {5}]".format(lx, hx, ly, hy, lz, hz))

    with open(infile, "r") as fh:
        src = Template(fh.read())
    mesh_dict = src.substitute(sub_dict)

    with open(outfile, "w") as out_fh:
        out_fh.write(mesh_dict)


def process_stl(stl_in, dict_in, stl_out, dict_out, nx=128, ny=128, nz=128, pad_z=3.0, gz=False, min_height=0.0, rotate=0):

    hill_mesh = mesh.Mesh.from_file(stl_in)
    if rotate != 0:
        hill_mesh.rotate(np.array([0,0,1]), rotate*np.pi/180.0)

    # Shift origin to one corner
    hill_mesh.translate(-1.0*hill_mesh.min_)
    hill_mesh.update_min()
    hill_mesh.update_max()
    lims = np.zeros((3, 2), dtype='float')
    terrain_size = hill_mesh.max_ - hill_mesh.min_
    lims[:, 0] = hill_mesh.min_
    lims[:, 1] = hill_mesh.max_
    lims[2, 1] = max(lims[2, 0] + pad_z*(hill_mesh.max_[2] - hill_mesh.min_[2]), min_height)
    if (lims[2, 1] - lims[2,0])/nz > 20.0:
        nz = int((lims[2, 1] - lims[2,0])/20.0)

    bmesh_extras = {'nx': nx, 'ny': ny, 'nz': nz, 'infile': dict_in, 'quiet': True}

    if gz:
        # Would like to have enough points in z so that the terrain has roughly cubic blocks
        # Assume x and y are already roughly similar, so we base on x cell size

        z_range = (lims[2, 1] - lims[2, 0])
        x_cell = max(terrain_size[0]/nx, terrain_size[1]/ny)        # max edge length of cells in x or y dir
        z_cell = z_range/nz               # edge length of cells in z dir
        if z_cell > 1.5*x_cell or z_cell < 0.5*x_cell:
            height_terrain = terrain_size[2]                # Height of terrain block (in real units)
            nz_terrain = int(height_terrain/x_cell)         # Number of cells in terrain block z
            ppz_terrain = min(0.65, float(nz_terrain)/nz)   # Proportion of cells in terrain block z
            nz_terrain = int(ppz_terrain*nz)
            phz_terrain = height_terrain/z_range  # Proportion of total height in terrain z

            # Calculate new grading to match cell sizes
            dz_terrain = height_terrain/nz_terrain      # Height of z cell in terrain block
            height_air = z_range - height_terrain       # Total height of air block
            dz_air = height_air/(nz-nz_terrain)         # Mean height of air cell (if uniform)
            nz_air = nz - nz_terrain
            if dz_terrain < dz_air:
                k_air = newton(grading_function, 1.5, fprime=dgrading_function_dk,
                               args=(nz_air, height_air, dz_terrain))
            elif dz_terrain > dz_air:
                k_air = newton(grading_function, 0.9, fprime=dgrading_function_dk,
                               args=(nz_air, height_air, dz_terrain))
            else:
                k_air = 1
            r_air = r_from_k(k_air, nz_air)

            bmesh_extras['gz'] = '( ({phzt:0.3f} {ppzt:0.3f} 1) ({phza:0.3f} {ppza:0.3f} {rza:0.2f}) )'.format(
                ppzt=ppz_terrain, phzt=phz_terrain, ppza=(1.0-ppz_terrain), phza=(1.0-phz_terrain), rza=r_air)
    create_terrainDict(dict_out, lims, stl_out, **bmesh_extras)
    hill_mesh.save(stl_out)
    return lims


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate terrainDict from stl file')
    parser.add_argument('-si', '--stl-in', required=True, help='Input stl file')
    parser.add_argument('-so', '--stl-out', required=True, default=None, help='Output stl file')
    parser.add_argument('-di', '--dict-in', default='terrainDict.in', help='Input dictionary file (usually .in)')
    parser.add_argument('-do', '--dict-out', required=True, help='Output dictionary file')
    parser.add_argument('-nx', type=int, default=128,
                        help='Number of points in x direction (uniform)')
    parser.add_argument('-ny', type=int, default=128,
                        help='Number of points in y direction (uniform)')
    parser.add_argument('-nz', type=int, default=64,
                        help='Number of points in z direction (uniform)')
    parser.add_argument('-mh', type=float, default=0.0,
                        help='Minimum block height in m')
    parser.add_argument('-pz', '--pad-z', type=float, default=2.0, help='Multiples of terrain height to add above mesh')
    parser.add_argument('-gz', '--autograde-z', action='store_true', required=False,
                        help='Automatically grade z for cubic cells')
    parser.add_argument('-r', '--rotate', default=0.0, type=float, required=False,
                        help='Rotate stl mesh about z (vertical) axis')
    args = parser.parse_args()

    limits = process_stl(stl_in=args.stl_in, dict_in=args.dict_in, stl_out=args.stl_out, dict_out=args.dict_out,
                       nx=args.nx, ny=args.ny, nz=args.nz, pad_z=args.pad_z,
                       gz=args.autograde_z, min_height=args.mh, rotate=args.rotate)
    print('{0:0.2f} {1:0.2f}'.format(limits[1, 0], limits[1, 1]))
