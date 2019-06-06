#!/usr/bin/env python


import numpy as np
import io
import linecache as lc
import time
import os
import torch
import argparse


batch = 'batch02_F_109_S15x15_W08_t00430.npy'
zMax = 1200


parser = argparse.ArgumentParser(description='Script to create IC for OF')
parser.add_argument('-batch', dest='batch', default=batch, help='Name of the batch')
parser.add_argument('-z', dest='zMax', default=zMax, help='Altitude of the mesh')
args = parser.parse_args()

# Import prediction
prediction = torch.tensor(np.load(args.batch))

xMax = 1500
yMax = 1500
zMax = int(args.zMax)




west_coord = np.loadtxt('WestCoordinates')
n_west =len(west_coord)


#find nearest y
iy_west = west_coord[:,1]

y0=0
y = yMax/64

for i in range(0,64):
    iy_west=np.where( (iy_west<y)&(iy_west>=y0), i, iy_west)

    y0=y
    y=y+yMax/64


#find nearest z
iz_west = west_coord[:,2]

z0=0
z = zMax/64

for i in range(0,64):
    iz_west=np.where( (iz_west<z)&(iz_west>=z0), i, iz_west)

    z0=z
    z=z+zMax/64

west = np.zeros( (n_west, 7) )


for j in range(0,n_west):
    west[j,:] = prediction[:,int(iz_west[j]),int(iy_west[j]),0]

    i = 0
    while west[j,5] == 0.0:
        i=i+1
        west[j,:] = prediction[:,int(iz_west[j])+i,int(iy_west[j]),0]






east_coord = np.loadtxt('EastCoordinates')
n_east =len(east_coord)


#find nearest y
iy_east = east_coord[:,1]

y0=0
y = yMax/64

for i in range(0,64):
    iy_east=np.where( (iy_east<y)&(iy_east>=y0), i, iy_east)

    y0=y
    y=y+yMax/64

#find nearest z
iz_east = east_coord[:,2]

z0=0
z = zMax/64

for i in range(0,64):
    iz_east=np.where( (iz_east<z)&(iz_east>=z0), i, iz_east)

    z0=z
    z=z+zMax/64

east = np.zeros( (n_east, 7) )


for j in range(0,n_east):
    east[j,:] = prediction[:,int(iz_east[j]),int(iy_east[j]),63]

    i = 0
    while east[j,5] == 0.0:
        i=i+1
        east[j,:] = prediction[:,int(iz_east[j])+i,int(iy_east[j]),63]





hill_coord = np.loadtxt('HillCoordinates')
n_hill =len(hill_coord)


#find nearest x
ix_hill = hill_coord[:,0]

x0=0
x = xMax/64

for i in range(0,64):
    ix_hill=np.where( (ix_hill<x)&(ix_hill>=x0), i, ix_hill)

    x0=x
    x=x+xMax/64


#find nearest y
iy_hill = hill_coord[:,1]

y0=0
y = yMax/64

for i in range(0,64):
    iy_hill=np.where( (iy_hill<y)&(iy_hill>=y0), i, iy_hill)

    y0=y
    y=y+yMax/64


#find nearest z (prendo quello sopra il terreno)
iz_hill = hill_coord[:,2]

z0=0
z = zMax/64

for i in range(0,64):

    iz_hill=np.where( (iz_hill<z)&(iz_hill>=z0), i, iz_hill)

    z0=z
    z=z+zMax/64


hill = np.zeros( (n_hill, 7) )

for j in range(0,n_hill):
    hill[j,:] = prediction[:,int(iz_hill[j]),int(iy_hill[j]),int(ix_hill[j])]

    i = 0
    while hill[j,5] == 0.0:
        i=i+1
        hill[j,:] = prediction[:,int(iz_hill[j])+i,int(iy_hill[j]),int(ix_hill[j])]




# Internal field
cells_coord = np.loadtxt('CellCoordinates')
n_cells = len(cells_coord)

#find nearest x
ix_cell = cells_coord[:,0]

x0=0
x = xMax/64

for i in range(0,64):
    ix_cell=np.where( (ix_cell<x)&(ix_cell>=x0), i, ix_cell)

    x0=x
    x=x+xMax/64

#find nearest y
iy_cell = cells_coord[:,1]

y0=0
y = yMax/64

for i in range(0,64):
    iy_cell=np.where( (iy_cell<y)&(iy_cell>=y0), i, iy_cell)

    y0=y
    y=y+yMax/64


#find nearest z (solo se sopra il terreno)
iz_cell = cells_coord[:,2]

z0=0
z = zMax/64

for i in range(0,64):
    iz_cell=np.where( (iz_cell<z)&(iz_cell>=z0), i, iz_cell)

    z0=z
    z=z+zMax/64

cells = np.zeros( (n_cells, 7) )

for b in range(0,n_cells):
    cells[b,:] = prediction[:,int(iz_cell[b]),int(iy_cell[b]),int(ix_cell[b])]

    i = 0
    while cells[b,5] == 0.0:
        i=i+1
        cells[b,:] = prediction[:,int(iz_cell[b])+i,int(iy_cell[b]),int(ix_cell[b])]




# epsilon
with open("OF_0/epsilon_in",'w') as epsilon_in:
    np.savetxt(epsilon_in, abs(cells[:,5]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_cells)+'\n(', footer=')\n', comments='')
with open("OF_0/epsilon_west",'w') as epsilon_west:
    np.savetxt(epsilon_west, abs(west[:,5]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_west)+'\n(', footer=')\n', comments='')
with open("OF_0/epsilon_east",'w') as epsilon_east:
    np.savetxt(epsilon_east, abs(east[:,5]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_east)+'\n(', footer=')\n', comments='')
with open("OF_0/epsilon_hill",'w') as epsilon_hill:
    np.savetxt(epsilon_hill, abs(hill[:,5]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_hill)+'\n(', footer=')\n', comments='')

# k
with open("OF_0/k_in",'w') as k_in:
    np.savetxt(k_in, abs(cells[:,3]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_cells)+'\n(', footer=')\n', comments='')
with open("OF_0/k_east",'w') as k_east:
    np.savetxt(k_east, abs(east[:,3]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_east)+'\n(', footer=')\n', comments='')
with open("OF_0/k_hill",'w') as k_hill:
    np.savetxt(k_hill, abs(hill[:,3]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_hill)+'\n(', footer=')\n', comments='')

# nut
with open("OF_0/nut_in",'w') as nut_in:
    np.savetxt(nut_in, abs(cells[:,6]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_cells)+'\n(', footer=')\n', comments='')
with open("OF_0/nut_west",'w') as nut_west:
    np.savetxt(nut_west, abs(west[:,6]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_west)+'\n(', footer=')\n', comments='')
with open("OF_0/nut_east",'w') as nut_east:
    np.savetxt(nut_east, abs(east[:,6]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_east)+'\n(', footer=')\n', comments='')
with open("OF_0/nut_hill",'w') as nut_hill:
    np.savetxt(nut_hill, abs(hill[:,6]) , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_hill)+'\n(', footer=')\n', comments='')

# p
with open("OF_0/p_in",'w') as p_in:
    np.savetxt(p_in, cells[:,4] , fmt='%.11f', header='nonuniform List<scalar> \n '+str(n_cells)+'\n(', footer=')\n', comments='')

# U 
with open("OF_0/U_in",'w') as U_in:
    U_in.write("nonuniform List<vector>\n%i\n(\n" %(n_cells))
    for j in range(0,n_cells):
        U_in.write("(%f %f %f)\n" %(cells[j,0],cells[j,1],cells[j,2]))
    U_in.write(")\n")

with open("OF_0/U_west",'w') as U_west:
    U_west.write("nonuniform List<vector>\n%i\n(\n" %(n_west))
    for j in range(0,n_west):
        U_west.write("(%f %f %f)\n" %(west[j,0],west[j,1],west[j,2]))
    U_west.write(")\n")

with open("OF_0/U_east",'w') as U_east:
    U_east.write("nonuniform List<vector>\n%i\n(\n" %(n_east))
    for j in range(0,n_east):
        U_east.write("(%f %f %f)\n" %(east[j,0],east[j,1],east[j,2]))
    U_east.write(")\n")
