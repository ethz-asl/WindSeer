#### import the simple module from the paraview
from paraview.simple import *
import os.path
import argparse

parser = argparse.ArgumentParser(description='Resample onto new mesh using paraview')
parser.add_argument('-cd', '--case-dir', default=None, required=True, help='Case directory')
parser.add_argument('-md', '--mesh-dir', default=None, required=True, help='Source mesh directory')
parser.add_argument('-cf', '--case-foam', default='bolund.foam',
    help='Input case foam file (in case dir)')
parser.add_argument('-cm', '--mesh-foam', default='testgrid.foam',
    help='Test grid foam file (in mesh dir)')
parser.add_argument('-o', '--outfile', default='structured_mesh.csv',
    help='Output csv file')
args = parser.parse_args()

# create a new 'OpenFOAMReader'
inputfoam = OpenFOAMReader(FileName=os.path.join(args.case_dir, args.case_foam), SkipZeroTime=True)
inputfoam.MeshRegions = ['internalMesh']
inputfoam.CellArrays = ['U', 'epsilon', 'k', 'nut', 'p']

# Properties modified on inputfoam
inputfoam.MeshRegions = ['north_face']

# create a new 'OpenFOAMReader'
testgridfoam = OpenFOAMReader(FileName=os.path.join(args.mesh_dir, args.mesh_foam))
testgridfoam.MeshRegions = ['internalMesh']

# Properties modified on testgridfoam
testgridfoam.MeshRegions = ['south_face']

# set active source
SetActiveSource(inputfoam)

# create a new 'Resample With Dataset'
resampleWithDataset1 = ResampleWithDataset(Input=inputfoam,
    Source=testgridfoam)

# Properties modified on resampleWithDataset1
resampleWithDataset1.Tolerance = 2.22044604925031e-16

# save data
SaveData(args.outfile, proxy=resampleWithDataset1)
