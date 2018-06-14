from __future__ import print_function

#### import the simple module from the paraview
import paraview.simple as para
import os.path
import argparse

parser = argparse.ArgumentParser(description='Resample onto new mesh using paraview')
parser.add_argument('-cd', '--case-dir', default=None, required=True, help='Case directory')
parser.add_argument('-md', '--mesh-dir', default=None, required=True, help='Source mesh directory')
parser.add_argument('-cf', '--case-foam', default='bolund.foam',
    help='Input case foam file (in case dir)')
parser.add_argument('-cm', '--mesh-foam', default='testgrid.foam',
    help='Test grid foam file (in mesh dir)')
parser.add_argument('-o', '--outfile', default='structured_mesh',
    help='Output csv file (without file extension)')
args = parser.parse_args()

# create a new 'OpenFOAMReader'
inputfoam = para.OpenFOAMReader(FileName=os.path.join(args.case_dir, args.case_foam), SkipZeroTime=True)
inputfoam.MeshRegions = ['north_face']
inputfoam.CellArrays = ['U', 'epsilon', 'k', 'nut', 'p']

# create a new 'OpenFOAMReader'
testgridfoam = para.OpenFOAMReader(FileName=os.path.join(args.mesh_dir, args.mesh_foam))
testgridfoam.MeshRegions = ['south_face']

for t in inputfoam.TimestepValues:
    inputfoam.UpdatePipeline(time=t)
    tfile = args.outfile+'_t{0:04.0f}.csv'.format(t)
    print("Output file set to {0}".format(tfile))

    # set active source
    para.SetActiveSource(inputfoam)

    # create a new 'Resample With Dataset'
    resampleWithDataset1 = para.ResampleWithDataset(Input=inputfoam, Source=testgridfoam)

    # Properties modified on resampleWithDataset1
    resampleWithDataset1.Tolerance = 2.22044604925031e-16

    # save data
    para.SaveData(tfile, proxy=resampleWithDataset1)
