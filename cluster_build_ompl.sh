# load the modules
module load gcc/4.8.5
module load boost/1.63.0
module load eigen
module load hdf5

# create the software directory
cd ~
mkdir software
cd software

# clone ompl and create the build directories
git clone https://github.com/ompl/ompl.git
cd ompl
git checkout 1.1.1
mkdir -p build/Release
cd build/Release

# delete the ompl custom FindEigen because it cannot find the installed Eigen module
rm ../../CMakeModules/FindEigen3.cmake

# execute the building on a cluster computer
# only submit the job here because on the cluster git clone is not possible
bsub -Is -n 8 -W 1:00 -R "rusage[mem=4096]" bas

# build and install ompl
cmake -DCMAKE_INSTALL_PREFIX=~/software/ompl/build/lib ../../
make -j 8
make install

# return to the login node
#exit
