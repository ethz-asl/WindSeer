# This script load the required modules on the cluster
# and sets the necessary environment variables

#modules
module load gcc/8.2.0
module load eigen
module load hdf5/1.10.1
module load python_gpu/3.10.4


# set the stacksize, if not set the benchmark planner will run out of memory and generate a Segfault
export OMP_STACKSIZE=8m
