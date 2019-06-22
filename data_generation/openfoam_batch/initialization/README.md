# To do


## In file "Initialization.sh"

1)   Set the path "scratch_dir": /cluster/scratch/kürzel

2)   Set "dataset", "model_name" and "model_version"

3)   Set "batch"




## In Euler

1)  Create a clone of the git directory on /cluster/scratch/kürzel

2)  Copy the OpenFoam batch cases in a directory /cluster/scratch/kürzel/batch (batch must correspond to the one set in the initialization.sh)

3)  Set Euler environment for OF and python

        module load gcc/4.8.2 open_mpi/1.6.5 openfoam/4.1 qt/4.8.4 python/3.7.1 new llvm/4.0.1
        python -m pip install --user tensorboardX lz4 numpy tqdm matplotlib scipy  pandas h5py interpolation termcolor pyyaml
        pip install --user torch==0.4.1
        foam-init
        
4)  Install nn_wind_prediction package: go in intel_wind directory and execute:

        python -m pip install --user -e wind_prediction

5)  Create the executable for "Coordinates.cpp" and run Initialization.sh

        cd cluster/scratch/kürzel/intel_wind/data_generation/openfoam_batch/initialization 
        clang++ Coordinates.cpp -o coordinates.out
        bsub -W 4:00 ./Initialization.sh
        
6)  Go in the directory of the case that you want to simulate and:

        bsub -n 1 -W 4:00 -R "rusage[mem=11000]" simpleFoam

