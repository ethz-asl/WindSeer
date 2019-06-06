# To do


## In file "Initialization.sh"

-   Set the path "scratch_dir": /cluster/scratch/kürzel

-   Set the path "batch_dir":  /cluster/scratch/kürzel/batch_name

-   Ev. change "dataset", "model_name" and "model_version"




## In Euler

1)  Create a clone of the git directory on /cluster/scratch/kürzel

2)  Copy all OpenFoam batch cases in a directory /cluster/scratch/kürzel/batch_name 

3)  Set Euler environment for OF:

        module load gcc/4.8.2 open_mpi/1.6.5 openfoam/4.1 qt/4.8.4
        foam-init

4)  Run Initialization.sh

        cd cluster/scratch/kürzel/intel_wind/data_generation/openfoam_batch/initialization   
        bsub -W 4:00 ./Initialization.sh
        
5)  Go in the directory of the case that you want to simulate and:

        bsub -W 4:00 simpleFoam


