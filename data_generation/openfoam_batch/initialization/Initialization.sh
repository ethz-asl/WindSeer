#!/bin/bash



## DEFAULT DIRECTORIES

# Scratch directory
scratch_dir="/cluster/scratch/ftamara"		

# Intel wind directory
IW_dir="$scratch_dir/intel_wind"
# Wind prediction directory
WP_dir="$IW_dir/wind_prediction"
# initialization folder
IN_dir="$IW_dir/data_generation/openfoam_batch/initialization"

# Dataset
dataset="data/unconverged.tar"
# Model
model_name="full_dataset_neu3"
# Model version
model_version="latest"

# Batch
batch="batch05_N"
batch_dir="${scratch_dir}/${batch}"









# FIND PREDICTION

cd $WP_dir

#Number of files inside the tar
nr_tar=$(tar -tf $dataset | wc -l)
i_tar=0

#Loop to find prediction for all cases
while  (( "$i_tar" < "$nr_tar" ))
do
	python3 save_prediction_all.py -ds $dataset -model_name $model_name -model_version $model_version -i $i_tar
	i_tar=$[$i_tar+1]
done

#Move npy files 
cp data/batch0* "$IN_dir/npy_files"
rm data/batch0*



NPY_FILES="$IN_dir/npy_files/*"



# CONVERT TO ASCII

for f in $NPY_FILES 
do
	npy_name=$(echo $f| cut -d'/' -f 10)
  	batch_number=$(echo $npy_name| cut -d'_' -f 3)
  	batch_direction=$(echo $npy_name| cut -d'_' -f 4)

  	cd "${IN_dir}/coord"

  	if [ ! -f "CellCoordinates_${batch_number}_${batch_direction}" ]
  	then
		# Convert to ascii
		cd "${batch_dir}/${batch}_${batch_number}_${batch_direction}/simpleFoam"
		foamDictionary -entry 'writeFormat' -set "ascii" system/controlDict
		foamFormatConvert > /dev/null
	fi
done



# MESH

for f in $NPY_FILES 
do
	npy_name=$(echo $f| cut -d'/' -f 10)
  	batch_number=$(echo $npy_name| cut -d'_' -f 3)
  	batch_direction=$(echo $npy_name| cut -d'_' -f 4)

  	cd $IN_dir/coord

  	if [ ! -f "CellCoordinates_${batch_number}_${batch_direction}" ]
  	then
		# Run c++ file
		cd $IN_dir
		./coordinates.out "$batch_number" "$batch_direction" "$batch" "$scratch_dir"
	fi
done




# CONVERT TO BINARY

for f in $NPY_FILES 
do
	npy_name=$(echo $f| cut -d'/' -f 10)
  	batch_number=$(echo $npy_name| cut -d'_' -f 3)
  	batch_direction=$(echo $npy_name| cut -d'_' -f 4)

  	cd "${IN_dir}/coord"

  	if [ ! -f "CellCoordinates_${batch_number}_${batch_direction}" ]
  	then
		# Convert to ascii
		cd "${batch_dir}/${batch}_${batch_number}_${batch_direction}/simpleFoam"
		foamDictionary -entry 'writeFormat' -set "binary" system/controlDict
		foamFormatConvert > /dev/null
	fi
done



# INITIAL VECTORS

for f in $NPY_FILES 
do
	npy_name=$(echo $f| cut -d'/' -f 10)
  	batch_number=$(echo $npy_name| cut -d'_' -f 3)
  	batch_direction=$(echo $npy_name| cut -d'_' -f 4)
  	batch_velocity=$(echo $npy_name| cut -d'_' -f 5)

	cd "$batch_dir/${batch}_${batch_number}_${batch_direction}"
  	maxZ_line=$(grep "MAXZ" terrainDict)
  	maxZ1=$(echo $maxZ_line| cut -d' ' -f 2)
  	maxZ=$(echo $maxZ1| cut -d';' -f 1)

	cd $IN_dir
  	# Create initial vector
  	python3 InitialVector.py -batch $npy_name -z $maxZ 	
done



# SETTING OF

for f in $NPY_FILES 
do
	npy_name=$(echo $f| cut -d'/' -f 2)
  	batch_number=$(echo $npy_name| cut -d'_' -f 3)
  	batch_direction=$(echo $npy_name| cut -d'_' -f 4)
  	batch_velocity=$(echo $npy_name| cut -d'_' -f 5)

  	if [ $batch_velocity == "W01" ]
  	then
  		batch_velocity0="W1"
  	elif [ $batch_velocity == "W02" ]
  	then
  		batch_velocity0="W2"
  	elif [ $batch_velocity == "W03" ]
  	then
  		batch_velocity0="W3"
  	elif [ $batch_velocity == "W04" ]
  	then
  		batch_velocity0="W4"
  	elif [ $batch_velocity == "W05" ]
  	then
  		batch_velocity0="W5"
  	elif [ $batch_velocity == "W06" ]
  	then
  		batch_velocity0="W6"
  	elif [ $batch_velocity == "W07" ]
  	then
  		batch_velocity0="W7"
  	elif [ $batch_velocity == "W08" ]
  	then
  		batch_velocity0="W8"
  	elif [ $batch_velocity == "W09" ]
  	then
  		batch_velocity0="W9"
    else 
        batch_velocity0=$batch_velocity
  	fi


    cd "${IN_dir}/OF_0"
    mv "epsilon_in_${batch_number}_${batch_direction}_${batch_velocity}" "epsilon_in"
    mv "epsilon_west_${batch_number}_${batch_direction}_${batch_velocity}" "epsilon_west"
    mv "epsilon_east_${batch_number}_${batch_direction}_${batch_velocity}" "epsilon_east"
    mv "epsilon_hill_${batch_number}_${batch_direction}_${batch_velocity}" "epsilon_hill"
    mv "k_in_${batch_number}_${batch_direction}_${batch_velocity}" "k_in"
    mv "k_east_${batch_number}_${batch_direction}_${batch_velocity}" "k_east"
    mv "k_hill_${batch_number}_${batch_direction}_${batch_velocity}" "k_hill"
    mv "nut_in_${batch_number}_${batch_direction}_${batch_velocity}" "nut_in"
    mv "nut_west_${batch_number}_${batch_direction}_${batch_velocity}" "nut_west"
    mv "nut_east_${batch_number}_${batch_direction}_${batch_velocity}" "nut_east"
    mv "nut_hill_${batch_number}_${batch_direction}_${batch_velocity}" "nut_hill"
    mv "p_in_${batch_number}_${batch_direction}_${batch_velocity}" "p_in"
    mv "U_in_${batch_number}_${batch_direction}_${batch_velocity}" "U_in"
    mv "U_west_${batch_number}_${batch_direction}_${batch_velocity}" "U_west"
    mv "U_east_${batch_number}_${batch_direction}_${batch_velocity}" "U_east"




	cd "${batch_dir}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}"

    # Run simulation for 1 timestep (from 0) 
    foamDictionary -entry startFrom -set startTime system/controlDict
    foamDictionary -entry startTime -set 0 system/controlDict
    foamDictionary -entry endTime -set 1 system/controlDict
    foamDictionary -entry writeInterval -set 1 system/controlDict
    foamDictionary -entry writeFormat -set ascii system/controlDict

	simpleFoam

	# Start from 4000
	mv 1 4000
	rm -r 4000/phi 4000/uniform

	foamDictionary -entry FoamFile.location -set \"4000\" 4000/epsilon
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/k
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/nut
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/p
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/U

	foamDictionary -entry startTime -set 4000 system/controlDict
	foamDictionary -entry endTime -set 14000 system/controlDict
	foamDictionary -entry writeInterval -set 5000 system/controlDict

    cd "${scratch_dir}"	# Enter initial vector in 4000 files
	foamDictionary -entry internalField -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/epsilon_in"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/epsilon" > /dev/null
	foamDictionary -entry boundaryField.west_face.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/epsilon_west"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/epsilon" > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/epsilon_east"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/epsilon" > /dev/null
	foamDictionary -entry boundaryField.hill_geometry.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/epsilon_hill"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/epsilon" > /dev/null

	foamDictionary -entry internalField -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/k_in"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/k" > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/k_east"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/k" > /dev/null
	foamDictionary -entry boundaryField.hill_geometry.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/k_hill"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/k" > /dev/null

	foamDictionary -entry internalField -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/nut_in"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/nut" > /dev/null
	foamDictionary -entry boundaryField.west_face.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/nut_west"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/nut" > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/nut_east"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/nut" > /dev/null
	foamDictionary -entry boundaryField.hill_geometry.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/nut_hill"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/nut" > /dev/null

	foamDictionary -entry internalField -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/p_in"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/p" > /dev/null

	foamDictionary -entry internalField -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/U_in"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/U" > /dev/null
	foamDictionary -entry boundaryField.west_face.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/U_west"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/U" > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "intel_wind/data_generation/openfoam_batch/initialization/OF_0/U_east"' "${batch}/${batch}_${batch_number}_${batch_direction}/${batch_velocity0}/4000/U" > /dev/null


    cd "${IN_dir}/OF_0"

    rm epsilon_east epsilon_hill epsilon_in epsilon_west 
    rm k_east k_hill k_in 
    rm nut_east nut_hill nut_in nut_west
    rm p_in
    rm U_east U_in U_west
done




