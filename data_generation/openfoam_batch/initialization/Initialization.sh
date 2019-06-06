#!/bin/bash



# DEFAULT DIRECTORIES

# Scratch directory
scratch_dir="/cluster/scratch/ftamara"		

# Intel wind directory
IW_dir="$scratch_dir/intel_wind"
# Wind prediction directory
WP_dir="$IW_dir/wind_prediction"
# initialization folder
IN_dir="$IW_dir/data_generation/openfoam_batch/initialization"

# Dataset
dataset="data/single_terrain.tar"			#############################
# Model
model_name="full_dataset_model-autoscale"	#############################
# Model version
model_version="lastes"						#############################

# Batch
batch_dir="$scratch_dir/batch02_F"
# PolyMesh directory
PM_dir="constant/polyMesh"









# FIND PREDICTION

cd $WP_dir

# Number of files inside the tar
nr_tar=$(tar -tf $dataset | wc -l)
i_tar=0

# Loop to find prediction for all cases
while  (( "$i_tar" < "$nr_tar" ))
do
	python3 save_prediction_all.py -ds $dataset -model_name $model_name -model_version $model_version -i $i_tar
	i_tar=$[$i_tar+1]
done

# Move npy files 
cp data/batch02* "$IN_dir/npy_files"
rm data/batch02*
cd


NPY_FILES="$IN_dir/npy_files/*"


# MESH

for f in $NPY_FILES 
do
	npy_name=$(echo $f| cut -d'/' -f 10)
  	batch_number=$(echo $npy_name| cut -d'_' -f 3)
  	batch_direction=$(echo $npy_name| cut -d'_' -f 4)

  	cd $IN_dir/coord

  	if [ ! -f "CellCoordinates_${batch_number}_${batch_direction}" ]
  	then

 	  	cd "$batch_dir/batch02_F_${batch_number}_${batch_direction}/simpleFoam"

		# Convert to ascii
		foamDictionary -entry 'writeFormat' -set "ascii" system/controlDict
		foamFormatConvert

		# Run c++ file
		g++ -o coordinates Coordinates.cpp
		./coordinates "${PM_dir}" "${IN_dir}"

		# Reconvert to binary
		foamDictionary -entry 'writeFormat' -set "binary" system/controlDict
		foamFormatConvert

		# Rename file
		cd IN_dir/coord
		mv CellCoordinates CellCoordinates_${batch_number}_${batch_direction}
		mv EastCoordinates EastCoordinates_${batch_number}_${batch_direction}
		mv HillCoordinates HillCoordinates_${batch_number}_${batch_direction}
		mv WestCoordinates WestCoordinates_${batch_number}_${batch_direction}

	fi

done


# SETTING OF

for f in $NPY_FILES 
do
	npy_name=$(echo $f| cut -d'/' -f 10)
  	batch_number=$(echo $npy_name| cut -d'_' -f 3)
  	batch_direction=$(echo $npy_name| cut -d'_' -f 4)
  	batch_velocity=$(echo $npy_name| cut -d'_' -f 5)


  	cd "$batch_dir/batch02_F_${batch_number}_${batch_direction}"
  	zMax=$(foamDictionary -entry TERRAIN_DICT.MAXZ -value terrainDict)

	cd $IN_dir
  	# Create initial vector
  	python3 InitialVector.py -batch $npy_name -z $zMax


  	cd "$batch_dir/batch02_F_${batch_number}_${batch_direction}/${batch_velocity}"

  	# Run simulation for 1 timestep (from 0) 
	foamDictionary -entry startTime -set 0 system/controlDict
	foamDictionary -entry endTime -set 1 system/controlDict
	foamDictionary -entry writeInterval -set 1 system/controlDict
	simpleFoam

	# Start from 4000
	mv 1 4000
	rm -r 4000/phi 4000/uniform

	foamDictionary -entry FoamFile.location -set \"4000\" 4000/epsilon
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/k
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/nut
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/p
	foamDictionary -entry FoamFile.location -set \"4000\" 4000/U

	foamDictionary -entry startFrom -set startTime system/controlDict
	foamDictionary -entry startTime -set 4000 system/controlDict
	foamDictionary -entry endTime -set 14000 system/controlDict
	foamDictionary -entry writeInterval -set 5000 system/controlDict

	# Enter initial vector in 4000 files
	foamDictionary -entry internalField -set '#include "$IN_dir/OF_0/epsilon_in"' 4000/epsilon > /dev/null
	foamDictionary -entry boundaryField.west_face.value -set '#include "$IN_dir/OF_0/epsilon_west"' 4000/epsilon > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "$IN_dir/OF_0/epsilon_east"' 4000/epsilon > /dev/null
	foamDictionary -entry boundaryField.hill_geometry.value -set '#include "$IN_dir/OF_0/epsilon_hill"' 4000/epsilon > /dev/null

	foamDictionary -entry internalField -set '#include "$IN_dir/OF_0/k_in"' 4000/k > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "$IN_dir/OF_0/k_east"' 4000/k > /dev/null
	foamDictionary -entry boundaryField.hill_geometry.value -set '#include "$IN_dir/OF_0/k_hill"' 4000/k > /dev/null

	foamDictionary -entry internalField -set '#include "$IN_dir/OF_0/nut_in"' 4000/nut > /dev/null
	foamDictionary -entry boundaryField.west_face.value -set '#include "$IN_dir/OF_0/nut_west"' 4000/nut > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "$IN_dir/OF_0/nut_east"' 4000/nut > /dev/null
	foamDictionary -entry boundaryField.hill_geometry.value -set '#include "$IN_dir/OF_0/nut_hill"' 4000/nut > /dev/null

	foamDictionary -entry internalField -set '#include "$IN_dir/OF_0/p_in"' 4000/p > /dev/null

	foamDictionary -entry internalField -set '#include "$IN_dir/OF_0/U_in"' 4000/U > /dev/null
	foamDictionary -entry boundaryField.west_face.value -set '#include "$IN_dir/OF_0/U_west"' 4000/U > /dev/null
	foamDictionary -entry boundaryField.east_face.value -set '#include "$IN_dir/OF_0/U_east"' 4000/U > /dev/null

	rm "$IN_dir/OF_0/*"

done




