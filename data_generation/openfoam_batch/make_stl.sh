#!/bin/bash
# ./make_stl.sh DIRECTORY_WITH_TIFS DIRECTORY_FOR_STLS
for file in $1/*.tif; do 
	base_file=$( basename $file .tif); 
	python phstl/phstl.py "${file}" "${2}/${base_file}.stl"
done
