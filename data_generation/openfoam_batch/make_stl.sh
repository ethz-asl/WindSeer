#!/bin/bash
# ./make_stl.sh PHSTL_DIR DIRECTORY_WITH_TIFS DIRECTORY_FOR_STLS
for file in $2/*.tif; do 
	base_file=$( basename $file .tif); 
	python "${1}/phstl.py" "${file}" "${3}/${base_file}.stl"
done
