#!/bin/bash

constant_files=(transportProperties turbulenceProperties)
system_files=(controlDict fvSchemes fvSolution)

create_base_case()
{
    # Create a new OpenFOAM directory at location $1 using symlinks from $2
    # Argument 1 is the target directory
    # Argument 2 is the file source
    # Argument 3 contains flags for ln
    local f
    mkdir -p $1/system
    for f in ${system_files[@]}; do
        ln -s $3 $2/system/$f $1/system/$f
    done

    mkdir -p $1/constant
    for f in ${constant_files[@]}; do
        ln -s $3 $2/constant/$f $1/constant/$f
    done
}

check_files()
{
    local return_value
    return_value=0

    for file in "$@"; do
        if [ ! -f $file ]; then
            echo "Required file $file not found."
            return_value=1
        fi
    done
    return $return_value
}


check_path() {
    # This will check if the first input is a global path, if it is, return it, else
    # return $2/$1
    local output_path=""
    [[ $1 == /* ]] && output_path=$1 || output_path="${2}/${1}"
    if [[ -d "$output_path" ]]; then
        echo "$output_path"
        return 0
    else
        echo "Invalid path: $output_path" >&2
        return 1
    fi
}

