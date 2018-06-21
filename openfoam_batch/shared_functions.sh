#!/bin/bash

constant_files=(transportProperties turbulenceProperties)
system_files=(controlDict fvSchemes fvSolution)

create_base_case()
{
    local f
    # Argument 1 is the case name directory, arg 2 is the file source
    mkdir -p $1/system
    for f in ${system_files[@]}; do
        ln -s $2/system/$f $1/system/$f
    done

    mkdir -p $1/constant
    for f in ${constant_files[@]}; do
        ln -s $2/constant/$f $1/constant/$f
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
