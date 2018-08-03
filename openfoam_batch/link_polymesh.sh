#!/bin/bash

home_dir=$(pwd)
rebuild_links()
{
    poly_dir="$1/constant/polyMesh"
    if [ -d "$poly_dir" ]; then
        rm -r "$poly_dir"
        ln -s ../../W1/constant/polyMesh "$poly_dir"
    fi
}

for dir in $@; do
    cd "$dir"
    for wd in ./W1[0-9]; do
        rebuild_links $wd
    done
    for wd in ./W[2-9]; do
        rebuild_links $wd
    done
    cd "$home_dir"
done
