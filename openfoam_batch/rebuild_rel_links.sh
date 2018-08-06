#!/bin/bash

home_dir=$(pwd)
rebuild_poly_links()
{
    poly_dir="$1/constant/polyMesh"
    if [ -d "$poly_dir" ]; then
        rm -r "$poly_dir"
        ln -s ../../simpleFoam/constant/polyMesh "$poly_dir"
    fi
}

removelink()
{
    if [ -L "$1" ]; then
        cp "$1" TEMPFILE
        rm "$1"
        mv TEMPFILE "$1"
    else
        echo "ERROR: ${1} is not a link." >&2
        return 1
    fi
    #cp --remove-destination "$(readlink "$1")" "$1" || echo "ERROR: ${1} is not a link." >&2
}

relativelink()
{
    # relativelink TARGET LOCATION
    if [ ! -f "$2" ]; then
        echo "ERROR: Target ${2} not found."
        return 1
    fi
    if [ -L "$1" ]; then
      rm $1
      ln -s $2 $1
    fi
}

for dir in $@; do
    casename=$( basename $dir )
    cd "$dir"
    # First tidy up stl links, copy to base, then link in simpleFoam
    removelink "${casename}.stl"
    relativelink "../../../${casename.stl}" "simpleFoam/constant/triSurface/${casename}.stl"

    # Now tidy up terrainDict links
    removelink terrainDict
    relativelink "../../terrainDict" "reGrid/system/terrainDict"
    relativelink "../../terrainDict" "simpleFoam/system/terrainDict"

    # In W1 we want

    # Now replace all the polyMesh links to link to simpleFoam mesh
    for wd in "./W[0-9]*"; do
        rebuild_poly_links $wd
    done

    cd "$home_dir"
done
