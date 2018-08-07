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
    if [ -L "$2" ]; then
      rm $2
      ln -s $1 $2
    fi
}

for dir in $@; do
    casename=$( basename $dir )
    cd "$dir"
    # First tidy up stl links, copy to base, then link in simpleFoam
    removelink "${casename}.stl"
    relativelink "../../../${casename}.stl" "simpleFoam/constant/triSurface/${casename}.stl"

    # Now tidy up terrainDict links
    removelink terrainDict
    relativelink "../../terrainDict" "reGrid/system/terrainDict"
    relativelink "../../terrainDict" "simpleFoam/system/terrainDict"

    # Replace links in constant and system dirs in reGrid
    removelink "reGrid/system/blockMeshDict"
    removelink "reGrid/system/controlDict"
    removelink "reGrid/system/fvSchemes"
    removelink "reGrid/system/fvSolution"

    # Replace links in simpleFoam
    removelink "simpleFoam/system/blockMeshDict"
    removelink "simpleFoam/system/controlDict"
    removelink "simpleFoam/system/fvSchemes"
    removelink "simpleFoam/system/fvSolution"

    removelink "simpleFoam/constant/transportProperties"
    removelink "simpleFoam/constant/turbulenceProperties"


    # Replace links in W1
    removelink "W1/system/blockMeshDict"
    removelink "W1/system/controlDict"
    removelink "W1/system/fvSchemes"
    removelink "W1/system/fvSolution"

    removelink "W1/constant/transportProperties"
    removelink "W1/constant/turbulenceProperties"

    # Now replace all the polyMesh links to link to simpleFoam mesh
    for fd in ${dir}/W[0-9]*; do
        [ ! -d "fd" ] && continue
        wd=$( basename "$fd" )
        rebuild_poly_links $wd
        if [ "$wd" != 'W1' ]; then
            relativelink "../../W1/system/fvSchemes" "${wd}/system/fvSchemes"
            relativelink "../../W1/system/fvSolution" "${wd}/system/fvSolution"
            relativelink "../../W1/constant/transportProperties" "${wd}/constant/transportProperties"
            relativelink "../../W1/constant/turbulenceProperties" "${wd}/constant/turbulenceProperties"
        fi
    done

    cd "$home_dir"
done
