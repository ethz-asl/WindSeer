#!/bin/bash

homedir=$(pwd)

for indir in "$@"
do
    tdict="${indir}/reGrid/system/terrainDict"
    sed -i '' "$tdict"

    echo "$tdict"
    tminy="200.0"
    tmaxy="1300.0"
    tmaxz="1100.0"

    tsx="86"
    tsy="63"
    tsz="95"

    foamDictionary -entry 'TERRAIN_DICT:MINY' -set $tminy $tdict
    foamDictionary -entry 'TERRAIN_DICT:MAXY' -set $tmaxy $tdict
    foamDictionary -entry 'TERRAIN_DICT:MAXZ' -set $tmaxz $tdict

    foamDictionary -entry 'TERRAIN_DICT:SUBGRADE:X:N' -set $tsx $tdict
    foamDictionary -entry 'TERRAIN_DICT:SUBGRADE:Y:N' -set $tsy $tdict
    foamDictionary -entry 'TERRAIN_DICT:SUBGRADE:Z:N' -set $tsz $tdict

    reGrid_dir="${indir}/reGrid"
    echo -en "\tBuilding resampled (regular) mesh for final output..."
    cd $reGrid_dir
    rm -f -r constant/
    rm -f -r dynamicCode/

    blockMesh > blockMesh.log 2> blockMesh.err

    if [ $? -ne 0 ]
    then
        echo " failed. Error report:"
        cat blockMesh.err
        echo -e "\tMoving to next case."
        continue
    fi
    echo " done."
    touch testgrid.foam
    cd "$homedir"
done
