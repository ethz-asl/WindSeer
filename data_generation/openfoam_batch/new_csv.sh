#!/bin/bash

# Set up some stuff for getopts
OPTIND=1
home_dir=$(pwd)
version=2

usage() {
    echo -e "Usage: $0 [OPTIONS] DIR0 DIR1 ..."
    echo -e "  DIR0 DIR1 ... Case directories"
    echo -e "  -v [2, 3]\n\tCSV version (2 default)"
    echo -e "  -h\n\tPrint this help and exit"
}

while getopts "v:h" opt; do
    case "$opt" in
        v)  version=$OPTARG ;;
        h|*)  usage
            exit 0
            ;;
    esac
done
shift $(expr $OPTIND - 1 )

if [ "$#" -lt 1 ]; then
    echo "ERROR: No case directories specified" >&2
    usage >&2
    exit 1
fi

if [ "$version" -eq 2 ]; then
    tminy="200.0"
    tmaxy="1300.0"
    tmaxz="1100.0"
    tsx="86"
    tsy="63"
    tsz="95"
elif [ "$version" -eq 3 ]; then
    tminy="0.0"
    tmaxy="1498.0"
    tmaxz="1100.0"
    tsx="90"
    tsy="90"
    tsz="95"
elif [ "$version" -eq 4 ]; then
    tminy="0.0"
    tmaxy="1498.0"
    tmaxz="1100.0"
    tsx="63"
    tsy="63"
    tsz="63"
else
    echo "ERROR: Version number not recognised" >&2
    exit 1
fi

echo -n "Generating terrainDict for V${version}, y=${tminy}:${tmaxy}, zmax=${tmaxz}"
echo " nx ny nz : $tsx $tsy $tsz"

for indir in "$@"
do
    tdict="${indir}/reGrid/system/terrainDict"
    sed -i '' "$tdict"

    echo "$tdict"

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
