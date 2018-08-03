#!/bin/bash

# Set up some stuff for getopts
OPTIND=1
home_dir=$(pwd)

csv_dir="/intel_share/data/cfd_3d_csv"
python_directory="/home/nick/src/intel_wind/openfoam_batch/python"

usage() {
    echo -e "Usage: $0 [OPTIONS] DIR0 DIR1 ..."
    echo -e "  DIR0 DIR1 ... Case directories"
    echo -e "  -c CSV_DIR\n\tOutput csv directory"
    echo -e "  -p PYTHON_DIR\n\tLocation of resample.py"
    echo -e "  -h\n\tPrint this help and exit"
}

while getopts "c:v:o:b:h" opt; do
    case "$opt" in
        c)  csv_dir=$OPTARG ;;
        p)  python_directory=$OPTARG ;;
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


for dir in $@; do
    casename=$(basename "$dir")
    base_dir="$home_dir/$dir"
    reGrid_dir="$base_dir/reGrid"
    cd $base_dir
    for (( w=1; w<=15; w+=1 )); do
        wind_directory="$base_dir/W$w"
        if [ -d "$wind_directory" ]; then
            if [ -f "$wind_directory/simpleFoam.err" ] && [
        cd "$wind_directory"
        touch hill.foam
        latest_time=$( foamListTimes -latestTime )
        printf -v csv_file "$csv_dir/%s_W%02d" $casename $w
        echo -e "\tCreating csv for t=$latest_time to $csv_file..."
        python "${python_directory}/resample.py" --three-d --case-dir $wind_directory \
            --mesh-dir $reGrid_dir --case-foam hill.foam --outfile $csv_file --time $latest_time
        if [ "$?" -gt 0 ]; then
            echo " failed!"
        else
            echo " done."
        fi
    done
    cd "$home_dir"
done
