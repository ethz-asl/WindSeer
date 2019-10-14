#!/bin/bash

# Set up some stuff for getopts
OPTIND=1
home_dir=$(pwd)
wind_delta=1
csv_dir="/intel_share/data/cfd_3d_csv"
python_dir="/home/nick/src/intel_wind/openfoam_batch/python"
csv_overwrite=0
unconverged_cases=0
verbose=0

usage() {
    echo -e "Usage: $0 [OPTIONS] DIR0 DIR1 ..."
    echo -e "  DIR0 DIR1 ... Case directories"
    echo -e "  -c CSV_DIR\n\tOutput csv directory"
    echo -e "  -p PYTHON_DIR\n\tLocation of resample.py"
    echo -e "  -w delta_wind\n\tWind step size \(w=1:delta_wind:15\)"
    echo -e "  -o\n\tOverwrite existing (non-empty) csv files"
    echo -e "  -v\n\tVerbose output"
    echo -e "  -u only UNCONVERGED cases"
    echo -e "  -h\n\tPrint this help and exit"
}

source shared_functions.sh

while getopts "c:p:w:iouvh" opt; do
    case "$opt" in
        c)  csv_dir=$OPTARG ;;
        p)  python_dir=$OPTARG ;;
        w)  wind_delta=$OPTARG ;;
        o)  csv_overwrite=1 ;;
        v)  verbose=1 ;;
        u)  echo "ONLY creating unconverged cases"
            unconverged_cases=1 ;;
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

python_dir=$( check_path "$python_dir" "$home_dir" ) || exit 1
csv_dir=$( check_path "$csv_dir" "$home_dir" ) || exit 1


for dir in $@; do
    casename=$(basename "$dir")
    base_dir=$( check_path "$dir" "$home_dir" ) || continue
    reGrid_dir="$base_dir/reGrid"
    cd $base_dir
    for (( w=1; w<=15; w+=$wind_delta )); do
        wind_directory="$base_dir/W$w"
        # if no wind directory, go to next
        [ ! -d "$wind_directory" ] && continue
        
        # IF simpleFoam2.err is present then we skip if: 
        #   1 We're creating converged solutions, and 2.err not empty
        #   2 We're creating unconverged solutions, and 2.err is empty
        # THEN: go to next
        sf2err="${wind_directory}/simpleFoam2.err"
        
        if [ -f $sf2err ]; then
            if [ "$unconverged_cases" -eq 0 ] && [ -s $sf2err ]; then
                [ "$verbose" -ne 0 ] &&
                    echo "  ${casename}/W${w}/simpleFoam2.err not empty, skipping"
                continue
            fi
            if [ "$unconverged_cases" -ne 0 ] && [ ! -s $sf2err ]; then
                [ "$verbose" -ne 0 ] &&
                    echo "  ${casename}/W${w}/simpleFoam2.err is empty, skipping"
                continue
            fi
        fi

        # if simpleFoam2.err isn't here, and .err is not there or not empty, go to next
        if [ ! -f $sf2err ]; then
            if [ ! -f "$wind_directory/simpleFoam.err" ] ||
                [ -s "$wind_directory/simpleFoam.err" ]; then
                [ "$verbose" -ne 0 ] &&
                    echo "  ${casename}/W${w}/simpleFoam.err not found or not empty, skipping"
                continue
            fi
        fi

        cd "$wind_directory"
        touch hill.foam
        solution_converged=$( check_converged )
        if [ "$solution_converged" -eq 0 ]; then
            [ "$verbose" -ne 0 ] &&
                echo "  ${casename}/W${w} did not converge in max iterations, skipping."
            continue
        fi

        latest_time=$( foamListTimes -latestTime )
        printf -v csv_file "$csv_dir/%s_W%02d" $casename $w
        printf -v full_csv "${csv_file}_t%04d0.csv" $latest_time
        if [ "$csv_overwrite" -eq 0 ] && [ -s "${full_csv}" ]; then
            [ "$verbose" -ne 0 ] &&
                echo "  ${full_csv} already exists and is not empty, skipping."
            continue
        fi

        [ "$verbose" -ne 0 ] &&
            echo "  Creating csv for t=$latest_time to $csv_file..."
        python "${python_dir}/resample.py" --three-d --case-dir $wind_directory \
            --mesh-dir $reGrid_dir --case-foam hill.foam --outfile $csv_file --time $latest_time
        if [ "$?" -gt 0 ]; then
            [ "$verbose" -ne 0 ] && echo " failed!"
        else
            [ "$verbose" -ne 0 ] && echo " done."
        fi
    done
    cd "$home_dir"
done
