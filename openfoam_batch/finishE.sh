home_dir=$(pwd)
w=2
wind_delta=1
csv_dir="$home_dir/csv"
case_dir=$1
casename=$(basename "$case_dir")
python_directory="$home_dir/python"
reGrid_dir="../reGrid"

cd $home_dir
for (( w=$w; w<=15; w+=$wind_delta )); do
    ./reuse_wind_case -c "${case_dir}/W1" -v 1 -o "$case_dir" $w
    if [ "$?" -eq 0 ]
    then
        wind_directory="${case_dir}/W${w}"
        cd $wind_directory
        touch hill.foam
        latest_time=$( foamListTimes -latestTime )
        printf -v csv_file "$csv_dir/%s_W%02d" $casename $w
        echo -en "\tCreating csv for t=$latest_time to $csv_file..."
        python "${python_directory}/resample.py" --three-d --case-dir $wind_directory \
            --mesh-dir $reGrid_dir --case-foam hill.foam --outfile $csv_file --time $latest_time
        if [ "$?" -gt 0 ]; then
            echo " failed!"
        else
            echo " done."
        fi
    fi
    cd $home_dir
done
