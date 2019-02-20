#!/bin/bash
home_dir=$(pwd)
csv_dir='/cluster/scratch/lawrancn/csv'

find csv -empty -name "E12x12_${1}" | while read line; do
        csv_file=$( basename "$line" | cut -c1-13 )
        case=$( echo "$csv_file" | cut -c1-9 )
        wind=$( echo "$csv_file" | cut -c12-13 | sed 's/^0*//' )
        case_dir="/cluster/scratch/lawrancn/E64_cases/${case}"
        wind_dir="${case_dir}/W${wind}"
        reGrid_dir="${case_dir}/reGrid"

        cd "$wind_dir"
        latest_time=$( foamListTimes -latestTime )
        csv_out="${csv_dir}/${csv_file}"

        python "${home_dir}/python/resample.py" --three-d --case-dir $wind_dir --mesh-dir $reGrid_dir --case-foam hill.foam --outfile $csv_out --time $latest_time

        cd $home_dir
done
