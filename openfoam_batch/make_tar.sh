#!/bin/bash
for of_case in $@; do
    casename=$( basename "$of_case" )
    casedir=$( dirname "$of_case" )
    tar -czvf "${casedir}/${casename}.tar.gz" "$of_case"
    if [ "$?" == 0 ]; then
        echo "Removing ${of_case}"
        rm -r "$of_case"
    fi
done

