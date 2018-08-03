 #!/bin/bash
for dir in ./S12x12_*; do
    cd $dir
    for wd in ./W1[0-9]; do
        cd $wd/constant
        rm -r polyMesh
        ln -s ../../W1/constant/polyMesh polyMesh
        cd ../..
    done
    for wd in ./W[2-9]; do
        cd $wd/constant
        rm -r polyMesh
        ln -s ../../W1/constant/polyMesh polyMesh
        cd ../..
    done
    cd ..
done
