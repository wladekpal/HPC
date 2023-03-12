#! /bin/bash

rm -f results.csv

for ACC in 8 16 32
do
    for THREADS in 2 4 8 16 32 64
    do
        export OMP_NUM_THREADS=$THREADS
        ./dft $ACC $THREADS
    done
done