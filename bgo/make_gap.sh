#!/bin/bash

# For each directory, check if there is a subdirectory named {directory}_gap.
# If there is, run make clean && make bench
for dir in ./*; do
    echo "Checking $dir"
    if [ -d "$dir" ]; then
        if [ -d "$dir/${dir}_gap" ]; then
            echo "Running make clean && make bench in $dir/${dir}_gap"
            make -C "$dir/${dir}_gap" clean
            make -C "$dir/${dir}_gap" bench
        fi
    fi
done