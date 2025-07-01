#!/bin/bash

# Outer loop for source class
for (( i=0; i<5; i++ )); do
    # Inner loop for target class
    for (( j=0; j<5; j++ )); do
        # Skip when j is equal to i
        if [ "$j" -eq "$i" ]; then
            continue
        fi
        echo "$i to $j"
        python generate_images.py --split val --source $i --target $j --kind reference
    done
done
