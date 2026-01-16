#!/bin/bash

# Directory to search (default = current directory, or pass as first arg)
DIR=${1:-.}

# Loop through files in the directory
for file in "$DIR"/*; do
    # Only process regular files
    if [[ -f "$file" ]]; then
        # Extract the line starting with "Best hyperparameters"
        line=$(grep -m 1 "^Best hyperparameters" "$file")

        if [[ -n "$line" ]]; then
            # Extract MAX_DEPTH using grep + sed
            max_depth=$(echo "$line" | grep -o "'MAX_DEPTH': [0-9]*" | sed "s/'MAX_DEPTH': //")

            # Print file and extracted depth
            echo "File: $file -> MAX_DEPTH = $max_depth"
        fi
    fi
done

