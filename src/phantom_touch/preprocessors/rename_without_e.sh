#!/bin/bash

# Directory to process
DIR="."  # Current directory

# Loop over all files in the directory
for file in "$DIR"/*; do
  # Make sure it's a regular file
  if [[ -f "$file" ]]; then
    # Extract the filename without the directory
    filename=$(basename "$file")
    
    # Remove the pattern _enumbers
    # This regex captures the pattern _enumbers and removes it
    newfilename=$(echo "$filename" | sed -E 's/_e[0-9]+//g')


    # Rename the file if the name changed
    if [[ "$filename" != "$newfilename" ]]; then
      mv "$DIR/$filename" "$DIR/$newfilename"
      echo "Renamed: $filename -> $newfilename"
    fi
  fi
done
