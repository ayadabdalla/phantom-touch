#!/bin/bash

# Source and destination directories
src_dir="/mnt/dataset_drive/ayad/phantom-touch/data/recordings/test_exp_streaming"
dst_dir="/mnt/dataset_drive/ayad/phantom-touch/data/recordings/test_exp_streaming/e00000"

# Create destination directory if it doesn't exist
mkdir -p "$dst_dir"

echo "Copying files with 'e00000' in the filename from $src_dir to $dst_dir..."

# Counter to track progress
count=0

# Loop through files that match "*x*" in the name
for file in "$src_dir"/*e00000*; do
    if [[ -f "$file" ]]; then
        cp "$file" "$dst_dir/"
        echo "Copied: $(basename "$file")"
        ((count++))
    fi
done

echo "Done. $count file(s) copied."
