#!/bin/bash

# Define the episode ranges manually (parsed from your yaml)
declare -A first_frames
first_frames=(
  [e0]=100
  [e1]=340
  [e2]=550
  [e3]=765
  [e4]=890
  [e5]=1130
  [e6]=1305
  [e7]=1410
  [e8]=1518
  [e9]=1635
  [e10]=1770
  [e11]=1910
  [e12]=2038
  [e13]=2265
  [e14]=2425
  [e15]=2600
)

last_frames=(
  [e0]=240
  [e1]=440
  [e2]=695
  [e3]=840
  [e4]=1030
  [e5]=1215
  [e6]=1362
  [e7]=1470
  [e8]=1580
  [e9]=1720
  [e10]=1855
  [e11]=2000
  [e12]=2132
  [e13]=2378
  [e14]=2525
  [e15]=2660
)
# Set your source folder (where the files are now)
source_folder="./"

# Set your destination root folder
destination_root="./episodes"

# Create destination root if it doesn't exist
mkdir -p "$destination_root"

# Only create directories and process files for episodes e10 and above
for episode in "${!first_frames[@]}"; do
    episode_number=${episode#e}
    if (( episode_number >= 0 )); then
        mkdir -p "$destination_root/$episode"
    fi
done

# Loop over all files in the source folder
for file in "$source_folder"/*.{png,raw}; do
  # Skip if no files matched
  [ -e "$file" ] || continue

  # Extract frame number: get the part after the last underscore
  filename=$(basename "$file")
  frame_part=${filename##*_}
  frame_number=${frame_part%%.*}
  frame_number=$((10#$frame_number))  # Convert to decimal (safe)

  # Find which episode it belongs to
  for episode in "${!first_frames[@]}"; do
    episode_number=${episode#e}

    first=${first_frames[$episode]}
    last=${last_frames[$episode]}

    if (( frame_number >= first && frame_number <= last )); then
      mv "$file" "$destination_root/$episode/"
      break
    fi
  done

done

echo "Sorting complete for episodes"
