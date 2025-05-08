# for i in {0..111}; do
#     src="/mnt/dataset_drive/ayad/phantom-touch/data/recordings/handover_collection_0/episodes/e7/Color_1920x1080_00000.png"
#     dest="/mnt/dataset_drive/ayad/phantom-touch/data/recordings/handover_collection_0/episodes/e$i/Color_1920x1080_00000.png"
#     if [ ! -f "$dest" ]; then
#         mkdir -p "$(dirname "$dest")"
#         cp "$src" "$dest"
#     fi
# done

# remove the file you added from all the other folders again
# for i in {0..111}; do
#     dest="/mnt/dataset_drive/ayad/phantom-touch/data/recordings/handover_collection_0/episodes/e$i/Color_1920x1080_00000.png"
#     if [ -f "$dest" ]; then
#         rm "$dest"
#     fi
# done

# remove the mask generated for the file you added from all the other folders again
for i in {0..111}; do
    dest="/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/sam2-vid_output/episodes/e$i/Mask_1920x1080_00000.png"
    if [ -f "$dest" ]; then
        rm "$dest"
    fi
done