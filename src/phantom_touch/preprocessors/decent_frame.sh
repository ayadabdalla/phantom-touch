# for i in {0..11}; do
#     src="/home/epon04yc/pick_and_place_phantom/episodes/e6/Color_1920x1080_3199724906ms_00669"
#     dest="/home/epon04yc/pick_and_place_phantom/episodes/e$i/Color_1920x1080_00000.png"
#     if [ ! -f "$dest" ]; then
#         mkdir -p "$(dirname "$dest")"
#         cp "$src" "$dest"
#     fi
# done

# remove the file you added from all the other folders again
# for i in {0..11}; do
#     dest="/home/epon04yc/pick_and_place_phantom/episodes/e$i/Color_1920x1080_00000.png"
#     if [ -f "$dest" ]; then
#         rm "$dest"
#     fi
# done

# remove the mask generated for the file you added from all the other folders again
for i in {0..11}; do
    dest="/home/epon04yc/pick_and_place_phantom/episodes/e$i/Mask_1920x1080_00000.png"
    if [ -f "$dest" ]; then
        rm "$dest"
    fi
done