for i in {0..111}; do
    src="/mnt/dataset_drive/ayad/phantom-touch/data/recordings/handover_collection_0/episodes/e7/Color_1920x1080_00000.png"
    dest="/mnt/dataset_drive/ayad/phantom-touch/data/recordings/handover_collection_0/episodes/e$i/Color_1920x1080_00000.png"
    if [ ! -f "$dest" ]; then
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
    fi
done