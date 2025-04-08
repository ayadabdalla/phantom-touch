# Specify the file paths
import os

file_paths = []
extension = ".ply"
file_specific_dir = "sam_segmented_hands"
# get the list of files that ends with .bin in some directory
directory =f"/home/abdullah/utn/phantom-touch/assets/data/recordings/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/{file_specific_dir}"
for filename in os.listdir(directory):
    if filename.endswith(extension):
        file_paths.append(os.path.join(directory, filename))
print(len(file_paths))