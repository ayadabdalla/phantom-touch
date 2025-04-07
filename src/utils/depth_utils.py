import numpy as np
import os
def load_raw_depth_images(raw_depth_directory_path):
    # list all the files in the directory and sort them
    directory = sorted([f for f in os.listdir(raw_depth_directory_path) if f.endswith('.npy')])
    numpy_depth = []
    for np_file in directory:
        numpy_depth.append(np.load(f"{raw_depth_directory_path}/{np_file}"))
    numpy_depth = np.array(numpy_depth)
    print(f"Number of files: {len(numpy_depth)}")
    return numpy_depth