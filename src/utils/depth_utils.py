import sys
import numpy as np
import os
def load_raw_depth_images(raw_depth_directory_path, shape=(632, 1000)):
    # list all the files in the directory and sort them
    directory = sorted([f for f in os.listdir(raw_depth_directory_path) if f.endswith('.npy')])
    numpy_depth = []
    for i,np_file in enumerate(directory):
        numpy_depth.append(np.load(f"{raw_depth_directory_path}/{np_file}"))
        if numpy_depth[-1].shape == shape:
            continue
        else:
            print(f"Depth image shape is not correct: {numpy_depth[-1].shape} and data index is {i}")
            # remove erroneous depth image from the array and display its name
            print(f"Removing depth image: {np_file}")
            numpy_depth.pop()
    numpy_depth = np.array(numpy_depth)
    print(f"Number of files: {numpy_depth.shape[0]}")
    return numpy_depth