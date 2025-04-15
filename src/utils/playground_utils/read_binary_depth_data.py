import numpy as np

# Specify the file path
file_path = '/mnt/dataset_drive/ayad/phantom-touch/data/recordings/handover_experiment_1/RawDepth_8682390ms_e00017_00068.raw'

# Define the dimensions of your depth matrix
# You need to know these in advance or infer them from the file size
height = 632  # Example height
width = 1000   # Example width

# Define the data type of the depth values
# Common types for depth data are np.float32 or np.uint16
data_type = np.uint16

# Read the binary file
depth_matrix = np.fromfile(file_path, dtype=data_type)

# Reshape the flat array into a 2D matrix
depth_matrix = depth_matrix.reshape((height, width))

print(f"Depth matrix shape: {depth_matrix.shape}")
print(f"Min depth: {depth_matrix.min()}, Max depth: {depth_matrix.max()}")
print(f"Mean depth: {depth_matrix.mean()}, Std depth: {depth_matrix.std()}")
