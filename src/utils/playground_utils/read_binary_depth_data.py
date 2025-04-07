import numpy as np

# Specify the file path
file_path = '/home/abdullah/utn/phantom-human2robot/playground/data/depth_only_output/_Depth_1742901201340.73510742187500.bin'

# Define the dimensions of your depth matrix
# You need to know these in advance or infer them from the file size
height = 480  # Example height
width = 848   # Example width

# Define the data type of the depth values
# Common types for depth data are np.float32 or np.uint16
data_type = np.float32

# Read the binary file
depth_matrix = np.fromfile(file_path, dtype=data_type)

# Reshape the flat array into a 2D matrix
depth_matrix = depth_matrix.reshape((height, width))

print(f"Depth matrix shape: {depth_matrix.shape}")
print(f"Min depth: {depth_matrix.min()}, Max depth: {depth_matrix.max()}")
print(f"Mean depth: {depth_matrix.mean()}, Std depth: {depth_matrix.std()}")
