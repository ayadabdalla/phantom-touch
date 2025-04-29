import numpy as np
import matplotlib.pyplot as plt

trajectory = np.load("/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e14/hand_keypoints_e14_right.npz")
positions = trajectory['positions']
from matplotlib import cm

# Create a color map based on the order
num_points = positions.shape[0]
colors = cm.magma(np.linspace(0, 1, num_points))

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(num_points - 1):
    ax.plot(
        positions[i:i+2, 0],
        positions[i:i+2, 1],
        positions[i:i+2, 2],
        color=colors[i]
    )

ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory Colored by Order')

plt.show()
