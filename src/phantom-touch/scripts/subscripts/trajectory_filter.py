from matplotlib import cm, pyplot as plt
import numpy as np

trajectories = [
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e0/hand_keypoints_e0_right.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e1/hand_keypoints_e1_right.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e2/hand_keypoints_e2_right.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e3/hand_keypoints_e3_right.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e4/hand_keypoints_e4_right.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e5/hand_keypoints_e5_right.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e6/hand_keypoints_e6_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e7/hand_keypoints_e7_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e8/hand_keypoints_e8_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e9/hand_keypoints_e9_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e10/hand_keypoints_e10_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e11/hand_keypoints_e11_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e12/hand_keypoints_e12_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e13/hand_keypoints_e13_right.npz",
                "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e14/hand_keypoints_e14_right.npz",]

# def filter_trajectories(trajectories):
#     # do exponential average on only the z axis
#     filtered_trajectories = []
#     for trajectory in trajectories:
#         data = np.load(trajectory)
#         positions = data['positions']
#         filtered_z = np.zeros_like(positions[:, 2])
#         # expect the next 2 steps and compare to the actual and if it's too far ignore it
#         indeces = []
#         for i in range(0, len(positions)-2):
#             filtered_z[i] = 0.8 * positions[i + 2,2] + 0.2 * positions[i+1, 2]
#             # Check if the current z is too far from the expected z
#             if abs(positions[i, 2] - filtered_z[i]) > 0.1:
#                 # store index of the filtered z
#                 indeces.append(i)
#         positions = np.delete(positions, indeces, axis=0)
#         print(f"Filtered out: {len(indeces)}")
#         filtered_trajectories.append(positions)
        
#     return filtered_trajectories


# # Create a color map based on the order
# def plot_trajectory(positions):
#     num_points = positions.shape[0]
#     colors = cm.magma(np.linspace(0, 1, num_points))

#     # Plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for i in range(num_points - 1):
#         ax.plot(
#             positions[i:i+2, 0],
#             positions[i:i+2, 1],
#             positions[i:i+2, 2],
#             color=colors[i]
#         )

#     ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('3D Trajectory Colored by Order')
#     # set the limits of the axes
#     ax.set_zlim([0.5, 1.5])
#     # ax.set_xlim([-0.5, 0.1])
#     ax.set_ylim([-0.5, 0.5])

#     plt.show()
# if __name__ == "__main__":
#     filtered_trajectories = filter_trajectories(trajectories)
#     for i, filtered_trajectory in enumerate(filtered_trajectories):
#         print(f"Filtered trajectory {i}: {filtered_trajectory.shape}")
#         # plot the filtered trajectory in 3d
#         plot_trajectory(filtered_trajectory)


# def filter_trajectories(trajectories):
#     filtered_trajectories = []
#     for trajectory in trajectories:
#         data = np.load(trajectory)
#         positions = data['positions']      # shape: (N, 3)
#         normals = data['normals']          # shape: (N, 3)

#         filtered_z = np.zeros_like(positions[:, 2])
#         indeces = []
#         for i in range(0, len(positions) - 2):
#             filtered_z[i] = 0.8 * positions[i + 2, 2] + 0.2 * positions[i + 1, 2]
#             if abs(positions[i, 2] - filtered_z[i]) > 0.1:
#                 indeces.append(i)

#         positions = np.delete(positions, indeces, axis=0)
#         normals = np.delete(normals, indeces, axis=0)
#         print(positions.shape)
#         print(normals.shape)
#         print(f"Filtered out: {len(indeces)}")
#         filtered_trajectories.append((positions, normals))

#     return filtered_trajectories

def filter_trajectories(trajectories):
    # do exponential average on only the z axis
    filtered_positions = []
    filtered_normals = []
    filtered_thumbs = []
    for trajectory in trajectories:
        data = np.load(trajectory)
        positions = data["positions"]
        normals = data["normals"]
        thumbs = data["thumb_vectors"]
        filtered_z = np.zeros_like(positions[:, 2])
        # expect the next 2 steps and compare to the actual and if it's too far ignore it
        indeces = []
        for i in range(0, len(positions) - 2):
            filtered_z[i] = 0.8 * positions[i + 2, 2] + 0.2 * positions[i + 1, 2]
            # Check if the current z is too far from the expected z
            if abs(positions[i, 2] - filtered_z[i]) > 0.1:
                # store index of the filtered z
                indeces.append(i)
        positions = np.delete(positions, indeces, axis=0)
        normals = np.delete(normals, indeces, axis=0)
        thumbs = np.delete(thumbs, indeces, axis=0)
        print(f"Filtered out: {len(indeces)}")
        filtered_positions.append(positions)
        filtered_thumbs.append(thumbs)
        filtered_normals.append(normals)
    return filtered_positions, filtered_normals, filtered_thumbs

def plot_trajectory_with_normals(positions, normals):
    num_points = positions.shape[0]
    colors = cm.magma(np.linspace(0, 1, num_points))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory lines
    for i in range(num_points - 1):
        ax.plot(
            positions[i:i+2, 0],
            positions[i:i+2, 1],
            positions[i:i+2, 2],
            color=colors[i]
        )

    # Scatter points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, marker='o')

    # Plot normal vectors as arrows
    for i in range(0, num_points):  # reduce clutter by plotting every 5th
        p = positions[i]
        n = normals[i]
        ax.quiver(
            p[0], p[1], p[2],
            n[0], n[1], n[2],
            length=0.05,
            color='cyan',
            normalize=True
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory with Normal Vectors')
    ax.set_zlim([0.5, 1.5])
    #invert the z axis
    ax.invert_zaxis()
    ax.set_ylim([-0.5, 0.5])
    plt.show()

if __name__ == "__main__":
    positions,normals,_ = filter_trajectories(trajectories)
    for i, (position_e, normal_e) in enumerate(zip(positions,normals)):
        print(f"Trajectory {i}: {position_e.shape}")
        plot_trajectory_with_normals(position_e, normal_e)
