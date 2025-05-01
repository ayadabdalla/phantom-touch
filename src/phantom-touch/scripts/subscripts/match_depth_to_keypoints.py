import sys
import cv2
import numpy as np
from tqdm import tqdm
from utils.data_utils import load_keypoints_grouped_by_frame, load_pcds
from utils.depth_utils import load_raw_depth_images
from utils.hw_camera import fx, fy, cx, cy
import open3d as o3d
from utils.pointcloud_utils import print_stats
from omegaconf import OmegaConf
import os
import glob
from utils.rgb_utils import load_rgb_images
from scipy import linalg


def interpolate_depth_image(depth_image, invalid_val=0.25):
    """
    Fill missing (invalid) values in a single depth image.
    """
    # apply bilateral filter from cv2
    # convert to 32bit float
    depth_image = depth_image.astype(np.float32)
    depth_image_filled = cv2.bilateralFilter(
        depth_image, d=5, sigmaColor=75, sigmaSpace=75
    )
    # convert back to 16bit
    depth_image_filled = depth_image_filled.astype(np.uint16)
    return depth_image_filled


def interpolate_depth_list(depth_images, invalid_val=0):
    """
    Interpolate a list of depth images and return a single stacked ndarray.
    Output shape: (N, H, W)
    """
    interpolated_list = [
        interpolate_depth_image(img, invalid_val)
        for img in tqdm(depth_images, desc="Interpolating Depth Images")
    ]
    interpolated_array = np.stack(
        interpolated_list, axis=0
    )  # Stack along a new first dimension
    return interpolated_array


def calculate_target_position_and_orientation(hand_keypoints):
    """
    Calculate the target position as the midpoint between thumb and index fingertips,
    and the target orientation by fitting a plane through thumb and index finger points.

    Parameters:
    -----------
    hand_keypoints : numpy.ndarray
        Array of shape (21, 3) containing 3D coordinates of 21 hand keypoints
        Keypoints are organized in the following order:
        0: Wrist
        1-4: Thumb (from base to tip)
        5-8: Index finger (from base to tip)
        9-12: Middle finger (from base to tip)
        13-16: Ring finger (from base to tip)
        17-20: Pinky finger (from base to tip)

    Returns:
    --------
    tuple: (target_position, normal_vector, plane_points)
        target_position: numpy.ndarray of shape (3,) - midpoint between thumb and index tips
        normal_vector: numpy.ndarray of shape (3,) - unit normal vector of the fitted plane
        plane_points: numpy.ndarray of shape (8, 3) - points used for plane fitting
    """
    # Validate input
    if not isinstance(hand_keypoints, np.ndarray):
        hand_keypoints = np.array(hand_keypoints)

    if hand_keypoints.shape != (21, 3):
        raise ValueError(
            f"Expected hand keypoints of shape (21, 3), got {hand_keypoints.shape}"
        )

    # Extract index finger and thumb keypoints
    index_finger = hand_keypoints[5:9]  # indices 5-8
    thumb = hand_keypoints[1:5]  # indices 1-4

    # 1. Calculate target position (pt) - midpoint between thumb tip and index tip
    thumb_tip = thumb[3]  # The 4th point of thumb (index 4 in original array)
    index_tip = index_finger[
        3
    ]  # The 4th point of index finger (index 8 in original array)
    target_position = (thumb_tip + index_tip) / 2

    # 2. Calculate target orientation by fitting a plane through thumb and index finger points
    # Combine all points from thumb and index finger
    plane_points = np.vstack((thumb, index_finger))

    # Fit a plane to the points using Singular Value Decomposition (SVD)
    centroid = np.mean(plane_points, axis=0)
    _, _, vh = np.linalg.svd(plane_points - centroid)
        # Center the points
    centered_points = plane_points - centroid

    # Singular Value Decomposition
    u, s, vh = linalg.svd(centered_points)

    # The normal vector to the plane is the last singular vector
    normal_vector = vh[2, :]

    # Normalize the vector to unit length
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # ensure the normal vector points in the negative z direction
    if normal_vector[2] > 0:
        normal_vector = -normal_vector
    # the principal axis is the thumb, fit a line through the first 4 points using pca
    thumb_axis = thumb[3] - thumb[0]
    thumb_axis = thumb_axis / np.linalg.norm(thumb_axis)

    return target_position, normal_vector, plane_points, thumb_axis


# load config
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = OmegaConf.load(f"{parent_directory}/../conf/3d_projection.yaml")

# define paths
trajectory_root = config.trajectory_directory
recordings_root = config.recordings_directory
sam2hand_root = config.sam2hand_directory
hamer_root = config.hamer_output_directory
vitpose_root = config.vitpose_output_directory

# === Load Data ===
# Load color images and their paths
numpy_color, color_paths = load_rgb_images(
    recordings_root, prefix="Color_", return_path=True
)
print(f"Color shape: {numpy_color.shape}")

# Load depth images
numpy_depth = load_raw_depth_images(
    recordings_root, shape=[numpy_color.shape[1], numpy_color.shape[2]]
)
print(f"Depth shape: {numpy_depth.shape}")

# load vitpose keypoints
vitpose_keypoints2d, vitpose_keypoints2d_paths = load_keypoints_grouped_by_frame(
    vitpose_root, prefix="vitpose_", return_path=True
)
print(f"Keypoints shape: {len(vitpose_keypoints2d)}")

# load sam2 pcds
sam2_pcds, sam2_pcd_paths = load_pcds(sam2hand_root, prefix="Color_", return_path=True)

assert (
    numpy_depth.shape[0] == numpy_color.shape[0]
), "Number of frames in masks and color images do not match"
assert (
    numpy_depth.shape[0] == numpy_depth.shape[0]
), "Number of frames in masks and depth images do not match"

visualize_pcds = []
target_positions_per_episode = []
normal_vectors_per_episode = []
keypoints_per_episode = []
thumb_vector_per_episode = []
invalid_keypoints_per_episode = []

previous_episode = os.path.basename(os.path.dirname(color_paths[0]))
dynamic_episode_index = 0
for idx, (depth_map, color_frame, color_path, keypoints_per_frame, sam2_pcd) in tqdm(
    enumerate(
        zip(numpy_depth, numpy_color, color_paths, vitpose_keypoints2d, sam2_pcds)
    ),
    total=numpy_depth.shape[0],
    desc="Generating Point Clouds",
):
    # if episode string is not equal to the previous one, reinitialize the lists
    episode = os.path.basename(os.path.dirname(color_path))
    if episode != previous_episode:
        target_positions_per_episode = []
        normal_vectors_per_episode = []
        keypoints_per_episode = []
        thumb_vector_per_episode = []
        invalid_keypoints_per_episode = []
        visualize_pcds = []
        dynamic_episode_index = 0
    episode = os.path.basename(os.path.dirname(color_path))
    sam2_pcd = sam2_pcd[0]
    chamfer_distances = []
    target_positions_per_frame = []
    normal_vectors_per_frame = []
    thumb_vectors_per_frame = []
    pcds_per_frame = []
    for i, keypoints2d in enumerate(keypoints_per_frame):
        invalid_keypoint = False
        keypoints2d = keypoints2d[:, :2].astype(int)
        if (
            keypoints2d[:, 0].max() > depth_map.shape[1]
            or keypoints2d[:, 1].max() > depth_map.shape[0]
        ):
            print(
                f"Keypoints out of bounds for frame {color_path}, hand {i} skipping..."
            )
            invalid_keypoint = True
            continue

        # create depth keypoints map
        depth_keypoints_map = np.zeros((depth_map.shape[0], depth_map.shape[1]))
        depth_keypoints_map = depth_map
        depth_keypoints_map[keypoints2d[:, 1], keypoints2d[:, 0]] = depth_map[
            keypoints2d[:, 1], keypoints2d[:, 0]
        ]

        # create points and colors
        points = np.zeros((len(keypoints2d), 3))
        colors = np.zeros((len(keypoints2d), 3))
        for j, (x, y) in enumerate(keypoints2d):
            z = depth_map[y, x]
            if j == 4:
                if z <= 250 or z >=5000:
                    invalid_keypoint = True
            elif j == 8:
                if z <= 250 or z >=5000:
                    invalid_keypoint = True
            else:
                colors[j] = [1, 1, 1]
            points[j] = [x, y, z]

        # convert to camera coordinates
        points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx / 1000.0
        points[:, 1] = (points[:, 1] - cy) * points[:, 2] / fy / 1000.0
        points[:, 2] = points[:, 2] / 1000.0

        # create pcd for hand keypoints
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds_per_frame.append(pcd)

        # load sam2 segmented hand pcd
        sam2_pcd = sam2_pcd.voxel_down_sample(voxel_size=0.01)

        # compute chamfer distance between hand keypoints and sam2 pcd
        chamfer_distance = pcd.compute_point_cloud_distance(
            sam2_pcd
        )  # pcd is the target, sam2_pcd is the source
        chamfer_distance = np.asarray(chamfer_distance)
        chamfer_distance = np.mean(chamfer_distance)
        chamfer_distances.append(chamfer_distance)
        target_position, normal_vector, _, thumb_vector = calculate_target_position_and_orientation(
            points
        )
        target_positions_per_frame.append(target_position)
        normal_vectors_per_frame.append(normal_vector)
        thumb_vectors_per_frame.append(thumb_vector)
    if invalid_keypoint:
        invalid_keypoints_per_episode.append(dynamic_episode_index)
        continue
    dynamic_episode_index += 1
    # filter the pcd per frame based on the lowest chamfer distance

    min_chamfer_distance = min(chamfer_distances)
    min_chamfer_index = chamfer_distances.index(min_chamfer_distance)
    target_position = target_positions_per_frame[min_chamfer_index]
    target_positions_per_episode.append(target_position)
    normal_vector = normal_vectors_per_frame[min_chamfer_index]
    normal_vectors_per_episode.append(normal_vector)
    thumb_vector = thumb_vectors_per_frame[min_chamfer_index]
    thumb_vector = thumb_vector / np.linalg.norm(thumb_vector)
    thumb_vector_per_episode.append(thumb_vector)
    pcd = pcds_per_frame[min_chamfer_index]
    keypoints = np.asarray(pcd.points)
    keypoints_per_episode.append(keypoints)
    # add target position and normal to point cloud
    # pcd.points = o3d.utility.Vector3dVector(np.vstack((points, target_position)))
    pcd.points = o3d.utility.Vector3dVector(np.expand_dims(target_position, 0))
    # # colors = np.vstack((colors, [0, 1, 0]))
    colors = np.expand_dims(np.array([0, 1, 1]), axis=0)
    points = np.asarray(pcd.points)
    # # save point cloud
    save_name = f"hand_keypoints_{episode}_right.npz"
    print(len(target_positions_per_episode))
    os.makedirs(os.path.join(trajectory_root, episode), exist_ok=True)
    np.savez_compressed(
        os.path.join(trajectory_root, episode, save_name),
        positions=target_positions_per_episode,
        normals=normal_vectors_per_episode,
        keypoints=keypoints_per_episode,
        thumb_vectors=thumb_vector_per_episode,
        invalid_keypoints=invalid_keypoints_per_episode, #TODO: remove this and save only the valid frame numbers
    )
    # # normal_vector = normal_vector / np.linalg.norm(normal_vector) * 0.1
    # # pcd.points = o3d.utility.Vector3dVector(np.vstack((points, target_position + normal_vector)))
    # # colors = np.vstack((colors, [0, 1, 1]))

    pcd.colors = o3d.utility.Vector3dVector(colors)
    previous_episode = episode
    visualize_pcds.append(pcd)

# visualize the point clouds
sams = []
for pcd in sam2_pcds:
    pcd = pcd[0]
    # pcd = pcd.voxel_down_sample(voxel_size=0.09)
    # pcd.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.1)[0]
    sams.append(pcd)

# full_visualization_pcds = []

# for pcd, sam in zip(visualize_pcds, sams):
#     full_visualization_pcds.append(pcd)
#     full_visualization_pcds.append(sam)
# full_visualization_pcds.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

# # Simple visualization (no real alpha, but looks lighter)
# o3d.visualization.draw_geometries(full_visualization_pcds,
#                                   point_show_normal=False,
#                                   width=1024,
#                                   height=768,
#                                   window_name="Simple PointCloud Viewer")


# save merged point cloud
# o3d.io.write_point_cloud(os.path.join(traj_dir, f"hand_augmented_pcd_{idx}.ply"), pcd)

print(f"Processed frame {idx}")


print("Batch processing complete.")
