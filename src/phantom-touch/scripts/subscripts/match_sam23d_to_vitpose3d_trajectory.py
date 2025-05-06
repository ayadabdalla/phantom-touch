import sys
import cv2
import numpy as np
from tqdm import tqdm
from utils.data_utils import load_keypoints_grouped_by_frame, load_pcds
from utils.depth_utils import load_raw_depth_images
from utils.hw_camera import fx, fy, cx, cy
import open3d as o3d
from utils.phantomutils import calculate_target_position_and_orientation
from utils.pointcloud_utils import print_stats
from omegaconf import OmegaConf
import os
from utils.rgb_utils import load_rgb_images

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
# Load depth images
numpy_depth = load_raw_depth_images(
    recordings_root, shape=[numpy_color.shape[1], numpy_color.shape[2]]
)
# load vitpose keypoints
vitpose_keypoints2d, vitpose_keypoints2d_paths = load_keypoints_grouped_by_frame(
    vitpose_root, prefix="vitpose_", return_path=True
)
# load sam2 pcds
sam2_pcds, sam2_pcd_paths = load_pcds(sam2hand_root, prefix="Color_", return_path=True)
assert (
    numpy_depth.shape[0] == numpy_color.shape[0]
), "Number of frames in masks and color images do not match"
assert (
    numpy_depth.shape[0] == len(vitpose_keypoints2d)
), "Number of keypoints files and depth images do not match"

print(
    f"Loaded {numpy_depth.shape} frames of depth images, {numpy_color.shape} frames of color images, and {len(vitpose_keypoints2d)} files of keypoints."
)

# === Process Data ===
# Initialize lists to store data for each episode
target_positions_per_episode = []
normal_vectors_per_episode = []
keypoints_per_episode = []
thumb_vector_per_episode = []
valid_frame_numbers_per_episode = []
previous_episode = os.path.basename(os.path.dirname(color_paths[0]))

for idx, (depth_map, color_frame, color_path, keypoints_per_frame, sam2_pcd) in tqdm(
    enumerate(
        zip(numpy_depth, numpy_color, color_paths, vitpose_keypoints2d, sam2_pcds)
    ),
    total=numpy_depth.shape[0],
    desc="Generating Point Clouds",
):
    episode = os.path.basename(os.path.dirname(color_path))
    frame_index = os.path.splitext(color_path)[0].split('_')[-1]
    if episode != previous_episode:
        target_positions_per_episode = []
        normal_vectors_per_episode = []
        keypoints_per_episode = []
        thumb_vector_per_episode = []
        valid_frame_numbers_per_episode = []

    sam2_pcd = sam2_pcd[0]
    chamfer_distances = []
    target_positions_per_frame = []
    normal_vectors_per_frame = []
    thumb_vectors_per_frame = []
    pcds_per_frame = []
    invalid_keypoint = False

    for i, keypoints2d in enumerate(keypoints_per_frame):
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

        depth_keypoints_map = depth_map
        depth_keypoints_map[keypoints2d[:, 1], keypoints2d[:, 0]] = depth_map[
            keypoints2d[:, 1], keypoints2d[:, 0]
        ]

        points = np.zeros((len(keypoints2d), 3))
        colors = np.zeros((len(keypoints2d), 3))
        for j, (x, y) in enumerate(keypoints2d):
            z = depth_map[y, x]
            if j in [4, 8]:
                if z <= 250 or z >= 5000:
                    invalid_keypoint = True
            else:
                colors[j] = [1, 1, 1]
            points[j] = [x, y, z]

        points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx / 1000.0
        points[:, 1] = (points[:, 1] - cy) * points[:, 2] / fy / 1000.0
        points[:, 2] = points[:, 2] / 1000.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcds_per_frame.append(pcd)

        sam2_pcd = sam2_pcd.voxel_down_sample(voxel_size=0.01)

        chamfer_distance = pcd.compute_point_cloud_distance(sam2_pcd)
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
        continue

    valid_frame_numbers_per_episode.append(frame_index)

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

    pcd.points = o3d.utility.Vector3dVector(np.expand_dims(target_position, 0))
    colors = np.expand_dims(np.array([0, 1, 1]), axis=0)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    save_name = f"hand_keypoints_{episode}_right.npz"
    os.makedirs(os.path.join(trajectory_root, episode), exist_ok=True)
    np.savez_compressed(
        os.path.join(trajectory_root, episode, save_name),
        positions=target_positions_per_episode,
        normals=normal_vectors_per_episode,
        keypoints=keypoints_per_episode,
        thumb_vectors=thumb_vector_per_episode,
        valid_frames=valid_frame_numbers_per_episode,
    )

    previous_episode = episode

print("Batch processing complete.")
