import os
import numpy as np
import cv2
import open3d as o3d
import glob
from tqdm import tqdm  # <-- NEW
from utils.hw_camera import fx, fy, cx, cy
from utils.depth_utils import load_raw_depth_images
from omegaconf import OmegaConf
from utils.rgb_utils import load_rgb_images
from utils.sam2utils import search_folder

# === Configuration ===
parent_script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
repo_dir = search_folder("/home", "phantom-touch")
sam2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

threed_conf = OmegaConf.load(f"{parent_script_dir}/conf/3d_projection.yaml")
paths_cfg = OmegaConf.load(f"repo_dir/cfg/paths.yaml")

rgb_directory_path = paths_cfg.recordings_directory
masks_directory_path = paths_cfg.masks_directory
output_directory_path = paths_cfg.sam2hand_directory
shape = threed_conf.shape

# === Load Data ===
# Load color images and their paths
numpy_color, color_paths = load_rgb_images(rgb_directory_path, prefix="Color_", return_path=True)
print(f"Color shape: {numpy_color.shape}")

# Load mask images and their paths
numpy_masks, mask_paths = load_rgb_images(masks_directory_path, prefix="Mask_", return_path=True)
print(f"Mask shape: {numpy_masks.shape}")

# Load depth images
numpy_depth = load_raw_depth_images(rgb_directory_path, shape=shape)
print(f"Depth shape: {numpy_depth.shape}")

# numpy_depth = numpy_depth.reshape(numpy_depth.shape[0], numpy_color.shape[1], numpy_color.shape[2])

assert numpy_masks.shape[0] == numpy_color.shape[0], "Number of frames in masks and color images do not match"
assert numpy_masks.shape[0] == numpy_depth.shape[0], "Number of frames in masks and depth images do not match"

# === Processing with Progress Bar ===
for idx, (depth_frame, color_frame, mask_frame) in tqdm(enumerate(zip(numpy_depth, numpy_color, numpy_masks)), total=numpy_depth.shape[0], desc="Generating Point Clouds"):
    # Preprocess images
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

    height, width = color_frame.shape[:2]

    # Apply mask to depth
    depth_frame = np.where(mask_frame != 0, depth_frame, np.nan)
    depth_frame = depth_frame.astype(np.uint16)

    # Set camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    start_x = threed_conf.crop.x
    start_y = threed_conf.crop.y
    intrinsic.set_intrinsics(width, height, int(fx), int(fy), int(cx - start_x), int(cy - start_y))

    # Create Open3D RGBD image
    depth_frame_o3d = o3d.geometry.Image(depth_frame)
    color_frame_o3d = o3d.geometry.Image(color_frame)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_frame_o3d, depth_frame_o3d, convert_rgb_to_intensity=False
    )

    # Generate PointCloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, project_valid_depth_only=True)

    # === Save ===
    original_path = color_paths[idx]  # Full path
    episode_name = os.path.basename(os.path.dirname(original_path))  # e.g., "e0"
    original_filename = os.path.basename(original_path)  # e.g., "Color_0034.png"
    save_name = os.path.splitext(original_filename)[0] + ".ply"  # e.g., "Color_0034.ply"

    episode_output_dir = os.path.join(output_directory_path, episode_name)
    os.makedirs(episode_output_dir, exist_ok=True)  # Create folder if it doesn't exist

    save_path = os.path.join(episode_output_dir, save_name)

    o3d.io.write_point_cloud(save_path, pcd)

    # Optional: print after every N saves if you want
    # if idx % 50 == 0:
    #     print(f"Saved: {save_path} with {len(np.asarray(pcd.points))} points")

print("✅ All point clouds generated!")
