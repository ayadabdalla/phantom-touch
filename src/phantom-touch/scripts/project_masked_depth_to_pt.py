import os
import sys
import numpy as np
import cv2
import open3d as o3d
from utils.hw_camera import fx, fy, cx, cy
from utils.depth_utils import load_raw_depth_images
from omegaconf import OmegaConf
from utils.rgb_utils import load_rgb_images


# get script metadata
script_dir = os.path.dirname(os.path.abspath(__file__))
config = OmegaConf.load(f"{script_dir}/../../sam2/conf/config.yaml")
rgb_directory_path = config.sam2videoPredictor.video_frames_dir
masks_directory_path = config.sam2videoPredictor.output_dir
output_directory_path = config.sam2videoPredictor.output_dir

### Load the data
# load the color images
numpy_color = load_rgb_images(rgb_directory_path, prefix="Color_")
#load the raw depth images
numpy_depth = load_raw_depth_images(config.sam2videoPredictor.video_frames_dir)

numpy_depth.reshape(
numpy_depth.shape[0], numpy_color.shape[1] ,numpy_color.shape[2])

# load the masks
numpy_masks = load_rgb_images(masks_directory_path, prefix="frame_")
# assert that numpy_masks and numpy_color have the same number of frames
assert numpy_masks.shape[0] == numpy_color.shape[0], "Number of frames in masks and color images do not match"
# assert that numpy_masks and numpy_depth have the same number of frames
assert numpy_masks.shape[0] == numpy_depth.shape[0], "Number of frames in masks and depth images do not match"
###

pcd = o3d.geometry.PointCloud()
i = 0
# Process each frame in the video
for depth_frame, color_frame, mask_frame in zip(numpy_depth, numpy_color, numpy_masks):
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    width, height = color_frame.shape[:2]  # (width, height, channels)
    print(f"depth frame max: {np.max(depth_frame)}")
    depth_original = (
        depth_frame.copy()
    )  # create a copy of the original depth frame for visualization
    # convert the depth frame from numpy.float16 to float
    depth_frame = np.where(
        mask_frame != 0, depth_frame, np.nan
    )  # apply the mask to the depth frame
    depth_frame = depth_frame.astype(np.uint16)

    # render the point cloud
    intrinsic = o3d.camera.PinholeCameraIntrinsic()  # Set the intrinsic parameters
    start_x=418
    start_y=148
    intrinsic.set_intrinsics(width, height, int(fx), int(fy), int(cx-start_x), int(cy-start_y))
    depth_frame_o3d = o3d.geometry.Image(
        depth_frame
    )  # Create open3d depth image from the current depth image
    color_frame_o3d = o3d.geometry.Image(
        color_frame
    )  # Convert color frame to open3d image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_frame_o3d, depth_frame_o3d, convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic, project_valid_depth_only=True
    )

    # # save and visualize the point cloud
    print(f"Number of points in the point cloud: {len(np.asarray(pcd.points))}")
    o3d.io.write_point_cloud(f"{output_directory_path}/frame_{i}.ply", pcd)
    i += 1