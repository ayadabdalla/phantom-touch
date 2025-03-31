# This script takes a video of a mask and a color video, and a raw depth list and outputs
# a ply point cloud for each frame in the directory it was run in.

import numpy as np
import cv2
import open3d as o3d
from assets.constants import fx, fy, cx, cy
from utils.depth_utils import load_raw_depth_images

# Load the mask video
# the mask video should be obtained from sam2-sieve=>sam2-video workflow
mask_video_path = "/home/abdullah/utn/phantom-human2robot/sandbox/output_wf2_mask_video_2025-03-31_14-54-29.mp4"
cap_mask = cv2.VideoCapture(mask_video_path)
# Load the corresponding color video --> fetch the compiled video from the sam2-sieve=>sam2-video workflow
color_video_path = "/home/abdullah/utn/phantom-human2robot/sandbox/color_video_compiled_for_sieve_2025-03-31_14-54-29.mp4"  # Replace with the actual path to your color video
cap_color = cv2.VideoCapture(color_video_path)

# TODO: match raw depth data with the mask video
# Load the corresponding raw depth video
raw_depth_directory_path = "/home/abdullah/utn/phantom-human2robot/playground_sieve_sam_hamer/data/recordings/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/depth_raw_npy"
numpy_depth = load_raw_depth_images(raw_depth_directory_path)
# Create an initial empty point cloud
pcd = o3d.geometry.PointCloud()
i = 0


# Process each frame in the video
while cap_mask.isOpened() and cap_color.isOpened():
    ret_mask, mask_frame = cap_mask.read()
    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)  # video is in RGB
    ret_color, color_frame = cap_color.read()
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    if not ret_mask or not ret_color:
        break
    width, height = color_frame.shape[:2]  # (width, height, channels)
    depth_frame = numpy_depth[i] * 1000  # to mm
    depth_frame = depth_frame.reshape(width, height)  # convert to 2D
    depth_original = (
        depth_frame.copy()
    )  # create a copy of the original depth frame for visualization
    depth_frame = np.where(
        mask_frame != 0, depth_frame, np.nan
    )  # apply the mask to the depth frame

    # render the point cloud
    intrinsic = o3d.camera.PinholeCameraIntrinsic()  # Set the intrinsic parameters
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
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

    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] > 0.07)[0])
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < 0.5)[0])

    # visualize the depth image
    depth_original = depth_original - np.nanmin(
        depth_original
    )  # normalize the depth image first
    depth_original = depth_original / np.nanmax(depth_original)
    depth_original = cv2.equalizeHist((depth_original * 255).astype(np.uint8))
    # cv2.imshow("Depth Image", depth_original)

    # visualize the mask
    # cv2.imshow("Mask", mask_frame)

    # visualize the color image
    # color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Color Image", color_frame)

    # end visualization mechanism
    # cv2.waitKey(0)

    # save and visualize the point cloud
    i += 1
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(f"frame_{i}.ply", pcd)
# Release the video captures
cap_mask.release()
cap_color.release()
