# This script takes a video of a mask and a color video, and a raw depth list and outputs
# a ply point cloud for each frame in the directory it was run in.

import os
import sys
import numpy as np
import cv2
import open3d as o3d


from utils.hw_camera import fx, fy, cx, cy
from utils.depth_utils import load_raw_depth_images
from omegaconf import OmegaConf
from utils.rgb_utils import fetch_rgb_video

# get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
config = OmegaConf.load(f"{script_dir}/../../sam2/conf/config.yaml")
rgb_directory_path = config.sam2videoPredictor.video_frames_dir
masks_directory_path = config.sam2videoPredictor.output_dir
def load_rgb_images(rgb_directory_path, prefix=None):
    rgb_images = []
    for filename in sorted(os.listdir(rgb_directory_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(rgb_directory_path, filename)
            img = cv2.imread(img_path)
            rgb_images.append(img)
    return np.array(rgb_images)
# load the color images
numpy_color = load_rgb_images(rgb_directory_path, prefix="Color_")
#load the raw depth images
numpy_depth = load_raw_depth_images(config.sam2videoPredictor.video_frames_dir)
numpy_depth.reshape(
    numpy_depth.shape[0], numpy_color.shape[1] ,numpy_color.shape[2])
# Create an initial empty point cloud
numpy_masks = load_rgb_images(masks_directory_path, prefix="frame_")
# assert that numpy_masks and numpy_color have the same number of frames
assert numpy_masks.shape[0] == numpy_color.shape[0], "Number of frames in masks and color images do not match"
# assert that numpy_masks and numpy_depth have the same number of frames
assert numpy_masks.shape[0] == numpy_depth.shape[0], "Number of frames in masks and depth images do not match"
pcd = o3d.geometry.PointCloud()
i = 0
# Process each frame in the video
for depth_frame, color_frame, mask_frame in zip(numpy_depth, numpy_color, numpy_masks):
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
    width, height = color_frame.shape[:2]  # (width, height, channels)
    print(f"depth frame max: {np.max(depth_frame)}")
    depth_frame = depth_frame.reshape(width, height)  # convert to 2D
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
    # print intrinsic parameters
    print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
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

    # pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] > 0.07)[0])
    # pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 2] < 0.5)[0])

    # # visualize the depth image
    # depth_original = depth_original - np.nanmin(
    #     depth_original
    # )  # normalize the depth image first
    # depth_original = depth_original / np.nanmax(depth_original)
    # depth_original = cv2.equalizeHist((depth_original * 255).astype(np.uint8))
    # cv2.imshow("Depth Image", depth_original)

    # # visualize the mask
    # mask_frame = mask_frame.astype(np.uint8)
    # mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("Mask", mask_frame)

    # # visualize the color image
    # color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2BGR)
    # cv2.imshow("Color Image", color_frame)

    # # show masked depth image
    # masked_depth = cv2.bitwise_and(depth_original, depth_original, mask=mask_frame)
    # masked_depth = cv2.cvtColor(masked_depth, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("Masked Depth Image", masked_depth)
    # # end visualization mechanism
    # cv2.waitKey(0)

    # # save and visualize the point cloud
    print(f"Number of points in the point cloud: {len(np.asarray(pcd.points))}")
    # o3d.visualization.draw_geometries([pcd])

    i += 1

    o3d.io.write_point_cloud(f"/mnt/dataset_drive/ayad/phantom-touch/data/output/test_exp_streaming/sam2-vid_output/frame_{i}.ply", pcd)
    # get number of points in the point cloud
    print(f"Number of points in the point cloud: {len(np.asarray(pcd.points))}")