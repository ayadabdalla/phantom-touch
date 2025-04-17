# load the numpy keypoints and the depth map
import sys
import cv2
import numpy as np
from utils.hw_camera import fx, fy, cx, cy
import open3d as o3d
from utils.pointcloud_utils import print_stats
from omegaconf import OmegaConf
import os

# get metadata
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = OmegaConf.load(f"{parent_directory}/conf/3d_projection.yaml")
keypoints2d_path = config.keypoints2d_path
depth_map_path = config.depth_map_path

# load script inputs
depth_map = np.load(depth_map_path, allow_pickle=True)
color_image_path = config.color_image_path
color_image = cv2.imread(color_image_path)
keypoints2d = np.load(keypoints2d_path, allow_pickle=True)

# keep only deoth map pixels indexed by the keypoints
keypoints = keypoints2d[:, :2] # remove confidence score
keypoints = keypoints.astype(int) # convert to int
depth_keypoints_map = np.zeros((depth_map.shape[0], depth_map.shape[1])) # create a mask for the keypoints
depth_keypoints_map[keypoints[:,1],keypoints[:,0]] = depth_map[keypoints[:,1],keypoints[:,0]]
depth_keypoints_map[depth_keypoints_map == 0] = np.nan # use nan instead of zeros for the mask
depth_keypoints_map = depth_keypoints_map.astype(np.float32) # convert depth map to float32


# create point cloud from rgbd image
depth_image = o3d.geometry.Image(depth_keypoints_map)
color_image = o3d.geometry.Image(color_image)
pcd = o3d.geometry.PointCloud()
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, convert_rgb_to_intensity=False)
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=depth_map.shape[1], height=depth_map.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

# load equivalent sam2 segmented pcd hand
pcd_sam = o3d.io.read_point_cloud(config.sam2hand_pcd_path)

# display pcds statistics for analysis
print_stats(pcd_sam)
print_stats(pcd)

# visualize with coordinate frame
o3d.visualization.draw_geometries([pcd, pcd_sam, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])