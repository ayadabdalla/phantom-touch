# load the numpy keypoints and the depth map
import sys
import cv2
import numpy as np
from utils.hw_camera import fx, fy, cx, cy
import open3d as o3d

def print_stats(pcd):
    print(
        "X-axis: ",
        np.min(np.asarray(pcd.points)[:, 0]),
        np.max(np.asarray(pcd.points)[:, 0]),
        np.mean(np.asarray(pcd.points)[:, 0]),
    )
    print(
        "Y-axis: ",
        np.min(np.asarray(pcd.points)[:, 1]),
        np.max(np.asarray(pcd.points)[:, 1]),
        np.mean(np.asarray(pcd.points)[:, 1]),
    )
    print(
        "Z-axis: ",
        np.min(np.asarray(pcd.points)[:, 2]),
        np.max(np.asarray(pcd.points)[:, 2]),
        np.mean(np.asarray(pcd.points)[:, 2]),
    )
# keypoints3d_path = "/home/abdullah/utn/robotics/cameras/data/output/handover_collection_1/sample/hamer_output/keypoints_3d_Color_1920x1080_17075770ms_e00000_00054_right.npy"
keypoints2d_path = "/home/abdullah/utn/robotics/cameras/data/output/handover_collection_1/sample/hamer_output/keypoints_2d_Color_1920x1080_17075770ms_e00000_00054_left.npy"
# keypoints3d = np.load(keypoints3d_path, allow_pickle=True)
keypoints2d = np.load(keypoints2d_path, allow_pickle=True)
depth_map_path = "/home/abdullah/utn/robotics/cameras/data/recordings/handover_collection_1/sample/extracted_RawDepth_17075772ms_e00000_00054.raw.npy"
depth_map = np.load(depth_map_path, allow_pickle=True)
color_image_path = "/home/abdullah/utn/robotics/cameras/data/recordings/handover_collection_1/sample/Color_1920x1080_17075770ms_e00000_00054.png"
color_image = cv2.imread(color_image_path)

# print(keypoints3d)
print(keypoints2d)
# replace the 3rd coordinate in the 2d keypoints with the depth value indexed by the 2d keypoints
keypoints = keypoints2d[:, :2]
print(keypoints)
keypoints = keypoints.astype(int)
# create a point cloud from the depth and rgb images and indexed by only the keypoints
# set all the values other than the keypoints to nan
depth_keypoints_map = np.zeros((depth_map.shape[0], depth_map.shape[1]))

depth_keypoints_map[keypoints[:,1],keypoints[:,0]] = depth_map[keypoints[:,1],keypoints[:,0]]
depth_keypoints_map[depth_keypoints_map == 0] = np.nan
depth_keypoints_map = depth_keypoints_map.astype(np.float32)
# print the non nan values in the depth map
# create open3d images
depth_image = o3d.geometry.Image(depth_keypoints_map)
color_image = o3d.geometry.Image(color_image)
# create open3d point cloud
pcd = o3d.geometry.PointCloud()
# create rgbd image
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image, convert_rgb_to_intensity=False)
# create pinhole camera intrinsic
intrinsic = o3d.camera.PinholeCameraIntrinsic()
intrinsic.set_intrinsics(width=depth_map.shape[1], height=depth_map.shape[0], fx=fx, fy=fy, cx=cx, cy=cy)
# create point cloud from rgbd image
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
print_stats(pcd)
# color the pcd in red
# pcd.paint_uniform_color([1, 0, 0])  # Red for the mesh-derived points

# visualize the point cloud
# load an additional point cloud for visualization
pcd_sam = o3d.io.read_point_cloud("/home/abdullah/utn/robotics/cameras/data/output/handover_collection_1/sample/sam2-vid_output/frame_3.ply")
print_stats(pcd_sam)
o3d.visualization.draw_geometries([pcd,pcd_sam])
# visualize with coordinate frame
o3d.visualization.draw_geometries([pcd, pcd_sam, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])