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
depth_map_path = config.depth_map_path
vitpose_keypoints_path = config.vitpose_keypoints_path
hamer_keypoints_path = config.hamer_keypoints_path
color_image_path = config.color_image_path


# load script inputs
hamer_keypoints2d = np.load(
    hamer_keypoints_path, allow_pickle=True
)  # Normalized camera coordinates
depth_map = np.load(depth_map_path, allow_pickle=True)
color_image = cv2.imread(color_image_path)
# load and process the vitpose keypoints
vitpose_keypoints2d = np.load(vitpose_keypoints_path, allow_pickle=True)  # pixel coordinates
vitpose_keypoints2d = vitpose_keypoints2d[:, :2]  # remove confidence score
vitpose_keypoints2d = vitpose_keypoints2d.astype(int) # convert to int

# get the depth map for the keypoints
depth_keypoints_map = np.zeros((depth_map.shape[0], depth_map.shape[1])) # create a mask for the keypoints
depth_keypoints_map[vitpose_keypoints2d[:,1],vitpose_keypoints2d[:,0]] = depth_map[vitpose_keypoints2d[:,1],vitpose_keypoints2d[:,0]]
depth_keypoints_map[depth_keypoints_map == 0] = np.nan # use nan instead of zeros for the mask
depth_keypoints_map = depth_keypoints_map.astype(np.float32) # convert depth map to float32

#alternatively create a point cloud from the depth image without o3d to have control over the points indeces
points = np.zeros((len(vitpose_keypoints2d), 3))
colors = np.zeros((len(vitpose_keypoints2d), 3))
for i, (x, y) in enumerate(vitpose_keypoints2d):
    z = depth_map[y, x]
    points[i] = [x, y, z]
    # color point 4 and 8 in red and rest in blue
    if i == 4 or i == 8:
        colors[i] = [1, 0, 0] # red
    else:
        colors[i] = [0, 0, 1] # blue
# convert to camera coordinates
points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx / 1000.0 # convert to meters
points[:, 1] = (points[:, 1] - cy) * points[:, 2] / fy / 1000.0 # convert to meters
points[:, 2] = points[:, 2] / 1000.0 # convert to meters
# save point cloud to file
np.savez_compressed("hand_keypoints_pcd.npz", points=points, colors=colors)
# create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors) # normalize to [0, 1]
points = np.asarray(pcd.points)
colors = np.asarray(pcd.colors)

# load equivalent sam2 segmented pcd hand
pcd_sam = o3d.io.read_point_cloud(config.sam2hand_pcd_path)
pcd_sam = pcd_sam.voxel_down_sample(voxel_size=0.01)

# display pcds statistics for analysis
print(f"thumb index point cloud length: {len(np.asarray(pcd.points))}")
print_stats(pcd_sam)
print_stats(pcd)

# load extra points and normal vector
midpoint = np.load("target_position_orientation.npz")
target_position = midpoint['target_position']
normal_vector = midpoint['normal_vector']
# add the target position to the point cloud
pcd.points = o3d.utility.Vector3dVector(np.vstack((points, target_position)))
points = np.asarray(pcd.points)
# add the target position color to the point cloud
target_position_color = np.asarray([0,1,0]) # yellow color
pcd.colors = o3d.utility.Vector3dVector(np.vstack((colors, target_position_color))) # add the target position color to the point cloud
colors = np.asarray(pcd.colors)
# add the normal vector to the point cloud
normal_vector = np.array([normal_vector[0], normal_vector[1], normal_vector[2]])
normal_vector = normal_vector / np.linalg.norm(normal_vector) * 0.1 # normalize to 1 cm
pcd.points = o3d.utility.Vector3dVector(np.vstack((points, target_position + normal_vector)))
normal_vector_color = np.asarray([0,1,1]) # blue color
pcd.colors = o3d.utility.Vector3dVector(np.vstack((colors, normal_vector_color))) # add the normal vector color to the point cloud

# visualize with coordinate frame
o3d.visualization.draw_geometries(
    [pcd, pcd_sam,o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
)
