import os
import sys
import numpy as np
import trimesh
import open3d as o3d
from omegaconf import OmegaConf

# TODO: set up a class that does the downsampling, processing, ICP registration and visualization on top of o3d api

# data_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# get parent directory of the current file
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conf = OmegaConf.load(f"{parent_directory}/../conf/icp.yaml")
data_root = conf.data_root
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

# ----------- Load the Mesh and Sample Points from it -----------(hamer)
# Load the mesh (trimesh supports many formats including .obj)
mesh_hamer_path = os.path.join(
    data_root,
    conf.experiment_name,
    conf.pcd_hamer.data_ontology,
    f"{conf.pcd_hamer.data_sample_name}.{conf.pcd_hamer.data_extension}",
)
mesh_hamer = trimesh.load(mesh_hamer_path)
# Sample points uniformly from the mesh surface.
# Adjust the number of points as needed.
num_points = 10000
mesh_hamer_points = mesh_hamer.sample(num_points)
# Convert the sampled points to an Open3D point cloud.
pcd_hamer = o3d.geometry.PointCloud()
pcd_hamer.points = o3d.utility.Vector3dVector(mesh_hamer_points)
# get number of points in the mesh
print(f"Number of points in the mesh: {len(np.asarray(pcd_hamer.points))}")

# ----------- Load the PLY File -----------(sam2)
pcd_sam2_path = os.path.join(
    data_root,
    conf.experiment_name,
    conf.pcd_sam2.data_ontology,
    f"{conf.pcd_sam2.data_sample_name}.{conf.pcd_sam2.data_extension}",
)
print(f"Loading PLY file from: {pcd_sam2_path}")
pcd_sam2 = o3d.io.read_point_cloud(pcd_sam2_path)
print(f"Number of points in the PLY file: {len(np.asarray(pcd_sam2.points))}")


# ----------- Optional: Downsample for Faster ICP -----------
voxel_size = 0.005  # Adjust voxel size as needed
target = pcd_sam2_down  = pcd_sam2.voxel_down_sample(voxel_size)
target = target.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.1)[0]
voxel_size = 0.009  # Adjust voxel size as needed
source = pcd_hamer_down = pcd_hamer.voxel_down_sample(voxel_size)

# print the number of points in the downsampled point cloud
print(f"Number of points in the downsampled PLY file: {len(np.asarray(target.points))}")

# ----------- Preprocess the Point Clouds -----------
# Center the point clouds around the origin
def center_point_cloud(pcd):
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    points -= center
    pcd.points = o3d.utility.Vector3dVector(points)
center_point_cloud(source)
center_point_cloud(target)

# # invert the hamer point cloud
source.points = o3d.utility.Vector3dVector(
    np.asarray(source.points) * np.array([1, -1, -1])
)

# # Estimate normals for the point clouds
source.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
target.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)

# ----------- ICP Registration -----------
# Use an identity matrix as the initial transformation estimate.
initial_transform = np.eye(4)
max_correspondence_distance = 0.05

reg_p2l = o3d.pipelines.registration.registration_icp(
    source, target,
    max_correspondence_distance,  # Critical parameter - start with a reasonable value
    init=initial_transform,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=2000, relative_fitness=1e-6, relative_rmse=1e-6
    ),
)


print("ICP Transformation Matrix:")

# ----------- Visualize the Result -----------
# For visualization, color the two point clouds differently.
source.paint_uniform_color([1, 0, 0])  # Red for the mesh-derived points

# # Transform the mesh point cloud with the ICP result.
print(reg_p2l.transformation)
source.transform(reg_p2l.transformation)

# get the statistical details of each axis of each point cloud, min, max and mean
print("hamer Point Cloud:")
print_stats(source)
print("sam2 Point Cloud:")
print_stats(target)
# visualize the 3D coordinate system
o3d.visualization.draw_geometries([source, target, o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)])