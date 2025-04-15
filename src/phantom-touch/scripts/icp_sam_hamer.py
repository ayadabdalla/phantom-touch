import os
import sys
import numpy as np
import trimesh
import open3d as o3d

# TODO: set up a class that does the downsampling, processing, ICP registration and visualization on top of o3d api


data_source = "output"
experiment_name = "test_exp_streaming"
# experiment_specifics = "white_nonreflective_cloth_light_on_ambient_light"
data_ontology = "hamer_output"
# sub_data_sample_id = "0"
# sample_id = "0840"
# data_sample_name = f"frame_{sample_id}_{sub_data_sample_id}"
data_sample_name = "Color_1920x1080_24931695ms_e00000_00000_0"
data_extension = "obj"
repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------- Load the Mesh and Sample Points from it -----------(hamer)
# Load the mesh (trimesh supports many formats including .obj)
mesh_path = os.path.join(
    "/mnt/dataset_drive/ayad/phantom-touch",
    # repository_root,
    # "assets",
    "data",
    data_source,
    experiment_name,
    # experiment_specifics,
    data_ontology,
    f"{data_sample_name}.{data_extension}",
)
mesh = trimesh.load(mesh_path)
# Sample points uniformly from the mesh surface.
# Adjust the number of points as needed.
num_points = 10000
points = mesh.sample(num_points)
# Convert the sampled points to an Open3D point cloud.
pcd_mesh = o3d.geometry.PointCloud()
pcd_mesh.points = o3d.utility.Vector3dVector(points)
# get number of points in the mesh
print(f"Number of points in the mesh: {len(np.asarray(pcd_mesh.points))}")

# ----------- Load the Point Cloud from the PLY File -----------(sam2)
data_ontology = "sam2-vid_output"
data_extension = "ply"
data_sample_name = f"frame_1"
ply_path = os.path.join(
    "/mnt/dataset_drive/ayad/phantom-touch",
    # repository_root,
    # "assets",
    "data",
    data_source,
    experiment_name,
    # experiment_specifics,
    data_ontology,
    f"{data_sample_name}.{data_extension}",
)
print(f"Loading PLY file from: {ply_path}")
pcd_ply = o3d.io.read_point_cloud(ply_path)
print(f"Number of points in the PLY file: {len(np.asarray(pcd_ply.points))}")
# ----------- Optional: Downsample for Faster ICP -----------
voxel_size = 0.005  # Adjust voxel size as needed
# filter the point clouds for noisy points and outliers
pcd_mesh_filtered = pcd_mesh.remove_radius_outlier(
    nb_points=16, radius=0.05
)[0]  # remove points that are too far from the rest of the points
pcd_mesh_filtered = pcd_mesh_filtered.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0
)[0]  # remove points that are too far from the rest of the points

target = pcd_mesh_down = pcd_mesh_filtered.voxel_down_sample(voxel_size)
source = pcd_ply_down = pcd_ply.voxel_down_sample(voxel_size)


# ----------- Preprocess the Point Clouds -----------
# Center the point clouds around the origin
pcd_mesh_points_z = np.asarray(pcd_mesh_down.points)[:, 2] - np.mean(
    np.asarray(pcd_mesh_down.points)[:, 2]
)
pcd_mesh_points_y = np.asarray(pcd_mesh_down.points)[:, 1] - np.mean(
    np.asarray(pcd_mesh_down.points)[:, 1]
)
pcd_mesh_points_x = np.asarray(pcd_mesh_down.points)[:, 0] - np.mean(
    np.asarray(pcd_mesh_down.points)[:, 0]
)

pcd_ply_points_z = np.asarray(pcd_ply_down.points)[:, 2] - np.mean(
    np.asarray(pcd_ply_down.points)[:, 2]
)
pcd_ply_points_y = np.asarray(pcd_ply_down.points)[:, 1] - np.mean(
    np.asarray(pcd_ply_down.points)[:, 1]
)
pcd_ply_points_x = np.asarray(pcd_ply_down.points)[:, 0] - np.mean(
    np.asarray(pcd_ply_down.points)[:, 0]
)

pcd_ply_down.points = o3d.utility.Vector3dVector(
    np.column_stack((pcd_ply_points_x, pcd_ply_points_y, pcd_ply_points_z))
)
pcd_mesh_down.points = o3d.utility.Vector3dVector(
    np.column_stack((pcd_mesh_points_x, pcd_mesh_points_y, pcd_mesh_points_z))
)

# Estimate normals for the point clouds
pcd_mesh_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)
pcd_ply_down.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
)

# ----------- ICP Registration -----------
# Set a distance threshold (max correspondence points-pair distance)
threshold = 0.05
# Use an identity matrix as the initial transformation estimate.
trans_init = np.eye(4)

# # Perform point-to-point ICP registration.
reg_p2l = o3d.pipelines.registration.registration_icp(
    source,
    target,
    threshold,
    trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
)

print("ICP Transformation Matrix:")

# ----------- Visualize the Result -----------
# For visualization, color the two point clouds differently.
pcd_mesh_down.paint_uniform_color([1, 0, 0])  # Red for the mesh-derived points
pcd_ply_down.paint_uniform_color([0, 1, 0])  # Green for the PLY point cloud

# # Transform the mesh point cloud with the ICP result.
print(reg_p2l.transformation)
pcd_ply_down.transform(reg_p2l.transformation)

# Visualize both point clouds
o3d.visualization.draw_geometries([pcd_mesh_down])
o3d.visualization.draw_geometries([pcd_ply_down])

# get the statistical details of each axis of each point cloud, min, max and mean
print("Mesh Point Cloud:")
print(
    "X-axis: ",
    np.min(np.asarray(pcd_mesh_down.points)[:, 0]),
    np.max(np.asarray(pcd_mesh_down.points)[:, 0]),
    np.mean(np.asarray(pcd_mesh_down.points)[:, 0]),
)
print(
    "Y-axis: ",
    np.min(np.asarray(pcd_mesh_down.points)[:, 1]),
    np.max(np.asarray(pcd_mesh_down.points)[:, 1]),
    np.mean(np.asarray(pcd_mesh_down.points)[:, 1]),
)
print(
    "Z-axis: ",
    np.min(np.asarray(pcd_mesh_down.points)[:, 2]),
    np.max(np.asarray(pcd_mesh_down.points)[:, 2]),
    np.mean(np.asarray(pcd_mesh_down.points)[:, 2]),
)
print("PLY Point Cloud:")
print(
    "X-axis: ",
    np.min(np.asarray(pcd_ply_down.points)[:, 0]),
    np.max(np.asarray(pcd_ply_down.points)[:, 0]),
    np.mean(np.asarray(pcd_ply_down.points)[:, 0]),
)
print(
    "Y-axis: ",
    np.min(np.asarray(pcd_ply_down.points)[:, 1]),
    np.max(np.asarray(pcd_ply_down.points)[:, 1]),
    np.mean(np.asarray(pcd_ply_down.points)[:, 1]),
)
print(
    "Z-axis: ",
    np.min(np.asarray(pcd_ply_down.points)[:, 2]),
    np.max(np.asarray(pcd_ply_down.points)[:, 2]),
    np.mean(np.asarray(pcd_ply_down.points)[:, 2]),
)

# # Visualize the transformed point cloud
o3d.visualization.draw_geometries([pcd_mesh_down, pcd_ply_down])
