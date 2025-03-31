import os
import sys
import cv2
import numpy as np
import open3d as o3d

print("Load a ply point cloud, print it, and render it")
# load indices file
indices = np.load("indices.npy")
# load an array of all ply files in the current directory
files = [f for f in os.listdir('.') if f.endswith('.ply')]
for file in files:
    pcd = o3d.io.read_point_cloud(file)
    # filter every thing farther than 50 cm and closer than 7 cm
    points = np.asarray(pcd.points)
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,2] > -0.5)[0])
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:,2] < -0.07)[0])
    # use the indices to filter the point cloud
    pcd = pcd.select_by_index(indices[0])
    o3d.visualization.draw_geometries([pcd])
