import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2
import open3d as o3d
import time

# RealSense D405 intrinsics (example values, replace with your actual intrinsics)
fx=641.52124023
fy=641.52124023
cx=646.98626709
cy=362.58349609

# Load the masked depth video
masked_depth_video_path = "/home/abdullah/utn/phantom-human-videos/assets/masked_depth_output.avi"
cap_depth = cv2.VideoCapture(masked_depth_video_path)

# Load the corresponding color video
color_video_path = "/home/abdullah/utn/phantom-human-videos/assets/d405-color_xRPUyima.avi"  # Replace with the actual path to your color video
cap_color = cv2.VideoCapture(color_video_path)

# Create an initial empty point cloud
pcd = o3d.geometry.PointCloud()

# Process each frame in the video
frame_count = 0
while cap_depth.isOpened() and cap_color.isOpened():
    ret_depth, depth_frame = cap_depth.read()
    ret_color, color_frame = cap_color.read()
    if not ret_depth or not ret_color:
        break
    
    # For simplicity, process every 5th frame to avoid overwhelming the visualization
    frame_count += 1
    if frame_count % 5 != 0:
        continue
    
    # Convert to single-channel depth image if it's not already
    # take only the first channel
    depth_image = depth_frame # Assuming depth is in the first channel
    
    depth_image = depth_image.astype(np.float32) # Normalize depth values to [0, 1] range
    depth_image = cv2.imread("extracted_frames_color/frame_0060.jpg", cv2.IMREAD_ANYDEPTH)
    print(f"Depth image shape: {type(depth_image)}")


    # Display the depth image statistics
    print(f"Depth image shape: {depth_image.shape}")
    print(f"Min depth: {np.min(depth_image)}, Max depth: {np.max(depth_image)}")
    print(f"Depth mean: {np.mean(depth_image)}, std: {np.std(depth_image)}")
    
    # Get the shape of the depth image
    width, height = depth_image.shape
    print(f"Depth image size: {width} x {height}")
    
    # Set the intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    # Create open3d depth image from the current depth image
    depth_image_o3d = o3d.geometry.Image(depth_image)
    # transformation_matrix = np.array([
    #     [0.979678, 0.0693863, 0.188195, 0.232897],
    #     [0.199876, -0.416175, -0.887044, -0.13983],
    #     [0.0167736, 0.906633, -0.421586, 0.608288],
    #     [0, 0, 0, 1]
    # ])
    
    # Resize color frame to match depth frame size
    color_frame = cv2.imread("extracted_frames/frame_0060.jpg", cv2.IMREAD_UNCHANGED)
    print(f"Color image shape: {color_frame.shape}")
    # print|(f"depth image shape: {depth_image.shape}")
    color_frame_resized = cv2.resize(color_frame, (height, width))
    pcd.colors = o3d.utility.Vector3dVector(color_frame_resized.reshape(-1, 3) / 255.0)
    
    # Convert color frame to open3d image
    color_image_o3d = o3d.geometry.Image(color_frame_resized)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image_o3d, depth_image_o3d, convert_rgb_to_intensity=False
    )
    # Assign colors to the point cloud
    # initialize colors to black
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    # # filter the point cloud to remove NaN values
    # pcd = pcd.select_by_index(np.where(np.isfinite(pcd.points[:, 0]))[0])
    # remove outliers
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
    # # remove noise points
    # pcd = pcd.voxel_down_sample(voxel_size=0.001)
    # Display the point cloud directly without the need to call update_geometry
    o3d.visualization.draw_geometries([pcd])

    # # Optional: Display the depth image for reference
    # cv2.imshow("Depth Image", depth_image / np.max(depth_image))
    # # Optional: Display the color image for reference
    # cv2.imshow("Color Image", color_frame)
    # # Wait for key press
    # cv2.waitKey(0)    
    # # Optional: Slee
    # # p to slow down visualization
    # time.sleep(0.05)

# Release the video captures
cap_depth.release()
cap_color.release()

# Keep visualization window open until closed manually
print("Press 'q' in the visualization window to exit")