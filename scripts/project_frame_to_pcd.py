import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def depth_to_point_cloud(depth_image, intrinsic_matrix):
    """
    Converts a depth image to a 3D point cloud.
    
    Args:
        depth_image (numpy.ndarray): A 2D array where each pixel contains the depth value.
        intrinsic_matrix (numpy.ndarray): The 3x3 camera intrinsic matrix.
        
    Returns:
        numpy.ndarray: A 3D point cloud as an Nx3 array, where N is the number of valid points.
    """
    # Get the dimensions of the depth image
    height, width = depth_image.shape
    
    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    
    # Flatten the depth image and filter out invalid depth values (e.g., 0 or NaN)
    depth_values = depth_image.flatten()
    valid_indices = (depth_values > 0) & ~np.isnan(depth_values)  # Filter out invalid depths
    u = u[valid_indices]
    v = v[valid_indices]
    depth_values = depth_values[valid_indices]
    
    # Homogeneous pixel coordinates
    pixel_coords = np.vstack((u, v, np.ones_like(u)))
    # display pixel coordinates min, max and mean
    print(f"Pixel coordinates shape: {pixel_coords.shape}")
    print(f"Pixel coordinates min: {np.min(pixel_coords)}")
    print(f"Pixel coordinates max: {np.max(pixel_coords)}")
    print(f"Pixel coordinates mean: {np.mean(pixel_coords)}")

    # Back-project to 3D camera coordinates
    point_cloud_camera = np.linalg.inv(intrinsic_matrix) @ pixel_coords
    point_cloud_camera *= depth_values  # Scale by depth
    
    # Transpose to get Nx3 array
    point_cloud = point_cloud_camera.T
    
    return point_cloud, valid_indices, u, v

def plot_point_cloud(point_cloud, colors):
    """
    Plots a 3D point cloud using Matplotlib.
    
    Args:
        point_cloud (numpy.ndarray): An Nx3 array of 3D points.
        colors (numpy.ndarray): An Nx3 array of RGB colors for each point.
    """
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    # Create a grid
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate Z values on the grid
    Z = griddata((x, y), z, (X, Y), method='linear')
    # Create figure for 3D visualization
    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection ='3d')
    surf = ax.plot_surface(X=X, Y=Y, Z=Z, cmap='viridis', 
                          linewidth=0, antialiased=False)
    
    # Set labels
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('Depth (meters)')
    ax.set_title('3D Depth Visualization')
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example depth image (replace with your own depth image)
    depth_image_path = "/home/abdullah/utn/phantom-human-videos/extracted_frames/frame_0060.jpg"
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        raise ValueError("Could not read the depth image. Please check the path.")
    
    # Example color image (replace with your own color image)
    color_image_path = "/home/abdullah/utn/phantom-human-videos/extracted_frames_color/frame_0060.jpg"
    color_image = cv2.imread(color_image_path)
    if color_image is None:
        raise ValueError("Could not read the color image. Please check the path.")
    
    # Convert color image from BGR to RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    
    # Example intrinsic matrix (replace with your camera's intrinsic matrix)
    fx=641.52124023
    fy=641.52124023
    cx=646.98626709
    cy=362.58349609
    intrinsic_matrix = np.array([
        [fx, 0, cx],  # fx, 0, cx
        [0, fy, cy],  # 0, fy, cy
        [0, 0, 1]       # 0, 0, 1
    ], dtype=np.float32)
    
    # Generate the point cloud
    point_cloud, valid_indices, u, v = depth_to_point_cloud(depth_image, intrinsic_matrix)
    
    # Get the corresponding colors from the color image
    colors = color_image.reshape(-1, 3)  # Flatten the color image
    colors = colors[valid_indices]  # Filter colors based on valid depth indices
    
    # Print the point cloud
    print("Point Cloud (Nx3):")
    print(point_cloud)
    
    # Plot the point cloud with colors
    plot_point_cloud(point_cloud, colors)
    #show both images and the point cloud
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(depth_image, cmap='gray')
    plt.title('Depth Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(color_image)
    plt.title('Color Image')
    plt.axis('off')
    plt.show()