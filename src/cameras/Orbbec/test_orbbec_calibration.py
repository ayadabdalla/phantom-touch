#!/usr/bin/env python3
"""
Test Orbbec Camera Calibration
================================
This script visualizes the camera calibration by:
1. Loading RGB and depth images from recorded data
2. Creating a 3D point cloud from the depth data
3. Transforming the point cloud from camera frame to robot base frame
4. Displaying the result with coordinate axes in Open3D viewer

Usage:
    python test_orbbec_calibration.py --data_dir /path/to/data
    
    Where data_dir contains:
    - Color_*.png files (RGB images from Orbbec)
    - RawDepth_*.raw files (depth data from Orbbec)

The coordinate axes shown in the viewer:
- Red axis: X (pointing right in robot frame)
- Green axis: Y (pointing forward in robot frame)  
- Blue axis: Z (pointing up in robot frame)

If the calibration is correct, the scene should be properly aligned with the robot base frame.
"""

import argparse
import glob
import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
import re


# Orbbec Femto Bolt camera intrinsics
# These values match the calibrated camera used in the phantom-touch pipeline
ORBBEC_FX = 1124.38
ORBBEC_FY = 1123.93
ORBBEC_CX = 951.31
ORBBEC_CY = 543.694

# Image dimensions for Orbbec Femto Bolt
WIDTH = 1920
HEIGHT = 1080


def natural_sort_key(s):
    """
    Sort filenames naturally (e.g., file_1, file_2, file_10 instead of file_1, file_10, file_2).
    Extracts the numeric frame index from filenames like 'Color_1920x1080_3199677439ms_00000.png'
    """
    # Extract the frame number from the end of the filename (before extension)
    match = re.search(r'_(\d+)\.', s)
    return int(match.group(1)) if match else 0


def load_orbbec_data(data_dir, frame_idx=0):
    """
    Load RGB and depth data from saved Orbbec recordings.
    
    Args:
        data_dir: Directory containing Color_*.png and RawDepth_*.raw files
        frame_idx: Which frame to load (default: 0 for first frame)
    
    Returns:
        rgb_image: numpy array (H, W, 3) in BGR format
        depth_image: numpy array (H, W) with depth values in millimeters
    """
    data_path = Path(data_dir)
    
    # Find all color and depth files
    color_files = sorted(glob.glob(str(data_path / "Color_*.png")), key=natural_sort_key)
    depth_files = sorted(glob.glob(str(data_path / "RawDepth_*.raw")), key=natural_sort_key)
    
    if not color_files or not depth_files:
        raise FileNotFoundError(f"No Color_*.png or RawDepth_*.raw files found in {data_dir}")
    
    if frame_idx >= len(color_files) or frame_idx >= len(depth_files):
        raise IndexError(f"Frame index {frame_idx} out of range. Found {len(color_files)} color and {len(depth_files)} depth frames.")
    
    print(f"Loading frame {frame_idx}:")
    print(f"  Color: {Path(color_files[frame_idx]).name}")
    print(f"  Depth: {Path(depth_files[frame_idx]).name}")
    
    # Load RGB image
    rgb_image = cv2.imread(color_files[frame_idx])
    if rgb_image is None:
        raise ValueError(f"Failed to load RGB image from {color_files[frame_idx]}")
    
    # Load depth image (raw binary format, uint16, little-endian)
    depth_image = np.fromfile(depth_files[frame_idx], dtype=np.uint16)
    depth_image = depth_image.reshape((HEIGHT, WIDTH))
    
    return rgb_image, depth_image


def create_point_cloud_from_depth(rgb_image, depth_image, fx, fy, cx, cy):
    """
    Create an Open3D point cloud from RGB and depth images.
    
    Args:
        rgb_image: numpy array (H, W, 3) in BGR format
        depth_image: numpy array (H, W) with depth in millimeters
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates in pixels
    
    Returns:
        o3d.geometry.PointCloud: Point cloud in camera frame
    """
    # Convert depth from millimeters to meters
    depth_meters = depth_image.astype(np.float32) / 1000.0
    
    # Create Open3D images
    o3d_rgb = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    o3d_depth = o3d.geometry.Image(depth_meters)
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb,
        o3d_depth,
        depth_scale=1.0,  # Already in meters
        depth_trunc=3.0,  # Truncate depth beyond 3 meters
        convert_rgb_to_intensity=False
    )
    
    # Create camera intrinsic matrix
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=WIDTH,
        height=HEIGHT,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )
    
    # Generate point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        intrinsic
    )
    
    return pcd


def load_camera_to_robot_transform(calib_file):
    """
    Load the camera-to-robot transformation matrix.
    
    Args:
        calib_file: Path to .npy file containing 4x4 transformation matrix
    
    Returns:
        4x4 numpy array: Transformation matrix from camera frame to robot base frame
    """
    transform = np.load(calib_file)
    
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transformation matrix, got shape {transform.shape}")
    
    print(f"\nLoaded camera-to-robot transform from {calib_file}:")
    print(f"Translation (x, y, z): {transform[0:3, 3]}")
    
    # Extract rotation matrix and convert to euler angles for display
    rotation = transform[0:3, 0:3]
    print(f"Rotation matrix:\n{rotation}")
    
    return transform


def create_coordinate_frame(size=0.1):
    """
    Create a coordinate frame (RGB axes) for visualization.
    
    Args:
        size: Length of each axis in meters
    
    Returns:
        o3d.geometry.TriangleMesh: Coordinate frame mesh with RGB axes
    """
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def visualize_point_cloud_with_frame(pcd, coordinate_frame):
    """
    Visualize point cloud with coordinate frame using Open3D viewer.
    
    Args:
        pcd: o3d.geometry.PointCloud to display
        coordinate_frame: o3d.geometry.TriangleMesh coordinate axes
    """
    print("\n=== Open3D Visualization Controls ===")
    print("- Mouse left: Rotate view")
    print("- Mouse wheel: Zoom in/out")
    print("- Mouse right: Pan view")
    print("- Press 'Q' or close window to exit")
    print("=====================================\n")
    
    # Visualize
    o3d.visualization.draw_geometries(
        [pcd, coordinate_frame],
        window_name="Orbbec Camera Calibration Test",
        width=1280,
        height=720,
        left=50,
        top=50
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test Orbbec camera calibration by visualizing point cloud in robot frame",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/epon04yc/example_2",
        help="Directory containing Color_*.png and RawDepth_*.raw files (default: /home/epon04yc/example)"
    )
    
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to visualize (default: 0)"
    )
    
    parser.add_argument(
        "--calib_file",
        type=str,
        default="/home/epon04yc/phantom-touch/src/cameras/Orbbec/data/robotbase_camera_transform_orbbec_fr4.npy",
        help="Path to camera calibration .npy file"
    )
    
    parser.add_argument(
        "--axis_size",
        type=float,
        default=0.1,
        help="Size of coordinate axes in meters (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Orbbec Camera Calibration Test")
    print("=" * 60)
    
    # Step 1: Load RGB and depth data
    print("\n[1/4] Loading RGB and depth images...")
    rgb_image, depth_image = load_orbbec_data(args.data_dir, args.frame)
    print(f"      RGB shape: {rgb_image.shape}")
    print(f"      Depth shape: {depth_image.shape}")
    print(f"      Depth range: {depth_image.min()} - {depth_image.max()} mm")
    
    # Step 2: Create point cloud in camera frame
    print("\n[2/4] Creating point cloud from depth data...")
    pcd_camera_frame = create_point_cloud_from_depth(
        rgb_image, depth_image,
        ORBBEC_FX, ORBBEC_FY, ORBBEC_CX, ORBBEC_CY
    )
    print(f"      Point cloud contains {len(pcd_camera_frame.points)} points")
    
    # Step 3: Load calibration and transform to robot frame
    print("\n[3/4] Transforming to robot base frame...")
    camera_to_robot = load_camera_to_robot_transform(args.calib_file)
    
    # Transform point cloud from camera frame to robot base frame
    pcd_robot_frame = pcd_camera_frame.transform(camera_to_robot)
    print("      Transformation applied successfully")
    
    # Step 4: Create coordinate frame at robot base (origin)
    print("\n[4/4] Creating visualization...")
    coordinate_frame = create_coordinate_frame(size=args.axis_size)
    print(f"      Coordinate axes size: {args.axis_size} meters")
    print("      Red=X, Green=Y, Blue=Z (robot base frame)")
    
    # Visualize
    print("\nOpening visualization window...")
    visualize_point_cloud_with_frame(pcd_robot_frame, coordinate_frame)
    
    print("\n" + "=" * 60)
    print("Calibration test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
