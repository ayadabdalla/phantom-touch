"""Example script showing how to load and access 3D tracking data from NPZ files.

This script demonstrates:
- Loading tracking data with new unambiguous key names
- Understanding reference frames (camera vs robot)
- Accessing positions, rotations, and transforms
- Filtering by quality
- Trajectory analysis
"""

import numpy as np
import os


def load_episode_tracking(episode_num, tracking_dir):
    """
    Load tracking data for a specific episode.

    Args:
        episode_num: Episode number to load
        tracking_dir: Directory containing tracking NPZ files

    Returns:
        Dictionary containing all tracking data (numpy arrays accessible by key)
    """
    tracking_file = os.path.join(tracking_dir, f"episode_{episode_num:02d}_tracking.npz")

    if not os.path.exists(tracking_file):
        raise FileNotFoundError(f"Tracking file not found: {tracking_file}")

    data = np.load(tracking_file, allow_pickle=True)
    print(f"\nLoaded tracking data from: {tracking_file}")
    print(f"Available keys: {list(data.keys())}")

    return data


def print_tracking_summary(data):
    """Print a summary of the tracking data with new key names."""
    print("\n" + "="*70)
    print("3D TRACKING DATA SUMMARY")
    print("="*70)

    # Metadata
    print(f"\nEpisode: {data['episode_number']}")
    print(f"Frames tracked: {data['num_frames_tracked']} / {data['num_frames_total']}")
    print(f"Average fitness: {np.mean(data['icp_fitness']):.4f}")

    # CAD model info
    print(f"\nCAD Model: {data['cad_model_path']}")
    print(f"Sample points: {data['cad_num_sample_points']}")
    print(f"Scale factor: {data['cad_scale_factor']}")

    # Position statistics (ROBOT FRAME - main output)
    positions_robot = data['object_pos_in_robot']
    print(f"\nPosition range (ROBOT FRAME):")
    print(f"  X: [{positions_robot[:, 0].min():.3f}, {positions_robot[:, 0].max():.3f}] m")
    print(f"  Y: [{positions_robot[:, 1].min():.3f}, {positions_robot[:, 1].max():.3f}] m")
    print(f"  Z: [{positions_robot[:, 2].min():.3f}, {positions_robot[:, 2].max():.3f}] m")

    # Trajectory info
    displacement = data['displacement_from_start']
    total_displacement = np.linalg.norm(displacement[-1])
    print(f"\nTrajectory:")
    print(f"  Start position (robot): {data['first_pos_robot']}")
    print(f"  Total displacement: {total_displacement:.3f} m")

    # Frame info
    frame_indices = data['frame_idx']
    print(f"\nFrame indices: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")


def demonstrate_key_usage(data):
    """Demonstrate how to use different keys for different tasks."""
    print("\n" + "="*70)
    print("KEY USAGE EXAMPLES")
    print("="*70)

    # 1. Main position output (ROBOT FRAME)
    print("\n1. OBJECT POSITIONS (Robot Frame - Primary Output):")
    positions_robot = data['object_pos_in_robot']
    print(f"   Shape: {positions_robot.shape}")
    print(f"   First position: {positions_robot[0]}")
    print(f"   Use this for: Trajectories, analysis, visualization")

    # 2. Object orientations (ROBOT FRAME)
    print("\n2. OBJECT ROTATIONS (Robot Frame - Primary Output):")
    rotations_robot = data['R_object_in_robot']
    print(f"   Shape: {rotations_robot.shape}")
    print(f"   First rotation:\n{rotations_robot[0]}")
    print(f"   Use this for: Object orientation in world")

    # 3. Full transforms
    print("\n3. TRANSFORMS (Centered CAD → Robot Frame):")
    transforms = data['T_robot_from_cad']
    print(f"   Shape: {transforms.shape}")
    print(f"   Use this for: Visualizing CAD model at tracked poses")

    # 4. Intermediate data (CAMERA FRAME)
    print("\n4. INTERMEDIATE DATA (Camera Frame):")
    positions_camera = data['object_pos_in_camera']
    print(f"   Positions in camera: {positions_camera.shape}")
    print(f"   Use this for: Debugging, camera-space analysis")

    # 5. ICP data (CENTERED SPACE)
    print("\n5. ICP ROTATIONS (Centered Space):")
    icp_rotations = data['R_icp_in_centered_space']
    print(f"   Shape: {icp_rotations.shape}")
    print(f"   Use this for: Understanding ICP alignment")

    # 6. Quality metrics
    print("\n6. QUALITY METRICS:")
    fitness = data['icp_fitness']
    print(f"   Mean fitness: {fitness.mean():.4f}")
    print(f"   Min fitness: {fitness.min():.4f}")
    print(f"   Max fitness: {fitness.max():.4f}")
    print(f"   Use this for: Filtering low-quality frames")


def demonstrate_filtering(data):
    """Show how to filter data by quality."""
    print("\n" + "="*70)
    print("FILTERING BY QUALITY")
    print("="*70)

    fitness_threshold = 0.7
    good_mask = data['icp_fitness'] > fitness_threshold

    print(f"\nFiltering frames with fitness > {fitness_threshold}")
    print(f"  Total frames: {len(data['frame_idx'])}")
    print(f"  High quality frames: {good_mask.sum()}")
    print(f"  Percentage: {100 * good_mask.sum() / len(good_mask):.1f}%")

    # Get high quality positions
    good_positions = data['object_pos_in_robot'][good_mask]
    good_frames = data['frame_idx'][good_mask]

    print(f"\nHigh quality frame indices: {good_frames[:10]}{'...' if len(good_frames) > 10 else ''}")


def demonstrate_trajectory_analysis(data):
    """Demonstrate trajectory analysis."""
    print("\n" + "="*70)
    print("TRAJECTORY ANALYSIS")
    print("="*70)

    positions = data['object_pos_in_robot']

    # Total distance
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_distance = distances.sum()
    print(f"\nTotal distance traveled: {total_distance:.3f} m")

    # Displacement from start
    displacement_vectors = data['displacement_from_start']
    final_displacement = np.linalg.norm(displacement_vectors[-1])
    print(f"Straight-line displacement: {final_displacement:.3f} m")
    print(f"Path efficiency: {final_displacement / total_distance * 100:.1f}%")

    # Per-axis motion
    axis_names = ['X', 'Y', 'Z']
    print(f"\nMotion per axis:")
    for i, axis in enumerate(axis_names):
        axis_travel = np.abs(np.diff(positions[:, i])).sum()
        print(f"  {axis}: {axis_travel:.3f} m")


def demonstrate_frame_conversion(data):
    """Show how to convert between reference frames."""
    print("\n" + "="*70)
    print("REFERENCE FRAME CONVERSION")
    print("="*70)

    # Get transform
    T_robot_from_camera = data['T_robot_from_camera']
    T_camera_from_robot = np.linalg.inv(T_robot_from_camera)

    print("\nTransform from camera to robot:")
    print(T_robot_from_camera)

    # Convert first position from robot to camera
    pos_robot = data['object_pos_in_robot'][0]
    pos_camera_stored = data['object_pos_in_camera'][0]

    # Manual conversion
    pos_robot_hom = np.append(pos_robot, 1)
    pos_camera_hom = T_camera_from_robot @ pos_robot_hom
    pos_camera_computed = pos_camera_hom[:3]

    print(f"\nPosition conversion verification:")
    print(f"  Robot frame: {pos_robot}")
    print(f"  Camera (stored): {pos_camera_stored}")
    print(f"  Camera (computed): {pos_camera_computed}")
    print(f"  Match: {np.allclose(pos_camera_stored, pos_camera_computed)}")


def example_usage():
    """Complete example of using the tracking data."""

    # Path to tracking data (UPDATE THIS to your actual path)
    tracking_dir = "/home/epon04yc/experimental_phantomn_touch_collection/threeD_tracking_offline"
    episode_num = 1

    print("="*70)
    print("3D TRACKING DATA LOADER - NEW FORMAT")
    print("="*70)
    print("\nThis example demonstrates the new unambiguous key names")
    print("that clearly indicate reference frames (camera vs robot).")

    # Load the data
    try:
        data = load_episode_tracking(episode_num, tracking_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease update TRACKING_DIR in the script to point to your data.")
        return

    # Print summary
    print_tracking_summary(data)

    # Demonstrate key usage
    demonstrate_key_usage(data)

    # Filtering
    demonstrate_filtering(data)

    # Trajectory analysis
    demonstrate_trajectory_analysis(data)

    # Frame conversion
    demonstrate_frame_conversion(data)

    # Close file
    data.close()

    print("\n" + "="*70)
    print("QUICK REFERENCE")
    print("="*70)
    print("\nMain outputs (use these for analysis):")
    print("  • object_pos_in_robot    - Object positions (robot frame)")
    print("  • R_object_in_robot      - Object orientations (robot frame)")
    print("  • T_robot_from_cad       - Full transforms for visualization")
    print("  • icp_fitness            - Quality scores for filtering")

    print("\n" + "="*70)
    print("CAD ORIENTATION DEPENDENCY - IMPORTANT!")
    print("="*70)
    print("\nRotation matrices depend on CAD file's original orientation!")
    print("\n✓ CAD-orientation-INDEPENDENT:")
    print("  • object_pos_in_robot    - Only tracks centroid")
    print("  • T_robot_from_cad       - When APPLIED to centered CAD")

    print("\n⚠ DEPENDS on CAD file orientation:")
    print("  • R_icp_in_centered_space")
    print("  • R_object_in_robot")

    print("\nFor rotation analysis independent of CAD:")
    print("  R_relative[i] = R_object_in_robot[i] @ R_object_in_robot[0].T")
    print("  (Computes rotation from frame 0 to frame i)")

    print("\nSee README_DATA_FORMAT.md for complete documentation.")


if __name__ == "__main__":
    example_usage()
