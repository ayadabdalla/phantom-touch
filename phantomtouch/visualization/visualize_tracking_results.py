#!/usr/bin/env python3
"""
Visualize 3D tracking results by rendering point clouds and saving images.
Works in SSH environment by saving visualization images instead of displaying.
"""

import os
import sys
import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
RESULTS_DIR = "/home/epon04yc/phantom-touch/src/phantom-touch/scripts/temporary_images/temporary_masks"  # Update if needed
OUTPUT_VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")

def load_results(results_dir):
    """Load all tracking results."""
    print(f"Loading results from {results_dir}...")

    results = {}
    results['transforms'] = np.load(os.path.join(results_dir, "cad_to_observation_transforms.npy"))
    results['rotations'] = np.load(os.path.join(results_dir, "icp_rotations.npy"))
    results['orientations'] = np.load(os.path.join(results_dir, "absolute_orientations.npy"))
    results['positions'] = np.load(os.path.join(results_dir, "absolute_positions.npy"))
    results['translations'] = np.load(os.path.join(results_dir, "translations_from_initial.npy"))
    results['cad_info'] = np.load(os.path.join(results_dir, "cad_model_info.npy"), allow_pickle=True).item()

    print(f"Loaded {len(results['transforms'])} frames")
    return results

def render_point_cloud_to_image(pcd, filename, width=1920, height=1080):
    """Render a point cloud to an image file using Open3D off-screen rendering."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    vis.add_geometry(pcd)

    # Set camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    # Update and render
    vis.poll_events()
    vis.update_renderer()

    # Save image
    vis.capture_screen_image(filename, do_render=True)
    vis.destroy_window()

    print(f"Saved: {filename}")

def visualize_trajectory(positions, output_path):
    """Plot the trajectory of the object centroid."""
    positions = np.array(positions)

    fig = plt.figure(figsize=(15, 5))

    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    # X-Y projection
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
    ax2.scatter(positions[0, 0], positions[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('X-Y Projection')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')

    # Position over time
    ax3 = fig.add_subplot(133)
    frames = np.arange(len(positions))
    ax3.plot(frames, positions[:, 0], 'r-', label='X')
    ax3.plot(frames, positions[:, 1], 'g-', label='Y')
    ax3.plot(frames, positions[:, 2], 'b-', label='Z')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position Components Over Time')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved trajectory plot: {output_path}")

def visualize_orientation(orientations, output_path):
    """Visualize orientation changes using Euler angles."""
    from scipy.spatial.transform import Rotation

    orientations = np.array(orientations)
    euler_angles = []

    for orientation in orientations:
        r = Rotation.from_matrix(orientation)
        euler = r.as_euler('xyz', degrees=True)
        euler_angles.append(euler)

    euler_angles = np.array(euler_angles)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    frames = np.arange(len(euler_angles))

    labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
    colors = ['r', 'g', 'b']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(frames, euler_angles[:, i], color=color, linewidth=2)
        ax.set_ylabel(f'{label} (deg)')
        ax.set_title(f'Orientation: {label}')
        ax.grid(True)

    axes[-1].set_xlabel('Frame')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved orientation plot: {output_path}")

def create_combined_visualization(cad_mesh, transforms, output_dir, max_frames=10):
    """Create visualizations showing CAD model aligned with observations."""
    os.makedirs(output_dir, exist_ok=True)

    # Sample frames to visualize
    n_frames = len(transforms)
    frame_indices = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=int)

    print(f"\nCreating combined visualizations for {len(frame_indices)} frames...")

    for idx, frame_idx in enumerate(frame_indices):
        # Load CAD model
        cad_pcd = o3d.geometry.PointCloud()
        cad_points = cad_mesh.sample(10000)
        cad_pcd.points = o3d.utility.Vector3dVector(cad_points)

        # Transform CAD model to observation pose
        cad_pcd.transform(transforms[frame_idx])
        cad_pcd.paint_uniform_color([1, 0, 0])  # Red

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Create visualization
        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = cad_pcd.points
        combined_pcd.colors = cad_pcd.colors

        # Render to image
        output_file = os.path.join(output_dir, f"frame_{frame_idx:04d}_cad_aligned.png")

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, visible=False)
        vis.add_geometry(combined_pcd)
        vis.add_geometry(coord_frame)

        # Set camera parameters
        ctr = vis.get_view_control()
        ctr.set_zoom(0.6)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_file, do_render=True)
        vis.destroy_window()

        print(f"  Rendered frame {frame_idx}/{n_frames}: {output_file}")

def main():
    print("=" * 60)
    print("3D Tracking Results Visualization")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    print(f"Saving visualizations to: {OUTPUT_VIS_DIR}\n")

    # Load results
    try:
        results = load_results(RESULTS_DIR)
    except Exception as e:
        print(f"Error loading results: {e}")
        print(f"Make sure tracking has been run and results are in: {RESULTS_DIR}")
        return

    # Load CAD model
    cad_model_path = results['cad_info']['model_path']
    print(f"\nLoading CAD model from: {cad_model_path}")
    cad_mesh = trimesh.load(cad_model_path)
    print(f"CAD model loaded: {len(cad_mesh.vertices)} vertices")

    # Visualize trajectory
    print("\n" + "=" * 60)
    print("Generating trajectory visualization...")
    print("=" * 60)
    trajectory_path = os.path.join(OUTPUT_VIS_DIR, "trajectory.png")
    visualize_trajectory(results['positions'], trajectory_path)

    # Visualize orientation
    print("\n" + "=" * 60)
    print("Generating orientation visualization...")
    print("=" * 60)
    orientation_path = os.path.join(OUTPUT_VIS_DIR, "orientation.png")
    visualize_orientation(results['orientations'], orientation_path)

    # Create combined visualizations
    print("\n" + "=" * 60)
    print("Generating 3D visualizations...")
    print("=" * 60)
    viz_3d_dir = os.path.join(OUTPUT_VIS_DIR, "3d_renders")
    create_combined_visualization(cad_mesh, results['transforms'], viz_3d_dir, max_frames=10)

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    positions = results['positions']
    print(f"Number of frames: {len(positions)}")
    print(f"Position range:")
    print(f"  X: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}] m")
    print(f"  Y: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] m")
    print(f"  Z: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}] m")

    total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    print(f"Total distance traveled: {total_distance:.4f} m")

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_VIS_DIR}")
    print("Files created:")
    print(f"  - trajectory.png")
    print(f"  - orientation.png")
    print(f"  - 3d_renders/ (multiple frames)")
    print("\nYou can download these images to view on your local machine:")
    print(f"  scp -r user@server:{OUTPUT_VIS_DIR} ./local_directory/")

if __name__ == "__main__":
    main()
