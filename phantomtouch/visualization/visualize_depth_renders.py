#!/usr/bin/env python3
"""
Visualize depth patch rendering results from MuJoCo.
Creates comparison images and animations for verification.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as iio
from pathlib import Path
from tqdm import tqdm

# Configuration
RENDER_DIR = "/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0/mujoco_depth_renders"
OUTPUT_VIS_DIR = os.path.join(RENDER_DIR, "visualizations")

def load_manifest(render_dir):
    """Load rendering manifest."""
    manifest_path = os.path.join(render_dir, "render_manifest.npz")
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None

    manifest = np.load(manifest_path, allow_pickle=True)
    print(f"Loaded manifest with {len(manifest['frame_idx'])} entries")
    return manifest

def visualize_depth_colormap(depth_mm, vmin=None, vmax=None):
    """Convert depth to colored visualization."""
    depth_m = depth_mm.astype(np.float32) / 1000.0

    if vmin is None:
        vmin = np.percentile(depth_m[depth_m > 0], 1)
    if vmax is None:
        vmax = np.percentile(depth_m[depth_m > 0], 99)

    # Normalize and apply colormap
    depth_norm = np.clip((depth_m - vmin) / (vmax - vmin), 0, 1)
    depth_colored = plt.cm.turbo(depth_norm)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)

    return depth_colored

def create_comparison_grid(render_dir, frame_idx, output_path):
    """Create a grid showing all cameras for a single frame."""
    # Get all images for this frame
    rgb_files = {}
    depth_files = {}

    for fname in os.listdir(render_dir):
        if fname.startswith(f"frame_{frame_idx:04d}"):
            if "_rgb.png" in fname:
                cam_name = fname.split("_")[2]
                rgb_files[cam_name] = os.path.join(render_dir, fname)
            elif "_depth.png" in fname:
                cam_name = fname.split("_")[2]
                depth_files[cam_name] = os.path.join(render_dir, fname)

    if not rgb_files:
        print(f"No images found for frame {frame_idx}")
        return

    # Load images
    n_cams = len(rgb_files)
    fig, axes = plt.subplots(2, n_cams, figsize=(5*n_cams, 10))

    if n_cams == 1:
        axes = axes.reshape(2, 1)

    for col, cam_name in enumerate(sorted(rgb_files.keys())):
        # RGB
        rgb = iio.imread(rgb_files[cam_name])
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{cam_name} - RGB", fontsize=12)
        axes[0, col].axis('off')

        # Depth
        depth_mm = iio.imread(depth_files[cam_name])
        depth_colored = visualize_depth_colormap(depth_mm)
        axes[1, col].imshow(depth_colored)
        axes[1, col].set_title(f"{cam_name} - Depth", fontsize=12)
        axes[1, col].axis('off')

    plt.suptitle(f"Frame {frame_idx}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_video_sequence(render_dir, camera_name, output_path, max_frames=100):
    """Create a video from rendered frames."""
    frames = []

    for frame_idx in range(max_frames):
        rgb_file = os.path.join(render_dir, f"frame_{frame_idx:04d}_{camera_name}_rgb.png")
        if not os.path.exists(rgb_file):
            break

        rgb = iio.imread(rgb_file)
        frames.append(rgb)

    if not frames:
        print(f"No frames found for camera {camera_name}")
        return

    print(f"Creating video with {len(frames)} frames...")
    iio.mimsave(output_path, frames, fps=10)
    print(f"Saved: {output_path}")

def analyze_depth_statistics(render_dir, manifest):
    """Analyze depth statistics across all frames."""
    print("\nAnalyzing depth statistics...")

    cameras = np.unique(manifest['camera_name'])
    stats = {cam: {'min': [], 'max': [], 'mean': []} for cam in cameras}

    for i, cam in enumerate(tqdm(manifest['camera_name'], desc="Processing frames")):
        depth_file = os.path.join(render_dir, manifest['depth_path'][i])
        if not os.path.exists(depth_file):
            continue

        depth_mm = iio.imread(depth_file)
        depth_m = depth_mm.astype(np.float32) / 1000.0
        valid = depth_m[depth_m > 0]

        if len(valid) > 0:
            stats[cam]['min'].append(valid.min())
            stats[cam]['max'].append(valid.max())
            stats[cam]['mean'].append(valid.mean())

    # Plot statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for cam in cameras:
        if not stats[cam]['min']:
            continue

        frames = np.arange(len(stats[cam]['min']))

        axes[0].plot(frames, stats[cam]['min'], label=cam, linewidth=2)
        axes[0].set_title('Minimum Depth', fontsize=12)
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Depth (m)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(frames, stats[cam]['max'], label=cam, linewidth=2)
        axes[1].set_title('Maximum Depth', fontsize=12)
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Depth (m)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(frames, stats[cam]['mean'], label=cam, linewidth=2)
        axes[2].set_title('Mean Depth', fontsize=12)
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('Depth (m)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_VIS_DIR, "depth_statistics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("="*60)
    print("Depth Patch Rendering Visualization")
    print("="*60)

    if not os.path.exists(RENDER_DIR):
        print(f"Render directory not found: {RENDER_DIR}")
        print("Run render_depth_patches_phantom.py first!")
        return

    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_VIS_DIR}\n")

    # Load manifest
    manifest = load_manifest(RENDER_DIR)
    if manifest is None:
        return

    # Create comparison grids for first, middle, and last frames
    print("\nCreating comparison grids...")
    n_unique_frames = len(np.unique(manifest['frame_idx']))
    sample_frames = [0, n_unique_frames // 2, n_unique_frames - 1]

    for frame_idx in sample_frames:
        output_path = os.path.join(OUTPUT_VIS_DIR, f"comparison_frame_{frame_idx:04d}.png")
        create_comparison_grid(RENDER_DIR, frame_idx, output_path)
        print(f"Created: {output_path}")

    # Create videos
    print("\nCreating videos...")
    cameras = np.unique(manifest['camera_name'])
    for cam in cameras:
        output_path = os.path.join(OUTPUT_VIS_DIR, f"video_{cam}.mp4")
        create_video_sequence(RENDER_DIR, cam, output_path, max_frames=n_unique_frames)

    # Analyze statistics
    analyze_depth_statistics(RENDER_DIR, manifest)

    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"Output directory: {OUTPUT_VIS_DIR}")
    print("\nGenerated files:")
    print("  - comparison_frame_*.png (sample frame comparisons)")
    print("  - video_*.mp4 (video sequences per camera)")
    print("  - depth_statistics.png (depth analysis)")
    print("\nTo view:")
    print(f"  scp -r user@server:{OUTPUT_VIS_DIR} ./local_dir/")
    print("="*60)

if __name__ == "__main__":
    main()
