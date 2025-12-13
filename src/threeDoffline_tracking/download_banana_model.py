#!/usr/bin/env python3
"""
Download and prepare a banana CAD model with real dimensions.
This script downloads a banana model, scales it to realistic dimensions,
and saves it for use in the 3D tracking pipeline.
"""

import os
import trimesh
import numpy as np
import requests
from pathlib import Path

# Output configuration
OUTPUT_DIR = "/mnt/dataset_drive/ayad/phantom-touch/data/cad_models"
OUTPUT_FILENAME = "banana_scaled.obj"

# Real banana dimensions (in meters)
BANANA_LENGTH = 0.19  # 19 cm - typical banana length
BANANA_DIAMETER = 0.035  # 3.5 cm - typical diameter

# Method 1: Download from a public URL (if available)
# Common free model sources - you may need to manually download if auth is required
BANANA_MODEL_URLS = [
    # These are example URLs - you'll need to find actual direct download links
    # Or manually download from the websites I mentioned earlier
]

def download_model_from_url(url, save_path):
    """Download a 3D model from URL."""
    print(f"Downloading model from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded to {save_path}")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def create_simple_banana_mesh():
    """
    Create a simple procedural banana mesh using parametric equations.
    This is a fallback if no model can be downloaded.
    """
    print("Creating procedural banana model...")

    # Create a curved cylinder to approximate a banana
    # Banana curve approximation using bezier-like curve
    segments = 50
    radius_base = BANANA_DIAMETER / 2

    vertices = []
    faces = []

    # Create vertices along the banana curve
    for i in range(segments):
        t = i / (segments - 1)

        # Banana curve (approximation)
        x = BANANA_LENGTH * t
        y = 0.03 * np.sin(np.pi * t)  # Slight curve
        z = -0.02 * (t - 0.5)**2  # Slight bend

        # Radius tapering at ends
        radius = radius_base * (1 - 0.3 * abs(2*t - 1))

        # Create circular cross-section
        n_sides = 12
        for j in range(n_sides):
            angle = 2 * np.pi * j / n_sides
            vx = x
            vy = y + radius * np.cos(angle)
            vz = z + radius * np.sin(angle)
            vertices.append([vx, vy, vz])

    vertices = np.array(vertices)

    # Create faces
    for i in range(segments - 1):
        for j in range(n_sides):
            # Current ring
            v0 = i * n_sides + j
            v1 = i * n_sides + (j + 1) % n_sides
            # Next ring
            v2 = (i + 1) * n_sides + (j + 1) % n_sides
            v3 = (i + 1) * n_sides + j

            # Two triangles per quad
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])

    faces = np.array(faces)

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Smooth the mesh
    trimesh.smoothing.filter_laplacian(mesh, iterations=3)

    return mesh

def scale_mesh_to_banana_dimensions(mesh):
    """Scale a mesh to realistic banana dimensions."""
    print("Scaling mesh to real banana dimensions...")

    # Get current bounding box
    bbox = mesh.bounds
    current_size = bbox[1] - bbox[0]

    # Determine which axis is the length (usually the longest)
    length_axis = np.argmax(current_size)
    current_length = current_size[length_axis]

    print(f"Current dimensions: {current_size}")
    print(f"Length axis: {['X', 'Y', 'Z'][length_axis]}")
    print(f"Current length: {current_length:.4f} m")

    # Scale to target length
    scale_factor = BANANA_LENGTH / current_length
    mesh.apply_scale(scale_factor)

    # Verify new dimensions
    new_bbox = mesh.bounds
    new_size = new_bbox[1] - new_bbox[0]
    print(f"Scaled dimensions: {new_size}")
    print(f"Scale factor: {scale_factor:.4f}")

    return mesh

def center_mesh(mesh):
    """Center the mesh at origin."""
    centroid = mesh.centroid
    mesh.vertices -= centroid
    print(f"Centered mesh (was at {centroid})")
    return mesh

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    print("=" * 60)
    print("Banana CAD Model Preparation Script")
    print("=" * 60)

    # Option 1: Try to download from URL
    model_downloaded = False
    temp_download_path = "/tmp/banana_download.obj"

    for url in BANANA_MODEL_URLS:
        if download_model_from_url(url, temp_download_path):
            try:
                mesh = trimesh.load(temp_download_path)
                model_downloaded = True
                print("Successfully loaded downloaded model")
                break
            except Exception as e:
                print(f"Failed to load downloaded model: {e}")

    # Option 2: Check if user has manually placed a model
    if not model_downloaded:
        manual_paths = [
            "/tmp/banana.obj",
            "/tmp/banana.stl",
            os.path.join(OUTPUT_DIR, "banana_original.obj"),
        ]

        for manual_path in manual_paths:
            if os.path.exists(manual_path):
                print(f"\nFound manually placed model at {manual_path}")
                try:
                    mesh = trimesh.load(manual_path)
                    model_downloaded = True
                    print("Successfully loaded manual model")
                    break
                except Exception as e:
                    print(f"Failed to load: {e}")

    # Option 3: Create procedural banana
    if not model_downloaded:
        print("\nNo downloaded model available. Creating procedural banana...")
        mesh = create_simple_banana_mesh()
    else:
        # Scale downloaded/manual model
        mesh = scale_mesh_to_banana_dimensions(mesh)

    # Center the mesh at origin
    mesh = center_mesh(mesh)

    # Save the prepared model
    print(f"\nSaving prepared banana model to {output_path}...")
    mesh.export(output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print(f"Model saved to: {output_path}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces: {len(mesh.faces)}")
    print(f"Bounds: {mesh.bounds}")
    print(f"Extents: {mesh.extents}")
    print(f"Volume: {mesh.volume:.6f} m³")

    # Save info file
    info_path = os.path.join(OUTPUT_DIR, "banana_model_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"Banana CAD Model Information\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model path: {output_path}\n")
        f.write(f"Target length: {BANANA_LENGTH} m ({BANANA_LENGTH*100} cm)\n")
        f.write(f"Target diameter: {BANANA_DIAMETER} m ({BANANA_DIAMETER*100} cm)\n")
        f.write(f"Actual dimensions: {mesh.extents}\n")
        f.write(f"Vertices: {len(mesh.vertices)}\n")
        f.write(f"Faces: {len(mesh.faces)}\n")
        f.write(f"Volume: {mesh.volume:.6f} m³\n")

    print(f"\nModel info saved to: {info_path}")
    print("\n" + "=" * 60)
    print("To use this model, update your tracking script:")
    print(f"CAD_MODEL_PATH = '{output_path}'")
    print("=" * 60)

    return output_path

if __name__ == "__main__":
    main()
