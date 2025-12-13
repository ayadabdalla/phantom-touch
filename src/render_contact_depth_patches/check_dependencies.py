#!/usr/bin/env python3
"""
Check dependencies and setup for depth patch rendering pipeline.
"""

import os
import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a Python module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError:
        print(f"  ✗ {package_name} - Install with: pip install {package_name}")
        return False

def check_file(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description}")
        print(f"     Path: {filepath}")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists."""
    if os.path.isdir(dirpath):
        print(f"  ✓ {description}")
        return True
    else:
        print(f"  ✗ {description}")
        print(f"     Path: {dirpath}")
        return False

def main():
    print("="*70)
    print("Phantom-Touch Depth Rendering Pipeline - Dependency Check")
    print("="*70)

    all_ok = True

    # Python packages
    print("\n1. Python Packages:")
    packages = [
        ("numpy", None),
        ("mujoco", None),
        ("imageio", None),
        ("matplotlib", None),
        ("cv2", "opencv-python"),
        ("open3d", None),
        ("trimesh", None),
        ("omegaconf", None),
        ("tqdm", None),
    ]

    for module, package in packages:
        if not check_import(module, package):
            all_ok = False

    # OnTouch repository
    print("\n2. OnTouch Repository:")
    ontouch_paths = [
        ("/home/epon04yc/ontouch", "OnTouch root"),
        ("/home/epon04yc/ontouch/depth_patch_renderer_session", "Renderer session"),
        ("/home/epon04yc/ontouch/depth_patch_renderer_session/data/fr3_simple_pick_up_digit_hand_wsensor/model.xml", "MuJoCo scene"),
        ("/home/epon04yc/ontouch/calibration/data/orbbec/robotbase_camera_transform_orbbec_fr4.npy", "Orbbec extrinsics"),
    ]

    for path, desc in ontouch_paths:
        if os.path.isdir(path):
            if not check_directory(path, desc):
                all_ok = False
        else:
            if not check_file(path, desc):
                all_ok = False

    # Phantom-Touch dataset
    print("\n3. Phantom-Touch Dataset:")
    phantom_paths = [
        ("/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0", "Dataset root"),
        ("/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0/sam2-vid_output", "Tracking output"),
    ]

    for path, desc in phantom_paths:
        if not check_directory(path, desc):
            all_ok = False
            print(f"     NOTE: This will be created if it doesn't exist")

    # Tracking results (optional)
    print("\n4. Tracking Results (Required):")
    tracking_files = [
        ("/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0/sam2-vid_output/absolute_positions.npy", "Object positions"),
        ("/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0/sam2-vid_output/absolute_orientations.npy", "Object orientations"),
    ]

    tracking_ok = True
    for path, desc in tracking_files:
        if not check_file(path, desc):
            tracking_ok = False
            all_ok = False

    if not tracking_ok:
        print("     → Run: python threeD_tracking_offline.py")

    # Banana model (optional - will be created)
    print("\n5. Banana CAD Model (Optional):")
    banana_path = "/mnt/dataset_drive/ayad/phantom-touch/data/cad_models/banana_scaled.obj"
    if check_file(banana_path, "Banana model"):
        pass
    else:
        print("     → Will be created by: python download_banana_model.py")

    # Scripts
    print("\n6. Pipeline Scripts:")
    script_dir = "/home/epon04yc/phantom-touch/src/phantom-touch/scripts"
    scripts = [
        "render_depth_patches_phantom.py",
        "update_mujoco_scene_for_banana.py",
        "visualize_depth_renders.py",
        "phantom_to_ontouch_pipeline.sh",
    ]

    for script in scripts:
        script_path = os.path.join(script_dir, script)
        if not check_file(script_path, script):
            all_ok = False

    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✓ All dependencies satisfied!")
        print("\nYou're ready to run the pipeline:")
        print("  cd /home/epon04yc/phantom-touch/src/phantom-touch/scripts")
        print("  ./phantom_to_ontouch_pipeline.sh")
    else:
        print("✗ Some dependencies are missing.")
        print("\nPlease install missing packages and ensure paths are correct.")
        print("\nFor Python packages:")
        print("  pip install numpy mujoco imageio matplotlib opencv-python open3d trimesh omegaconf tqdm")
    print("="*70)

    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
