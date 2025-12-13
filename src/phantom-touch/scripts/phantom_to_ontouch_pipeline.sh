#!/bin/bash
# Complete pipeline: Phantom-Touch dataset to OnTouch depth patch rendering
# This integrates object trajectories, robot trajectories, and MuJoCo rendering

set -e  # Exit on error

echo "=========================================="
echo "Phantom-Touch to OnTouch Integration Pipeline"
echo "=========================================="
echo ""

# Step 1: Ensure banana model exists
echo "Step 1: Checking banana CAD model..."
BANANA_MODEL="/mnt/dataset_drive/ayad/phantom-touch/data/cad_models/banana_scaled.obj"

if [ ! -f "$BANANA_MODEL" ]; then
    echo "  Banana model not found. Creating..."
    python download_banana_model.py
    echo "  ✓ Banana model created"
else
    echo "  ✓ Banana model exists"
fi

echo ""
echo "=========================================="

# Step 2: Create/update MuJoCo scene
echo "Step 2: Setting up MuJoCo scene for banana..."
python update_mujoco_scene_for_banana.py

echo "  ✓ MuJoCo scene configured"
echo ""
echo "=========================================="

# Step 3: Check if tracking has been done
echo "Step 3: Checking object tracking results..."
TRACKING_DIR="/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0/sam2-vid_output"

if [ ! -f "$TRACKING_DIR/absolute_positions.npy" ]; then
    echo "  WARNING: Object tracking not found!"
    echo "  Run 3D tracking first: python threeD_tracking_offline.py"
    echo ""
    read -p "  Continue anyway with dummy data? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Exiting. Please run tracking first."
        exit 1
    fi
else
    echo "  ✓ Object tracking results found"
fi

echo ""
echo "=========================================="

# Step 4: Render depth patches
echo "Step 4: Rendering depth patches with MuJoCo..."
echo "  This will:"
echo "  - Load object poses from tracking"
echo "  - Load robot trajectories (if available)"
echo "  - Render RGB and depth from all cameras"
echo "  - Save results with manifest"
echo ""

python render_depth_patches_phantom.py

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Output directory: {phantom_dataset_root}/mujoco_depth_renders/"
echo ""
echo "Generated files:"
echo "  - frame_XXXX_[camera]_rgb.png (RGB images)"
echo "  - frame_XXXX_[camera]_depth.png (Depth in mm as uint16)"
echo "  - render_manifest.npz (metadata and tracking info)"
echo ""
echo "Cameras rendered:"
echo "  - cam_left_digit (left DIGIT tactile)"
echo "  - cam_right_digit (right DIGIT tactile)"
echo "  - orbbec (main RGB-D camera)"
echo ""
echo "Next steps:"
echo "  1. Visualize results: python visualize_depth_renders.py"
echo "  2. Create OnTouch dataset: python create_ontouch_dataset.py"
echo ""
