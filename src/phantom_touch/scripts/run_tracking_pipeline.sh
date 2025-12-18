#!/bin/bash
# Complete 3D tracking pipeline runner
# This script runs the full tracking pipeline and generates visualizations

set -e  # Exit on error

echo "=========================================="
echo "3D Object Tracking Pipeline"
echo "=========================================="
echo ""

# Step 1: Check if banana model exists, if not create it
BANANA_MODEL="/mnt/dataset_drive/ayad/phantom-touch/data/cad_models/banana_scaled.obj"

if [ ! -f "$BANANA_MODEL" ]; then
    echo "Step 1: Creating banana CAD model..."
    python download_banana_model.py
    echo "✓ Banana model created"
else
    echo "Step 1: Banana model already exists at $BANANA_MODEL"
fi

echo ""
echo "=========================================="

# Step 2: Run 3D tracking
echo "Step 2: Running 3D tracking..."
echo "This will:"
echo "  - Load depth data and masks"
echo "  - Generate point clouds"
echo "  - Apply ICP with CAD model"
echo "  - Track position and orientation"
echo ""

python threeD_tracking_offline.py

echo "✓ Tracking complete"
echo ""
echo "=========================================="

# Step 3: Generate detailed visualizations
echo "Step 3: Generating detailed visualizations..."
python visualize_tracking_results.py

echo "✓ Visualizations complete"
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results location:"
echo "  Output directory: Check the script output above for the exact path"
echo ""
echo "Generated files:"
echo "  1. Tracking data (.npy files)"
echo "  2. trajectory_preview.png (quick preview)"
echo "  3. visualizations/trajectory.png (detailed trajectory)"
echo "  4. visualizations/orientation.png (orientation analysis)"
echo "  5. visualizations/3d_renders/ (3D point cloud renders)"
echo ""
echo "To download visualizations to your local machine:"
echo "  scp -r user@server:/path/to/output/visualizations ./local_dir/"
echo ""
