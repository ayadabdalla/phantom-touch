# Phantom-Touch to OnTouch Depth Patch Rendering Integration

This integration adds MuJoCo-based depth patch rendering from the OnTouch repository to the Phantom-Touch dataset. It uses object trajectories from 3D tracking, robot trajectories, and calibrated camera extrinsics to render realistic depth and RGB images.

## Overview

The pipeline:
1. Uses tracked object poses (position + orientation) from Phantom-Touch
2. Loads robot joint trajectories (if available)
3. Renders depth patches using MuJoCo simulation with:
   - Banana CAD model (instead of strawberry)
   - FR3 robot with DIGIT tactile sensors
   - Calibrated Orbbec camera
4. Saves RGB, depth, and metadata for downstream use

## Files Created

### Core Scripts

| File | Purpose |
|------|---------|
| `render_depth_patches_phantom.py` | Main rendering script |
| `update_mujoco_scene_for_banana.py` | Creates MuJoCo scene with banana |
| `visualize_depth_renders.py` | Visualizes rendering results |
| `phantom_to_ontouch_pipeline.sh` | Complete pipeline runner |

### Configuration

| File | Purpose |
|------|---------|
| `config/depth_patch_renderer_phantom.yaml` | Rendering configuration |
| `mujoco_scene/phantom_banana_scene.xml` | MuJoCo scene (created by script) |

## Prerequisites

1. **3D Object Tracking**: Must run `threeD_tracking_offline.py` first to generate:
   - `absolute_positions.npy`
   - `absolute_orientations.npy`

2. **Banana CAD Model**: Will be created automatically by `download_banana_model.py`

3. **Dependencies**:
   ```bash
   pip install mujoco imageio numpy matplotlib tqdm
   ```

## Quick Start

### Option 1: Run Complete Pipeline

```bash
cd /home/epon04yc/phantom-touch/src/phantom-touch/scripts
./phantom_to_ontouch_pipeline.sh
```

This will:
1. Create banana CAD model (if needed)
2. Setup MuJoCo scene
3. Render depth patches
4. Create visualizations

### Option 2: Run Step-by-Step

```bash
# 1. Create banana model
python download_banana_model.py

# 2. Setup MuJoCo scene
python update_mujoco_scene_for_banana.py

# 3. Render depth patches
python render_depth_patches_phantom.py

# 4. Visualize results
python visualize_depth_renders.py
```

## Configuration

Edit `config/depth_patch_renderer_phantom.yaml` to customize:

```yaml
# Dataset paths
phantom_dataset_root: "/path/to/your/phantom/dataset"
banana_cad_model: "/path/to/banana.obj"

# Camera settings
orbbec_extrinsics: "/path/to/extrinsics.npy"

# Rendering
render_width: 240
render_height: 320
trajectory_rate_hz: 10.0

# Cameras to render
cameras:
  - "cam_left_digit"
  - "cam_right_digit"
  - "orbbec"
```

## Output Structure

```
{phantom_dataset_root}/
└── mujoco_depth_renders/
    ├── frame_0000_orbbec_rgb.png
    ├── frame_0000_orbbec_depth.png
    ├── frame_0000_cam_left_digit_rgb.png
    ├── frame_0000_cam_left_digit_depth.png
    ├── frame_0000_cam_right_digit_rgb.png
    ├── frame_0000_cam_right_digit_depth.png
    ├── ...
    ├── render_manifest.npz
    └── visualizations/
        ├── comparison_frame_0000.png
        ├── video_orbbec.mp4
        ├── video_cam_left_digit.mp4
        ├── video_cam_right_digit.mp4
        └── depth_statistics.png
```

## Manifest Format

The `render_manifest.npz` contains:

```python
{
    'frame_idx': np.array([0, 0, 0, 1, 1, 1, ...]),  # Frame indices
    'camera_name': np.array(['orbbec', 'left', 'right', ...]),  # Camera names
    'rgb_path': np.array(['frame_0000_orbbec_rgb.png', ...]),  # RGB filenames
    'depth_path': np.array(['frame_0000_orbbec_depth.png', ...]),  # Depth filenames
    'object_position': np.array([[x, y, z], ...]),  # Object positions
    'object_orientation': np.array([[[R]], ...]),  # Object orientations (3x3)
}
```

## Depth Format

- **Format**: 16-bit PNG (uint16)
- **Units**: Millimeters
- **Range**: 0-65535 mm (0-65.535 m)
- **Conversion**: `depth_m = depth_png.astype(float) / 1000.0`

## Integration with OnTouch

The rendered depth patches are compatible with OnTouch's `DepthToTouchDataset`:

```python
from datasets.ontouch_dataset import DepthToTouchDataset

dataset = DepthToTouchDataset(
    root_dir="{phantom_dataset_root}/mujoco_depth_renders",
    camera_name="cam_left_digit",  # or "cam_right_digit"
    contact_max_m=0.002,  # 2mm contact threshold
)
```

## Camera Specifications

### Orbbec Femto Bolt
- **Position**: `(0.600, -0.194, 1.201)` m in robot base frame
- **Orientation**: Quaternion `(0.6937, -0.0086, 0.003, -0.7201)`
- **FOV**: 51° (fovy)
- **Extrinsics**: Loaded from `/home/epon04yc/ontouch/calibration/data/orbbec/`

### DIGIT Tactile Cameras
- **Left**: `cam_left_digit` on left finger
- **Right**: `cam_right_digit` on right finger
- **Position**: 80mm behind fingertip
- **FOV**: 60°
- **Purpose**: Tactile contact visualization

## Troubleshooting

### "Object tracking not found"
Run 3D tracking first:
```bash
python threeD_tracking_offline.py
```

### "Banana CAD model not found"
Create the model:
```bash
python download_banana_model.py
```

### "MuJoCo scene error"
Verify scene paths in the config and ensure FR3 XML files are accessible.

### "No robot trajectory found"
This is optional. The system will use a static robot pose if no trajectory is provided.

## Advanced Usage

### Using Custom Object Models

1. Prepare your mesh (OBJ, STL, etc.)
2. Update scene XML in `update_mujoco_scene_for_banana.py`
3. Adjust object mass and inertia properties

### Synchronizing with Robot Trajectories

If you have robot joint states, save them as:
```
{phantom_dataset_root}/robot_joint_states/robot_joint_states.npy
```

Format: `(T, 7)` numpy array with 7 joint angles in radians

### Camera Coordinate Frames

The pipeline assumes:
- **Object poses**: In camera frame (from tracking)
- **Robot poses**: In robot base frame
- **Transform**: `camera_to_world` matrix converts camera → world

Update `camera_to_world` in config if your tracking is in a different frame.

## References

- **OnTouch Repository**: `/home/epon04yc/ontouch/`
- **MuJoCo Docs**: https://mujoco.readthedocs.io/
- **Phantom-Touch**: Paper/repo reference here

## Contact

For issues or questions:
1. Check this README
2. Examine the manifest and visualizations
3. Review OnTouch original implementation

---

Generated for Phantom-Touch dataset integration
