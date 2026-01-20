# 3D Tracking Data Format

## Overview
All 3D tracking data is saved as **one NPZ file per episode** with clear, unambiguous key names that indicate reference frames.

## File Naming
```
episode_01_tracking.npz
episode_02_tracking.npz
episode_03_tracking.npz
...
```

## Reference Frames

Understanding reference frames is **critical** for using this data correctly:

### 1. **CAMERA FRAME**
- Origin: Camera optical center
- Axes: Standard camera convention (x right, y down, z forward)
- Used for: Intermediate calculations, ICP alignment

### 2. **ROBOT FRAME**
- Origin: Robot base
- Axes: Robot's world coordinate system
- Used for: **All final outputs** (main data you'll use)

### 3. **CENTERED CAD FRAME**
- Origin: CAD model centroid at (0,0,0)
- Used for: ICP alignment (CAD and observations both centered)

### 4. **CENTERED SPACE**
- Both CAD and observation centered at origin for ICP
- Rotations from ICP are in this space

## Data Structure

Each NPZ file contains the following keys:

### 🎯 Main Output Data (ROBOT FRAME)

#### Transforms
- **`T_robot_from_cad`**: `(N, 4, 4)` - Transformation matrices from centered CAD model to robot frame
  - Use this to visualize the CAD model at each tracked pose
  - `T @ centered_cad_points` gives you points in robot frame

#### Positions
- **`object_pos_in_robot`**: `(N, 3)` - Object centroid position in robot frame
  - **Primary position output** - use this for trajectories, analysis, etc.
  - In meters, robot world coordinates

#### Rotations
- **`R_object_in_robot`**: `(N, 3, 3)` - Object orientation in robot frame
  - **Primary rotation output** - use this for object orientation
  - 3×3 rotation matrix per frame

#### Trajectory
- **`displacement_from_start`**: `(N, 3)` - Displacement from first tracked position
  - Relative motion from start of tracking
  - In robot frame, meters

### 📊 Intermediate Data

#### Positions (Camera Frame)
- **`object_pos_in_camera`**: `(N, 3)` - Object centroid in camera frame
  - Intermediate data before transformation to robot frame

#### Rotations (Centered Space)
- **`R_icp_in_centered_space`**: `(N, 3, 3)` - ICP rotation result
  - Rotation from centered CAD → centered observation
  - In the centered alignment space

### 📈 Quality Metrics
- **`icp_fitness`**: `(N,)` - ICP fitness score per frame (0-1, higher is better)
- **`frame_idx`**: `(N,)` - Original frame indices in the episode

### 🔧 CAD Model Information
- **`cad_model_path`**: String - Path to CAD model file
- **`cad_num_sample_points`**: Integer - Number of points sampled from CAD
- **`cad_centroid_in_cad_frame`**: `(3,)` - CAD centroid (should be ~0 since centered)
- **`cad_scale_factor`**: Float - Scale factor applied to CAD (typically 0.001)

### 🔄 Reference Transforms
- **`T_robot_from_camera`**: `(4, 4)` - Transform from camera frame to robot frame
  - Fixed calibration transform
  - Same for all frames in episode
- **`first_pos_robot`**: `(3,)` - First tracked position in robot frame
  - Reference point for `displacement_from_start`

### 📝 Metadata
- **`episode_number`**: Integer - Episode number
- **`num_frames_tracked`**: Integer - Number of successfully tracked frames
- **`num_frames_total`**: Integer - Total frames in episode

## Usage Examples

### Basic Loading

```python
import numpy as np

# Load tracking data
data = np.load('episode_01_tracking.npz', allow_pickle=True)

# Get main outputs (robot frame)
positions = data['object_pos_in_robot']  # (N, 3)
rotations = data['R_object_in_robot']    # (N, 3, 3)
transforms = data['T_robot_from_cad']    # (N, 4, 4)
fitness = data['icp_fitness']            # (N,)
frame_numbers = data['frame_idx']        # (N,)

print(f"Tracked {len(positions)} frames")
print(f"Average fitness: {fitness.mean():.3f}")
```

### Accessing Specific Frame

```python
# Get data for frame index 42
frame_to_find = 42
mask = data['frame_idx'] == frame_to_find

if mask.any():
    idx = np.where(mask)[0][0]
    position = data['object_pos_in_robot'][idx]
    rotation = data['R_object_in_robot'][idx]
    transform = data['T_robot_from_cad'][idx]

    print(f"Frame {frame_to_find}:")
    print(f"  Position: {position}")
    print(f"  Fitness: {data['icp_fitness'][idx]:.3f}")
```

### Filtering by Quality

```python
# Get only high-quality tracking results
quality_threshold = 0.7
good_mask = data['icp_fitness'] > quality_threshold

good_positions = data['object_pos_in_robot'][good_mask]
good_frames = data['frame_idx'][good_mask]

print(f"High quality frames: {len(good_positions)}/{len(data['frame_idx'])}")
```

### Visualizing CAD Model at Tracked Poses

```python
import trimesh

# Load CAD model (must match the one used for tracking)
cad_mesh = trimesh.load(str(data['cad_model_path']))

# Center and scale CAD (same as pipeline)
cad_mesh.vertices -= cad_mesh.vertices.mean(axis=0)
cad_mesh.vertices *= float(data['cad_scale_factor'])

# Transform to frame 10's pose in robot frame
frame_idx = 10
T = data['T_robot_from_cad'][frame_idx]

# Apply transform to CAD mesh
cad_at_pose = cad_mesh.copy()
cad_at_pose.apply_transform(T)

# Now cad_at_pose is in robot frame at the tracked position/orientation
```

### Trajectory Analysis

```python
# Compute trajectory statistics
positions = data['object_pos_in_robot']

# Distance traveled
distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
total_distance = distances.sum()
print(f"Total distance traveled: {total_distance:.3f} m")

# Velocity (if you know frame rate)
fps = 30  # example
velocities = distances * fps
avg_velocity = velocities.mean()
print(f"Average velocity: {avg_velocity:.3f} m/s")

# Position range
print(f"X range: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] m")
print(f"Y range: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] m")
print(f"Z range: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] m")
```

### Converting Between Frames

```python
# If you need camera frame data
T_robot_from_camera = data['T_robot_from_camera']
T_camera_from_robot = np.linalg.inv(T_robot_from_camera)

# Position in robot → position in camera
pos_robot = data['object_pos_in_robot'][0]
pos_robot_hom = np.append(pos_robot, 1)
pos_camera_hom = T_camera_from_robot @ pos_robot_hom
pos_camera = pos_camera_hom[:3]

# Or just use the stored camera positions
pos_camera_direct = data['object_pos_in_camera'][0]
```

## Quick Reference: Which Key to Use?

| Task | Key to Use | Frame |
|------|------------|-------|
| Object position for analysis | `object_pos_in_robot` | Robot |
| Object orientation | `R_object_in_robot` | Robot |
| Visualize CAD at pose | `T_robot_from_cad` | Robot |
| Trajectory tracking | `displacement_from_start` | Robot |
| Filter by quality | `icp_fitness` | N/A |
| Map to frame number | `frame_idx` | N/A |

## Important Notes

✅ **Use `object_pos_in_robot` for positions** - This is in robot world coordinates

✅ **Use `T_robot_from_cad` for visualization** - Apply to centered CAD model

✅ **Each frame is independent** - No temporal tracking between frames

✅ **Bad fitness doesn't propagate** - Each ICP starts from identity

⚠️ **Frame indices may have gaps** - Only successfully tracked frames are saved

⚠️ **`R_icp_in_centered_space` is in camera frame** - Use `R_object_in_robot` for robot frame

## CAD Orientation Dependency

**CRITICAL**: Rotation outputs depend on the CAD model's orientation in its file!

### Background

The CAD model is centered at origin but **keeps its original orientation** from the file. ICP finds the rotation needed to align this CAD to the observation. If you use a differently-oriented CAD file, you'll get different rotation matrices.

### What is CAD-orientation-independent:

| Data | Why Independent |
|------|----------------|
| `object_pos_in_robot` | Only tracks centroid (no rotation) |
| `displacement_from_start` | Position changes over time |
| **`T_robot_from_cad`** | **When applied** to your centered CAD model |

### What depends on CAD file orientation:

| Data | What it represents |
|------|-------------------|
| `R_icp_in_centered_space` | Rotation to align CAD-as-loaded (camera frame) |
| `R_object_in_robot` | Same rotation in robot frame |

### Why `T_robot_from_cad` works regardless of CAD orientation:

Although the rotation matrix values differ for different CAD orientations, when you **apply** the full transform to your centered CAD model, you **always** get the correct observed pose. The rotation compensates for the CAD's initial orientation.

**Example:**
```python
# Two CAD files with different orientations
CAD_A = load_cad("cup_upright.stl")      # Upright in file
CAD_B = load_cad("cup_sideways.stl")     # Rotated 90° in file

# Different transforms (because CAD orientations differ)
T_A = data['T_robot_from_cad'][frame_10]  # If tracked with CAD_A
T_B = data['T_robot_from_cad'][frame_10]  # If tracked with CAD_B

# But SAME final result when applied:
pose_A = T_A @ CAD_A  # Correct observed pose
pose_B = T_B @ CAD_B  # Same observed pose!
```

### How to use rotation data:

**1. For visualization (always correct):**
```python
import trimesh

# Load and prepare CAD (same as pipeline does)
cad = trimesh.load(str(data['cad_model_path']))
cad.vertices -= cad.vertices.mean(axis=0)  # Center
cad.vertices *= float(data['cad_scale_factor'])  # Scale

# Apply transform for frame 10
cad_at_frame_10 = cad.copy()
cad_at_frame_10.apply_transform(data['T_robot_from_cad'][10])

# cad_at_frame_10 is now correctly positioned in robot frame
```

**2. For rotation analysis (independent of CAD initial orientation):**
```python
# Compute relative rotations between frames
R_all = data['R_object_in_robot']

# Rotation from frame 0 to frame i (robot frame)
for i in range(len(R_all)):
    R_relative = R_all[i] @ R_all[0].T
    # CAD's initial orientation cancels out!

# Or rotation from frame i to frame j
R_i_to_j = R_all[j] @ R_all[i].T
```

## See Also
- [load_tracking_example.py](load_tracking_example.py) - Complete examples with code
- [threeD_tracking_offline.py](threeD_tracking_offline.py) - See module docstring for detailed frame documentation
