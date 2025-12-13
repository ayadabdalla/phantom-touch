#!/usr/bin/env python3
"""
Update the MuJoCo scene XML to replace strawberry with banana model.
Creates a new scene file specifically for phantom-touch dataset rendering.
"""

import os
import shutil
from pathlib import Path

# Paths
ORIGINAL_SCENE = "/home/epon04yc/ontouch/depth_patch_renderer_session/data/fr3_simple_pick_up_digit_hand_wsensor/model.xml"
PHANTOM_SCENE_DIR = "/home/epon04yc/phantom-touch/src/phantom-touch/scripts/mujoco_scene"
NEW_SCENE_PATH = os.path.join(PHANTOM_SCENE_DIR, "phantom_banana_scene.xml")
BANANA_CAD = "/mnt/dataset_drive/ayad/phantom-touch/data/cad_models/banana_scaled.obj"

# XML template for banana scene
BANANA_SCENE_XML = """<mujoco model="phantom_banana_renderer">
  <!-- Include FR3 robot components from original scene -->
  <include file="{original_scene_dir}/fr3_common.xml" />
  <include file="{original_scene_dir}/fr3_0.xml" />

  <compiler autolimits="true" eulerseq="xyz" />

  <option tolerance="1e-8" gravity="0 0 -9.81" timestep="0.002"/>

  <visual>
    <headlight ambient=".7 .7 .7" diffuse=".2 .2 .2" specular="0.1 0.1 0.1" />
    <rgba haze="0 0 0 0" />
    <map znear="0.001" />
    <scale contactwidth=".02" contactheight=".5" />
    <global azimuth="120" elevation="-20" fovy="4" />
  </visual>

  <statistic center="0.3 0 0.4" extent="1" />

  <default>
    <geom friction="1" solimp="0.9 0.9999 0.0001" />
  </default>

  <asset>
    <!-- Transparent ground material -->
    <material name="groundplane" rgba="0 0 0 0" reflectance="0" />

    <!-- Checker texture for floor -->
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4"
             width="300" height="300" mark="edge" markrgb=".2 .3 .4" />
    <material name="grid" texture="grid" texrepeat="3 1" texuniform="true" />

    <!-- Banana mesh -->
    <mesh name="Banana_mesh" file="{banana_model_path}" scale="1.0 1.0 1.0" />
  </asset>

  <worldbody>
    <!-- Lights -->
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <light pos="1 0 .3" dir="-1 0 -.3" />
    <light pos="-1 0 .3" dir="1 0 -.3" />

    <!-- Invisible collision floor -->
    <geom name="floor_invisible" type="plane" size="0 0 0.05"
          material="groundplane" friction="1 0.005 0.0001" />

    <!-- Cameras -->
    <camera name="orbbec" pos="0.600 -0.194 1.201" quat="0.6937 -0.0086 0.003 -0.7201" fovy="51" />

    <!-- Banana object with free joint for pose control -->
    <body name="Banana" pos="0.4 0 0.05">
      <joint name="banana_free" type="free"/>
      <geom name="Banana_geom" type="mesh" mesh="Banana_mesh"
            rgba="1 0.9 0 1" contype="2" conaffinity="1" />
      <inertial pos="0 0 0" mass="0.12" diaginertia="0.001 0.001 0.001" />
    </body>

    <!-- FR3 robot is included from fr3_0.xml -->
  </worldbody>
</mujoco>
"""

def create_phantom_scene():
    """Create a new MuJoCo scene for phantom-touch rendering."""

    print("="*60)
    print("Creating MuJoCo Scene for Phantom-Touch Banana Rendering")
    print("="*60)

    # Create scene directory
    os.makedirs(PHANTOM_SCENE_DIR, exist_ok=True)
    print(f"\nCreated scene directory: {PHANTOM_SCENE_DIR}")

    # Get original scene directory
    original_scene_dir = os.path.dirname(ORIGINAL_SCENE)

    # Check if banana model exists
    if not os.path.exists(BANANA_CAD):
        print(f"\nWARNING: Banana CAD model not found at {BANANA_CAD}")
        print("Run 'python download_banana_model.py' first!")
        return None

    # Get relative path from new scene to banana model
    # Or use absolute path
    banana_path = os.path.abspath(BANANA_CAD)

    # Create the XML with updated paths
    scene_xml = BANANA_SCENE_XML.format(
        original_scene_dir=original_scene_dir,
        banana_model_path=banana_path
    )

    # Write the new scene file
    with open(NEW_SCENE_PATH, 'w') as f:
        f.write(scene_xml)

    print(f"\nCreated new scene file: {NEW_SCENE_PATH}")

    # Create a README
    readme_path = os.path.join(PHANTOM_SCENE_DIR, "README.md")
    with open(readme_path, 'w') as f:
        f.write(f"""# Phantom-Touch MuJoCo Scene

This scene is configured for rendering depth patches from phantom-touch dataset.

## Files

- `phantom_banana_scene.xml`: Main scene file with banana object
- Includes FR3 robot from: `{original_scene_dir}`

## Key Features

- Banana object with free joint for full 6D pose control
- Orbbec camera configured with calibrated extrinsics
- DIGIT tactile finger cameras (left and right)
- Physics-based simulation environment

## Usage

Use this scene with `render_depth_patches_phantom.py`:

```python
python render_depth_patches_phantom.py
```

The renderer will:
1. Load object trajectories from phantom-touch tracking
2. Load robot trajectories (if available)
3. Render RGB and depth from all cameras
4. Save results with manifest

## Object Pose Control

The banana object uses a free joint:
- Joint name: `banana_free`
- Body name: `Banana`
- qpos format: [x, y, z, qw, qx, qy, qz]

Poses are set from the tracked object trajectories.
""")

    print(f"Created README: {readme_path}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Scene file: {NEW_SCENE_PATH}")
    print(f"Banana model: {banana_path}")
    print(f"Object name: Banana")
    print(f"Free joint: banana_free")
    print("\nTo use this scene, update the config:")
    print(f"  mujoco_scene_xml: \"{NEW_SCENE_PATH}\"")
    print(f"  object_body_name: \"Banana\"")
    print("="*60)

    return NEW_SCENE_PATH


def main():
    scene_path = create_phantom_scene()

    if scene_path:
        print("\n✓ Scene creation successful!")
        print(f"\nNext steps:")
        print(f"1. Update config/depth_patch_renderer_phantom.yaml")
        print(f"2. Run: python render_depth_patches_phantom.py")
    else:
        print("\n✗ Scene creation failed. Check banana model path.")


if __name__ == "__main__":
    main()
