#!/usr/bin/env python3
"""
Render depth patches for phantom-touch dataset using ontouch MuJoCo renderer.
Integrates object trajectories from phantom-touch with robot trajectories and extrinsics.
"""

import os
import sys
import logging
import time
import numpy as np
import mujoco as mj
import mujoco.viewer
import imageio.v2 as iio
from pathlib import Path

# Add ontouch to path to use its utilities
sys.path.insert(0, '/home/epon04yc/ontouch')
from depth_patch_renderer_session.visual_patch_utils.math_renderer_utils import maybe_deg2rad
from depth_patch_renderer_session.visual_patch_utils.robot_renderer_utils import ensure_Tx7, choose_arm_mapping
from depth_patch_renderer_session.visual_patch_utils.gripper_renderer_utils import (
    load_gripper_width_series,
    find_gripper_actuator,
    detect_ctrl_range,
    width_to_ctrl
)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ==================== Configuration ====================
# Paths
TRAJECTORY_ROOT = "/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0/dataset_temp/e0"
PHANTOM_DATASET_ROOT = "/mnt/dataset_drive/ayad/data/phantom-touch/data/output/handover_collection_0"
MUJOCO_SCENE_XML = "/home/epon04yc/phantom-touch/src/phantom-touch/data/banana_scene/phantom_banana_scene.xml"
ORBBEC_EXTRINSICS = "/home/epon04yc/ontouch/calibration/data/orbbec/robotbase_camera_transform_orbbec_fr4.npy"

# Replace strawberry with banana in the scene
BANANA_CAD_MODEL = "/mnt/dataset_drive/ayad/phantom-touch/data/cad_models/banana_scaled.obj"

# Rendering settings
RENDER_W = 240
RENDER_H = 320
TRAJ_RATE_HZ = 10.0  # Playback rate

# Debug settings
ENABLE_VIEWER = True  # Set to True to open MuJoCo viewer for visual debugging
PLAYBACK_SPEED = 0.5  # Slow down factor (0.5 = half speed, 0.1 = very slow)
FRAME_DELAY = 0.1  # Additional delay between frames in seconds

# Viewer Controls:
# - Mouse: Rotate view (left drag), Pan (right drag), Zoom (scroll)
# - Double-click: Select body
# - Ctrl+Right-click: Apply force
# - Space: Pause/Resume
# - ESC: Close viewer and continue rendering
# - Backspace: Reset view

# Output
OUTPUT_DIR_NAME = "mujoco_depth_renders"


def load_phantom_object_trajectory(phantom_root, directory="sam2_object_masks"):
    """
    Load object trajectory from phantom-touch tracking results.
    Returns: positions (Tx3), orientations (Tx3x3) in robot base frame
    """
    tracking_dir = os.path.join(phantom_root, directory)  

    positions = np.load(os.path.join(tracking_dir, "absolute_positions.npy"))
    orientations = np.load(os.path.join(tracking_dir, "absolute_orientations.npy"))

    logging.info(f"Loaded object trajectory in camera frame: {len(positions)} frames")

    positions_robot = []
    orientations_robot = []

    for pos_robot, ori_robot in zip(positions, orientations):
        T_robot = np.eye(4)
        T_robot[:3, :3] = ori_robot
        T_robot[:3, 3] = pos_robot

        positions_robot.append(T_robot[:3, 3])
        orientations_robot.append(T_robot[:3, :3])

    positions_robot = np.array(positions_robot)
    orientations_robot = np.array(orientations_robot)

    logging.info(f"  Robot frame position sample (frame 0): {positions_robot[0]}")
    logging.info(f"  Position range: X[{positions_robot[:, 0].min():.3f}, {positions_robot[:, 0].max():.3f}], "
                 f"Y[{positions_robot[:, 1].min():.3f}, {positions_robot[:, 1].max():.3f}], "
                 f"Z[{positions_robot[:, 2].min():.3f}, {positions_robot[:, 2].max():.3f}]")

    return positions_robot, orientations_robot


def load_phantom_robot_trajectory(trajectory_root, dataset_name="handover_collection_0_e0.npz"):
    """
    Load robot trajectory from phantom-touch dataset (npz file with state key).
    Assumes state is 14-dimensional: first 7 are joint positions, last is gripper action.
    Returns: joint_positions (Tx7), gripper_actions (T,)
    """
    # Check for robot trajectory in phantom dataset
    robot_traj_path = os.path.join(trajectory_root, dataset_name)
    if not os.path.exists(robot_traj_path):
        logging.warning(f"Robot trajectory not found at {robot_traj_path}")
        logging.warning("Will use static robot pose")
        return None, None

    data = np.load(robot_traj_path, allow_pickle=True)

    # Load joint positions (first 7 columns)
    traj = data["state"][:, :7]
    traj = maybe_deg2rad(ensure_Tx7(traj))
    logging.info(f"Loaded robot trajectory: {traj.shape}")

    # Load gripper actions (8th column if available)
    gripper_actions = None
    if data["state"].shape[1] > 7:
        gripper_actions = data["state"][:, -1]  # Gripper action (usually 0=open, 1=closed)
        logging.info(f"Loaded gripper actions: {gripper_actions.shape}")
        logging.info(f"  Gripper range: [{gripper_actions.min():.3f}, {gripper_actions.max():.3f}]")
    else:
        logging.warning("No gripper actions found in trajectory")

    return traj, gripper_actions


def load_camera_extrinsics(extrinsics_path=ORBBEC_EXTRINSICS):
    """Load camera to robot base transform."""
    if os.path.exists(extrinsics_path):
        extr = np.load(extrinsics_path)
        logging.info(f"Loaded Orbbec extrinsics:\n{extr}")
        return extr
    else:
        logging.warning(f"Extrinsics not found at {extrinsics_path}, using identity")
        return np.eye(4)


def rotation_matrix_to_quat_wxyz(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z) format for MuJoCo."""
    # Using Shepperd's method
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def setup_object_in_scene(model, object_name="Banana"):
    """Find the object body and its free joint for pose control."""
    try:
        body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, object_name)
        if body_id < 0:
            logging.error(f"Object body '{object_name}' not found in scene")
            return None, None

        # Find the free joint associated with this body
        for i in range(model.njnt):
            if model.jnt_bodyid[i] == body_id and model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE:
                joint_id = i
                qpos_addr = model.jnt_qposadr[joint_id]
                logging.info(f"Found object '{object_name}' with free joint at qpos[{qpos_addr}:{qpos_addr+7}]")
                return body_id, qpos_addr

        logging.error(f"No free joint found for body '{object_name}'")
        return body_id, None

    except Exception as e:
        logging.error(f"Error setting up object: {e}")
        return None, None


def set_object_pose(data, qpos_addr, position, orientation):
    """
    Set object pose using MuJoCo free joint.
    Free joint format: qpos[addr:addr+7] = [x, y, z, qw, qx, qy, qz]
    """
    if qpos_addr is None:
        return

    # Convert rotation matrix to quaternion
    quat = rotation_matrix_to_quat_wxyz(orientation)

    # Set position and orientation
    data.qpos[qpos_addr:qpos_addr+3] = position
    data.qpos[qpos_addr+3:qpos_addr+7] = quat

def width_to_ctrl(width_series, lo, hi, GRIPPER_MIN_WIDTH_M=0.0, GRIPPER_MAX_WIDTH_M=0.13):
    """
    Convert gripper widths to actuator ctrl within [lo, hi].

    Cases:
      - Normalized [0..1]: scale directly to [lo, hi].
      - Command-like (max > 3): clamp to [lo, hi].
      - Otherwise (meters): map [GRIPPER_MIN_WIDTH_M, GRIPPER_MAX_WIDTH_M] -> [lo, hi].
    """
    w = np.asarray(width_series, dtype=float).reshape(-1)
    if w.size == 0:
        return None

    wmin, wmax = np.nanmin(w), np.nanmax(w)

    # Normalized 0..1 (with a little tolerance)
    if wmin >= -0.1 and wmax <= 1.1:
        u = lo + np.clip(w, 0.0, 1.0) * (hi - lo)
        logging.info("Treating gripper series as normalized [0..1]; scaled to [%.3f, %.3f].", lo, hi)
        return u

    # Raw command-like (e.g., 0..255)
    if wmax > 3.0:
        u = np.clip(w, lo, hi)
        logging.info("Treating gripper series as raw commands; clamped to [%.3f, %.3f].", lo, hi)
        return u

    # Meters -> command #TODO: this will never be reached because the values in meters most likely will be > 0 and < 1
    w_clamped = np.clip(w, GRIPPER_MIN_WIDTH_M, GRIPPER_MAX_WIDTH_M)
    u = (w_clamped - GRIPPER_MIN_WIDTH_M) / max(1e-9, (GRIPPER_MAX_WIDTH_M - GRIPPER_MIN_WIDTH_M))
    u = lo + u * (hi - lo)
    logging.info("Converted widths [m] to command using [%.3f, %.3f] -> [%.3f, %.3f].",
                 GRIPPER_MIN_WIDTH_M, GRIPPER_MAX_WIDTH_M, lo, hi)
    return u

def render_depth_patches(phantom_root):
    """Main rendering loop for phantom-touch dataset."""

    logging.info("="*60)
    logging.info("Phantom-Touch Depth Patch Renderer")
    logging.info("="*60)

    # Load MuJoCo scene
    logging.info(f"Loading MuJoCo scene: {MUJOCO_SCENE_XML}")
    model = mj.MjModel.from_xml_path(MUJOCO_SCENE_XML)
    data = mj.MjData(model)

    # Load trajectories
    logging.info(f"Loading phantom-touch data from: {phantom_root}")
    obj_positions, obj_orientations = load_phantom_object_trajectory(phantom_root)
    robot_traj, gripper_actions = load_phantom_robot_trajectory(TRAJECTORY_ROOT)

    # Setup robot
    if robot_traj is not None:
        arm_idx, arm_names = choose_arm_mapping(model)
        logging.info(f"Arm mapping: {list(zip(range(7), arm_names, arm_idx))}")
        Tq = robot_traj.shape[0]
    else:
        # Use default robot pose
        Tq = 0
        arm_idx = None

    # Setup gripper
    act_id = find_gripper_actuator(model)
    gripper_ctrl = None
    if act_id is not None:
        lo, hi = detect_ctrl_range(model, act_id)
        logging.info(f"Found gripper actuator with range [{lo:.2f}, {hi:.2f}]")

        # Convert gripper actions to control commands if available
        if gripper_actions is not None:
            # Map gripper actions (0=open, 1=closed) to control range
            # gripper_ctrl = lo + gripper_actions * (hi - lo)
            gripper_ctrl = width_to_ctrl(gripper_actions, lo, hi)
            logging.info(f"Mapped gripper actions to control commands")
    else:
        logging.warning("No gripper actuator found")

    # Setup object
    _, qpos_addr = setup_object_in_scene(model, "Banana")

    # Setup renderer
    out_dir = os.path.join(phantom_root, OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Output directory: {out_dir}")

    renderer = mj.Renderer(model, width=RENDER_W, height=RENDER_H)
    renderer.enable_depth_rendering()

    # Find cameras to render
    cameras = ["cam_left_digit", "cam_right_digit", "orbbec"]
    cam_info = []
    for name in cameras:
        cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name)
        if cid >= 0:
            cam_info.append((cid, name))

    logging.info(f"Rendering cameras: {[n for _, n in cam_info]}")

    # Manifest for tracking what was rendered
    manifest = {
        'frame_idx': [],
        'camera_name': [],
        'rgb_path': [],
        'depth_path': [],
        'object_position': [],
        'object_orientation': [],
    }

    # Launch viewer if enabled
    viewer = None
    if ENABLE_VIEWER:
        logging.info("Launching MuJoCo viewer for visual debugging...")
        logging.info(f"Playback speed: {PLAYBACK_SPEED}x")
        logging.info("Press ESC in viewer to continue without visualization")
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.sync()

    # Main rendering loop
    n_frames = len(obj_positions)
    logging.info(f"Rendering {n_frames} frames...")

    try:
        for frame_idx in range(n_frames):
            # Set robot pose
            if robot_traj is not None and frame_idx < Tq:
                data.qpos[arm_idx] = robot_traj[frame_idx]

            # Set gripper control
            if gripper_ctrl is not None and act_id is not None and frame_idx < len(gripper_ctrl):
                data.ctrl[act_id] = gripper_ctrl[frame_idx]

            # Set object pose
            set_object_pose(data, qpos_addr, obj_positions[frame_idx], obj_orientations[frame_idx])

            # Update simulation
            data.qvel[:] = 0.0
            mj.mj_forward(model, data)

            # Roll out gripper actions by stepping the simulation
            for _ in range(10):
                mj.mj_step(model, data)

            # Update viewer if enabled
            if viewer is not None and viewer.is_running():
                viewer.sync()
                # Slow down playback
                time.sleep(FRAME_DELAY / PLAYBACK_SPEED)

            # Render each camera
            for cam_id, cam_name in cam_info:
                renderer.update_scene(data, camera=cam_id)

                # Render RGB
                # turn off depth rendering for RGB
                renderer.disable_depth_rendering()
                rgb = renderer.render()
                rgb_filename = f"frame_{frame_idx:04d}_{cam_name}_rgb.png"
                rgb_path = os.path.join(out_dir, rgb_filename)
                iio.imwrite(rgb_path, rgb)

                # Render depth
                renderer.enable_depth_rendering()
                depth_m = renderer.render()  # depth in meters
                depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
                depth_filename = f"frame_{frame_idx:04d}_{cam_name}_depth.png"
                depth_path = os.path.join(out_dir, depth_filename)
                iio.imwrite(depth_path, depth_mm)

                # Add to manifest
                manifest['frame_idx'].append(frame_idx)
                manifest['camera_name'].append(cam_name)
                manifest['rgb_path'].append(rgb_filename)
                manifest['depth_path'].append(depth_filename)
                manifest['object_position'].append(obj_positions[frame_idx])
                manifest['object_orientation'].append(obj_orientations[frame_idx])

            if frame_idx % 10 == 0:
                logging.info(f"  Rendered {frame_idx}/{n_frames} frames...")

    finally:
        # Close viewer if it was opened
        if viewer is not None:
            viewer.close()
            logging.info("Viewer closed")

    # Save manifest
    manifest_arrays = {
        'frame_idx': np.array(manifest['frame_idx']),
        'camera_name': np.array(manifest['camera_name']),
        'rgb_path': np.array(manifest['rgb_path']),
        'depth_path': np.array(manifest['depth_path']),
        'object_position': np.array(manifest['object_position']),
        'object_orientation': np.array(manifest['object_orientation']),
    }

    manifest_path = os.path.join(out_dir, "render_manifest.npz")
    np.savez(manifest_path, **manifest_arrays)
    logging.info(f"Saved manifest: {manifest_path}")

    logging.info("="*60)
    logging.info("Rendering Complete!")
    logging.info(f"Total frames rendered: {n_frames}")
    logging.info(f"Total images: {len(manifest['rgb_path'])}")
    logging.info(f"Output directory: {out_dir}")
    logging.info("="*60)


def main():
    """Entry point."""
    phantom_root = PHANTOM_DATASET_ROOT

    if not os.path.exists(phantom_root):
        logging.error(f"Phantom dataset root not found: {phantom_root}")
        return

    if not os.path.exists(MUJOCO_SCENE_XML):
        logging.error(f"MuJoCo scene file not found: {MUJOCO_SCENE_XML}")
        return

    render_depth_patches(phantom_root)


if __name__ == "__main__":
    main()
