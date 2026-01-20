"""MuJoCo utilities for inverse kinematics and robot simulation."""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


def initialize_mujoco_sim(scene_path, home_qpos=None):
    """
    Initialize MuJoCo simulation environment.
    
    Args:
        scene_path: Path to MuJoCo XML scene file
        home_qpos: Optional home joint configuration (7 joints)
        
    Returns:
        model, mj_data, ee_site_id, gripper_actuator_id, finger_joint1_id, finger_joint2_id, camera_id, renderer
    """
    model = mujoco.MjModel.from_xml_path(scene_path)
    mj_data = mujoco.MjData(model)
    
    # Get relevant IDs
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_0")
    if ee_site_id == -1:
        raise ValueError("Could not find 'tcp_0' site in the MuJoCo model")
    
    gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8_0")
    finger_joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1_0")
    finger_joint2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2_0")
    
    print(f"Gripper actuator ID: {gripper_actuator_id}")
    print(f"Finger joint1 ID: {finger_joint1_id}")
    print(f"Finger joint2 ID: {finger_joint2_id}")
    if gripper_actuator_id == -1:
        print("WARNING: Gripper actuator not found!")
    
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "orbbec")
    if camera_id == -1:
        raise ValueError("Could not find 'orbbec' camera in the MuJoCo model")
    
    # Initialize renderer
    renderer = mujoco.Renderer(model, height=1080, width=1920)
    
    # Reset and set home position
    mujoco.mj_resetData(model, mj_data)
    
    if home_qpos is None:
        # Default home configuration for FR3
        home_qpos = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    
    mj_data.qpos[:7] = home_qpos
    mujoco.mj_forward(model, mj_data)
    
    print(f"Home position TCP: {mj_data.site_xpos[ee_site_id]}")
    print(f"Home qpos: {mj_data.qpos[:7]}")
    
    return model, mj_data, ee_site_id, gripper_actuator_id, finger_joint1_id, finger_joint2_id, camera_id, renderer


def solve_ik(model, mj_data, ee_site_id, target_pos, target_rpy, 
             tcp_offset=0.0, alpha=0.3, tol=0.02, max_iter=200, reg=1e-3, verbose=False):
    """
    Solve inverse kinematics using Jacobian-based iterative method.
    
    Args:
        model: MuJoCo model
        mj_data: MuJoCo data
        ee_site_id: End-effector site ID
        target_pos: Target position [x, y, z] for tool center (fingertips)
        target_rpy: Target orientation as roll-pitch-yaw
        tcp_offset: Offset from TCP site to tool center (meters)
        alpha: Step size
        tol: Convergence tolerance
        max_iter: Maximum iterations
        reg: Regularization parameter
        verbose: Print debug info
        
    Returns:
        ik_success: Boolean indicating convergence
        final_error: Final position error norm
        tool_center_pos: Actual tool center position (tcp_offset meters from flange)
        tool_center_rpy: Actual tool center orientation as roll-pitch-yaw
    """
    # Convert RPY to rotation matrix
    target_rot_mat = R.from_euler('xyz', target_rpy).as_matrix()
    
    # Apply TCP offset: transform target from tool center (fingertips) to flange (tcp_0 site)
    # Subtract offset along the hand's z-axis (forward direction)
    if tcp_offset > 0:
        flange_target_pos = target_pos - target_rot_mat[:, 2] * tcp_offset
    else:
        flange_target_pos = target_pos
    
    # Store initial configuration
    initial_qpos = mj_data.qpos[:7].copy()
    
    # Prepare arrays for IK
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    ik_success = False
    final_error = float('inf')
    
    for i in range(max_iter):
        # Forward kinematics
        mujoco.mj_fwdPosition(model, mj_data)
        
        # Get current end-effector pose
        current_pos = mj_data.site_xpos[ee_site_id].copy()
        current_mat = mj_data.site_xmat[ee_site_id].copy().reshape(3, 3)
        
        # Compute position error (use flange target)
        error_pos = flange_target_pos - current_pos
        
        # Compute orientation error using rotation matrices
        error_rot_mat = target_rot_mat @ current_mat.T
        # Convert to axis-angle (small angle approximation)
        error_ori = np.array([
            error_rot_mat[2, 1] - error_rot_mat[1, 2],
            error_rot_mat[0, 2] - error_rot_mat[2, 0],
            error_rot_mat[1, 0] - error_rot_mat[0, 1]
        ]) * 0.5
        
        # Check convergence
        err = np.concatenate([error_pos, error_ori])
        err_norm = np.linalg.norm(err)
        final_error = np.linalg.norm(error_pos)
        
        if err_norm < tol:
            ik_success = True
            break
        
        # Compute site Jacobian
        mujoco.mj_jacSite(model, mj_data, jacp, jacr, ee_site_id)
        
        # Stack position and orientation Jacobians (only first 7 joints)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        
        # Compute pseudo-inverse with damping (Levenberg-Marquardt)
        J_T = J.T
        JJT = J @ J_T + reg * np.eye(6)
        J_pinv = J_T @ np.linalg.inv(JJT)
        
        # Update joint positions
        dq = alpha * J_pinv @ err
        mj_data.qpos[:7] += dq
        
        # Clamp to joint limits
        for j in range(7):
            mj_data.qpos[j] = np.clip(
                mj_data.qpos[j], 
                model.jnt_range[j, 0], 
                model.jnt_range[j, 1]
            )
    
    if not ik_success and verbose:
        final_pos = mj_data.site_xpos[ee_site_id].copy()
        print(f"  Position error: {final_error:.4f}m")
        print(f"  Target pos: {target_pos}")
        print(f"  Current pos: {final_pos}")
        print(f"  Target RPY: {target_rpy}")
        print(f"  Final qpos: {mj_data.qpos[:7]}")
        # Reset to initial configuration
        mj_data.qpos[:7] = initial_qpos
    
    # Calculate final tool center pose (tcp_offset meters from flange along z-axis)
    final_flange_pos = mj_data.site_xpos[ee_site_id].copy()
    final_flange_mat = mj_data.site_xmat[ee_site_id].copy().reshape(3, 3)
    
    # Tool center is tcp_offset meters along the flange's z-axis (forward direction)
    tool_center_pos = final_flange_pos + final_flange_mat[:, 2] * tcp_offset
    tool_center_rpy = R.from_matrix(final_flange_mat).as_euler('xyz')
    
    return ik_success, final_error, tool_center_pos, tool_center_rpy


def get_robot_state(model, mj_data, ee_site_id, finger_joint1_id):
    """
    Get current robot state including joint positions, end-effector pose, and gripper.
    
    Args:
        model: MuJoCo model
        mj_data: MuJoCo data
        ee_site_id: End-effector site ID
        finger_joint1_id: Finger joint ID for gripper state
        
    Returns:
        state: Concatenated state vector [joint_pos, ee_xyz, ee_rpy, gripper]
    """
    # Get joint state (first 7 joints for arm)
    joint_state = mj_data.qpos[:7].copy()
    
    # Get cartesian position
    ee_pos = mj_data.site_xpos[ee_site_id].copy()
    
    # Get rotation matrix and convert to euler angles
    ee_mat = mj_data.site_xmat[ee_site_id].copy().reshape(3, 3)
    ee_rpy = R.from_matrix(ee_mat).as_euler('xyz')
    position_state = np.concatenate([ee_pos, ee_rpy])
    
    # Get gripper state from finger joint position
    if finger_joint1_id >= 0:
        gripper_qpos_idx = model.jnt_qposadr[finger_joint1_id]
        gripper_state = mj_data.qpos[gripper_qpos_idx]
    else:
        gripper_state = 0.0
    
    state = np.concatenate((
        joint_state.flatten(),
        position_state.flatten(),
        np.array([gripper_state]).flatten(),
    ))
    
    return state


def set_gripper(model, mj_data, finger_joint1_id, finger_joint2_id, gripper_value):
    """
    Set gripper position by directly controlling finger joints.
    
    Args:
        model: MuJoCo model
        mj_data: MuJoCo data
        finger_joint1_id: First finger joint ID
        finger_joint2_id: Second finger joint ID
        gripper_value: Desired gripper value [0-1] normalized range
                      0 = fingers close together (small grip distance)
                      1 = fingers far apart (large grip distance)
    """
    # Clamp value to [0, 1] range
    gripper_value = max(0.0, min(1.0, gripper_value))
    
    # Map to joint position: 0 = closed (0m), 1 = open (0.04m max range)
    joint_position = gripper_value * 0.04
    
    # Set both finger joints to the same position (they're coupled via equality constraint)
    if finger_joint1_id >= 0:
        mj_data.qpos[finger_joint1_id] = joint_position
    if finger_joint2_id >= 0:
        mj_data.qpos[finger_joint2_id] = joint_position


def render_camera(renderer, mj_data, camera_id):
    """
    Render camera image.
    
    Args:
        renderer: MuJoCo renderer
        mj_data: MuJoCo data
        camera_id: Camera ID
        
    Returns:
        sim_image: RGB image array
    """
    renderer.update_scene(mj_data, camera=camera_id)
    sim_image = renderer.render()[:, :, :3]  # RGB only
    return sim_image
