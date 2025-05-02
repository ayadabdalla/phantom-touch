import logging
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import logging

from rcsss.control.fr3_desk import DummyResourceManager
from rcsss.envs.base import ControlMode
from rcsss.envs.factories import fr3_sim_env
from rcsss.envs.utils import (
    default_fr3_sim_gripper_cfg,
    default_fr3_sim_robot_cfg,
    default_mujoco_cameraset_cfg,
)

def filter_trajectories(trajectories):
    # do exponential average on only the z axis
    filtered_positions = []
    filtered_normals = []
    filtered_thumbs = []
    filtered_grips = []
    for trajectory in trajectories:
        data = np.load(trajectory)
        positions = data["positions"]
        normals = data["normals"]
        thumbs = data["thumb_vectors"]
        keypoints = data["keypoints"]
        grips = np.linalg.norm(positions - keypoints[:, 8, :3], axis=1)
        # normalize the grips
        grips = grips / np.max(grips)
        filtered_z = np.zeros_like(positions[:, 2])
        # expect the next 2 steps and compare to the actual and if it's too far ignore it
        indeces = []
        for i in range(0, len(positions) - 2):
            filtered_z[i] = 0.8 * positions[i + 2, 2] + 0.2 * positions[i + 1, 2]
            # Check if the current z is too far from the expected z
            if abs(positions[i, 2] - filtered_z[i]) > 0.1:
                # store index of the filtered z
                indeces.append(i)
        positions = np.delete(positions, indeces, axis=0)
        normals = np.delete(normals, indeces, axis=0)
        thumbs = np.delete(thumbs, indeces, axis=0)
        grips = np.delete(grips, indeces, axis=0)
        print(f"Filtered out: {len(indeces)}")
        filtered_positions.append(positions)
        filtered_thumbs.append(thumbs)
        filtered_normals.append(normals)
        filtered_grips.append(grips)
    return filtered_positions, filtered_normals, filtered_thumbs, indeces, filtered_grips
def normal_up_to_rotation_matrix(normal, up=None, eps=1e-6):
    # Normalize the input normal vector
    normal = -normal / np.linalg.norm(normal, 2)

    # Set the z-axis as the provided normal vector
    basis_three = up

    # Calculate a perpendicular vector to the z-axis for the x-axis
    # basis_one = np.cross(np.array([0, 0, 1]), basis_three)
    basis_one = normal
    basis_one = basis_one / np.linalg.norm(basis_one, 2)

    # if basis one is zero vector use x-axis
    if np.linalg.norm(basis_one, 2) == 0:
        print("basis one is zero vector")
        basis_one = np.array([1, 0, 0])

    # Calculate the y-axis as the cross product of the z-axis and x-axis
    basis_two = np.cross(basis_three, basis_one)
    basis_two = basis_two / np.linalg.norm(basis_two, 2)

    # Construct the rotation matrix
    rot = np.vstack((basis_one, basis_two, basis_three)).T
    return rot

def main():

    resource_manger = DummyResourceManager()

    with resource_manger:
        env = fr3_sim_env(
            control_mode=ControlMode.CARTESIAN_TQuart,
            robot_cfg=default_fr3_sim_robot_cfg(),
            collision_guard=False,
            gripper_cfg=default_fr3_sim_gripper_cfg(),
            camera_set_cfg=default_mujoco_cameraset_cfg(),
        )
        env.get_wrapper_attr("sim").open_gui()
        obs,_=env.reset()
        print(env.unwrapped.robot.get_cartesian_position())

        trajectories = [
            "/home/ayad/ontouch/phantomtouch/phantomdata/handover/trajectories/e14/hand_keypoints_e14_right.npz",
        ] 
        # filter trajectories
        positions, rotations, thumbs,indeces,grips = filter_trajectories(trajectories)
        extrinsics = np.load("/home/ayad/ontouch/calibration/data/orbbec/robotbase_camera_transform_orbbec_fr4.npy")
        # load the rgb image from the directory
        directory = f"/home/ayad/ontouch/phantomtouch/phantomdata/handover/inpainting_output/episodes/e14"
        # get the files in the directory
        files = os.listdir(directory)
        # get only the files that end with .png
        files = [f for f in files if f.endswith(".png")]
        # sort the files
        files.sort()
        # delete the indeces from the files
        data = np.load(trajectories[0])
        valid_indices = data["valid_frames"]
        print("valid indices", len(valid_indices))
        print(len(positions[0]), "positions")

        # if valid index string exists in file name then keep it
        files = [f for f in files if any(str(i) in f for i in valid_indices)]
    
        # files = [files[i] for i in range(len(files)) if i in valid_indices]
        files = [files[i] for i in range(len(files)) if i not in indeces]
        # convert positions to robot base frame
        positions = positions[0]
        rotations = rotations[0]
        thumbs = thumbs[0]
        grips = grips[0]

        positions = np.concatenate([positions, np.ones((positions.shape[0], 1))], axis=1)
        positions = np.dot(extrinsics, positions.T).T

        rotations = np.concatenate([rotations, np.zeros((rotations.shape[0], 1))], axis=1)
        rotations = np.dot(extrinsics, rotations.T).T
        rotations = rotations[:, :3]

        thumbs = np.concatenate([thumbs, np.zeros((thumbs.shape[0], 1))], axis=1)
        thumbs = np.dot(extrinsics, thumbs.T).T
        thumbs = thumbs[:, :3]

        rotation_matrices = []
        for i in range(rotations.shape[0]):
            normal = rotations[i]
            thumb = thumbs[i]
            rotation_matrix = normal_up_to_rotation_matrix(normal, thumb)
            rotation_matrices.append(rotation_matrix)

        rotation_matrix = np.array([
            [-0.00617474, -0.999715, 0.023071],
            [0.0317128, -0.0232556, -0.999226],
            [0.999478, -0.00543832, 0.0318473]
        ])
        r = R.from_matrix(rotation_matrix)
        target_orn = r.as_quat(scalar_first=True)  # Convert rotation matrix to quaternion
        # put w first
        if target_orn[0] < 0:
            target_orn = -target_orn
            print("quaternion is negative")
        else:
            print("quaternion is already positive")


        act = {"tquart": [0.59,0.18,0.24, target_orn[0], target_orn[1], target_orn[2], target_orn[3]], "gripper": 1}
        # rollout trajectories
        # generate positions.shape[0] copies of the target orn
        obs,_,_,_,_= env.step(act)

        images = []
        for i in range(positions.shape[0]):
            # rotation_matrix to quaternion
            r = R.from_matrix(rotation_matrices[i])
            target_orn = r.as_quat(scalar_first=False)  # Convert rotation matrix to quaternion
            act = {"tquart": [positions[i, 0], positions[i, 1], positions[i, 2], target_orn[0], target_orn[1], target_orn[2], target_orn[3]], "gripper": grips[i]}
            obs,_,_,_,_= env.step(act)
            image = obs["frames"]["orbbec"]['rgb']

            # load the image
            color_image = cv2.imread(os.path.join(directory, files[i]))
            # Normalize the color image and add an alpha channel
            color_image_normalized = color_image.astype(np.float32) / 255.0
            color_image_with_alpha = np.concatenate([color_image_normalized, np.ones_like(color_image_normalized[:, :, :1])], axis=2)

            # Normalize the Mujoco image
            # # resize image to 240 * 432
            image = cv2.resize(image, (432, 240))
            mujoco_image_normalized = image.astype(np.float32) / 255.0

            # Create alpha channel for Mujoco image - black pixels should be transparent
            # Assuming black is [0, 0, 0] in RGB
            mujoco_alpha = np.any(mujoco_image_normalized > 0, axis=2, keepdims=True).astype(np.float32)
            mujoco_image_with_alpha = np.concatenate([mujoco_image_normalized, mujoco_alpha], axis=2)

            # Perform the overlay operation
            overlayed_image = color_image_with_alpha[:, :, :3] * (1 - mujoco_image_with_alpha[:, :, 3:]) + \
                            mujoco_image_with_alpha[:, :, :3] * mujoco_image_with_alpha[:, :, 3:]

            # Convert back to 8-bit and BGR for display
            overlayed_image = (overlayed_image * 255).astype(np.uint8)
            images.append(overlayed_image)

if __name__ == "__main__":
    main()