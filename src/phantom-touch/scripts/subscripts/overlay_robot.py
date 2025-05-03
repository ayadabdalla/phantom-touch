import logging
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.phantomutils import filter_trajectories, normal_up_to_rotation_matrix


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
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/trajectories/e13/hand_keypoints_e13_right.npz",
        ] 
        # filter trajectories
        positions, rotations, thumbs,indeces,grips = filter_trajectories(trajectories)
        extrinsics = np.load("/home/epon04yc/phantom-touch/src/phantom-touch/data/robotbase_camera_transform_orbbec_fr4.npy")
        # load the rgb image from the directory
        directory = f"/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/inpainting_output/episodes/e13"
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

        # if valid index string exists in file name then keep it
        files = [f for f in files if any(str(i) in f for i in valid_indices)]
    
        # files = [files[i] for i in range(len(files)) if i in valid_indices]
        files = [files[i] for i in range(len(files)) if i not in indeces]
        # convert positions to robot base frame
        print(len(positions[0]), "positions")
        print(len(files), "images")
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
            cv2.imshow("Overlayed Image", overlayed_image)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()