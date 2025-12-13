import logging
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.phantomutils import normal_principal_to_rotation_matrix


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

from scipy.spatial.transform import Rotation as R


def smooth_quaternions(quaternions, window_size=5):
    pad = window_size // 2
    padded = np.pad(quaternions, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.zeros_like(quaternions)

    for i in range(len(quaternions)):
        window = padded[i:i+window_size]

        # Ensure continuity: flip quaternions to the same hemisphere
        for j in range(1, window.shape[0]):
            if np.dot(window[j], window[j - 1]) < 0:
                window[j] = -window[j]

        avg = np.mean(window, axis=0)
        avg /= np.linalg.norm(avg)
        smoothed[i] = avg

    return smoothed



def moving_average(data, window_size=5):
    pad_width = window_size // 2
    if data.ndim == 1:
        padded = np.pad(data, (pad_width,), mode="edge")
        kernel = np.ones(window_size) / window_size
        return np.convolve(padded, kernel, mode="valid")
    else:
        padded = np.pad(data, ((pad_width, pad_width), (0, 0)), mode="edge")
        kernel = np.ones(window_size) / window_size
        return np.array([
            np.convolve(padded[:, i], kernel, mode="valid")
            for i in range(data.shape[1])
        ]).T

def main():

    resource_manger = DummyResourceManager()

    with resource_manger:
        env = fr3_sim_env(
            # control_mode=ControlMode.CARTESIAN_TQuart,
            control_mode=ControlMode.CARTESIAN_TRPY,
            robot_cfg=default_fr3_sim_robot_cfg(),
            collision_guard=False,
            gripper_cfg=default_fr3_sim_gripper_cfg(),
            camera_set_cfg=default_mujoco_cameraset_cfg(),
        )
        # env.get_wrapper_attr("sim").open_gui()
        obs, _ = env.reset()
        # print(env.unwrapped.robot.get_cartesian_position())

        episodes = [
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e0/handover_collection_temp_e0.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e1/handover_collection_temp_e1.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e2/handover_collection_temp_e2.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e3/handover_collection_temp_e3.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e4/handover_collection_temp_e4.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e5/handover_collection_temp_e5.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e6/handover_collection_temp_e6.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e7/handover_collection_temp_e7.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e8/handover_collection_temp_e8.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e9/handover_collection_temp_e9.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e10/handover_collection_temp_e10.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e11/handover_collection_temp_e11.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e12/handover_collection_temp_e12.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e13/handover_collection_temp_e13.npz",
            "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e14/handover_collection_temp_e14.npz",
        ]
        extrinsics = np.load(
            "/home/epon04yc/phantom-touch/src/phantom-touch/data/robotbase_camera_transform_orbbec_fr4.npy"
        )

        for episode in episodes:
            data = np.load(episode)

            # positions = data["action"][:, :3]
            # rotations = data["action"][:, 3:6]
            # grips = data["action"][:, 6]
            print(data.keys())
            inpainted_images = data["inpainted"]
            positions = moving_average(data["action"][:, :3], window_size=5)
            # rotations = smooth_euler_rotations(data["action"][:, 3:6], window_size=5)
            rotations = R.from_quat(smooth_quaternions(
        R.from_euler('xyz', data["action"][:, 3:6]).as_quat(), window_size=5)
    ).as_euler('xyz')

            grips = moving_average(data["action"][:, 6], window_size=5)


            print(positions.shape)
            images = []
            for i in range(positions.shape[0]):
                act = {
                    "xyzrpy": [
                        positions[i, 0],
                        positions[i, 1],
                        positions[i, 2],
                        rotations[i, 0],
                        rotations[i, 1],
                        rotations[i, 2],
                    ],
                    "gripper": grips[i],
                }
                obs, _, _, _, _ = env.step(act)
                sim_image = obs["frames"]["orbbec"]["rgb"]

                # load the image
                color_image = inpainted_images[i]
                # Normalize the color image and add an alpha channel
                color_image_normalized = color_image.astype(np.float32) / 255.0
                color_image_with_alpha = np.concatenate(
                    [
                        color_image_normalized,
                        np.ones_like(color_image_normalized[:, :, :1]),
                    ],
                    axis=2,
                )

                # Normalize the Mujoco image
                # # resize image to 240 * 432
                sim_image = cv2.resize(sim_image, (432, 240))
                mujoco_image_normalized = sim_image.astype(np.float32) / 255.0

                # Create alpha channel for Mujoco image - black pixels should be transparent
                # Assuming black is [0, 0, 0] in RGB
                mujoco_alpha = np.any(
                    mujoco_image_normalized > 0, axis=2, keepdims=True
                ).astype(np.float32)
                mujoco_image_with_alpha = np.concatenate(
                    [mujoco_image_normalized, mujoco_alpha], axis=2
                )

                # Perform the overlay operation
                overlayed_image = (
                    color_image_with_alpha[:, :, :3]
                    * (1 - mujoco_image_with_alpha[:, :, 3:])
                    + mujoco_image_with_alpha[:, :, :3] * mujoco_image_with_alpha[:, :, 3:]
                )

                # Convert back to 8-bit and BGR for display
                overlayed_image = (overlayed_image * 255).astype(np.uint8)
                images.append(overlayed_image)
                cv2.imshow("Overlayed Image", overlayed_image)
                cv2.waitKey(1)


if __name__ == "__main__":
    main()
