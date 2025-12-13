import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils.data_utils import load_keypoints_grouped_by_frame, load_pcds
from utils.depth_utils import load_raw_depth_images
from utils.hw_camera import fx, fy, cx, cy
import open3d as o3d
from utils.phantomutils import calculate_action, filter_episode, invert_keypoints, overlay_image
from omegaconf import OmegaConf
import os
from utils.rgb_utils import load_rgb_images
from rcsss.envs.base import ControlMode
from rcsss.envs.factories import fr3_sim_env
import logging
from rcsss.envs.utils import (
    default_fr3_sim_gripper_cfg,
    default_fr3_sim_robot_cfg,
    default_mujoco_cameraset_cfg,
)

# load config
parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = OmegaConf.load(f"{parent_directory}/conf/phantom.yaml")
dataset_root = config.dataset_output_directory
recordings_root = config.recordings_directory
vitpose_root = config.vitpose_output_directory
sam2hand_root = config.sam2hand_directory
inpainting_root = config.inpainting_directory
experiment_name = config.experiment_name

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = fr3_sim_env(
    control_mode=ControlMode.CARTESIAN_TRPY,
    robot_cfg=default_fr3_sim_robot_cfg(),
    collision_guard=False,
    gripper_cfg=default_fr3_sim_gripper_cfg(),
    camera_set_cfg=default_mujoco_cameraset_cfg(),
)
obs, _ = env.reset()
for ep in range(0,1):
    recordings_root = os.path.join(
        config.recordings_directory, f"e{ep}"
    )
    vitpose_root = os.path.join(
        config.vitpose_output_directory, f"e{ep}"
    )
    sam2hand_root = os.path.join(
        config.sam2hand_directory, f"e{ep}"
    )
    inpainting_root = os.path.join(
        config.inpainting_directory, f"e{ep}"
    )
        # === Load Data ===
    # Load color images and their paths
    vitpose_keypoints2d, vitpose_keypoints2d_paths = load_keypoints_grouped_by_frame(
        vitpose_root, prefix="vitpose_", return_path=True
    )
    vitpose_keypoints2d = invert_keypoints(vitpose_keypoints2d,config)
    numpy_color, color_paths = load_rgb_images(
        recordings_root, prefix="Color_", return_path=True
    )
    inpainted_images = load_rgb_images(inpainting_root, prefix="Color_", return_path=False)
    numpy_depth = load_raw_depth_images(
        recordings_root, shape=[numpy_color.shape[1], numpy_color.shape[2]]
    )

    sam2_pcds, sam2_pcd_paths = load_pcds(sam2hand_root, prefix="Color_", return_path=True)

    extrinsic = np.load(
        "/home/epon04yc/phantom-touch/src/calibration/robotbase_camera_transform_orbbec_fr4.npy"
    )
    assert (
        numpy_depth.shape[0] == numpy_color.shape[0]
    ), "Number of frames in masks and color images do not match"
    assert numpy_depth.shape[0] == len(
        vitpose_keypoints2d
    ), "Number of keypoints files and depth images do not match"


    logger.info(
        f"Loaded {numpy_depth.shape} frames of depth images, \
        {numpy_color.shape} frames of color images, \
        {len(vitpose_keypoints2d)} files of keypoints, \
        {len(sam2_pcds)} sam segmented pcds, and \
        {inpainted_images.shape} inpainted images"
    )

    # === Process Data ===
    # Initialize lists to store data for each episode
    keypoints_per_episode = []
    actions_per_episode = []
    images_per_episode = []
    states_per_episode = []
    inpainted_images_per_episode = []
    originals_per_episode = []
    indexes_per_episode = []
    previous_episode = os.path.basename(os.path.dirname(color_paths[0]))

    # start data generation
    for idx, (
        depth_map,
        color_frame,
        color_path,
        keypoints_per_frame,
        sam2_pcd,
        inpainted,
    ) in tqdm(
        enumerate(
            zip(
                numpy_depth,
                numpy_color,
                color_paths,
                vitpose_keypoints2d,
                sam2_pcds,
                inpainted_images,
            ),
        ),
        total=numpy_depth.shape[0],
        desc="Generating DataSet",
    ):
        episode = os.path.basename(os.path.dirname(color_path))
        frame_index = os.path.splitext(color_path)[0].split("_")[-1]
        # get the minimum shape of the all the data iterated on
        last_index = min(
            numpy_depth.shape[0],
            numpy_color.shape[0],
            len(vitpose_keypoints2d),
            len(sam2_pcds),
        )
        if idx == last_index - 1:
            # discard the last image
            originals_per_episode = originals_per_episode[:-1]
            images_per_episode = images_per_episode[:-1]
            inpainted_images_per_episode = inpainted_images_per_episode[:-1]
            keypoints_per_episode = keypoints_per_episode[:-1]
            states_per_episode = states_per_episode[:-1]
            actions_per_episode = actions_per_episode[1:]
            indexes_per_episode = indexes_per_episode[:-1]
            data = {
                "action": actions_per_episode,
                "image_0": images_per_episode,
                "state": states_per_episode,
                "keypoints": keypoints_per_episode,
                "inpainted": inpainted_images_per_episode,
                "original": originals_per_episode,
                "indexes": indexes_per_episode
            }
            print(f"Saving episode {previous_episode} with {len(data['action'])} frames")
            data = filter_episode(data)
            save_name = f"{experiment_name}_{previous_episode}.npz"
            os.makedirs(os.path.join(dataset_root, previous_episode), exist_ok=True)
            np.savez_compressed(
                os.path.join(dataset_root, previous_episode, save_name),
                action=data["action"],
                image_0=data["image_0"],
                state=data["state"],
                keypoints=data["keypoints"],
                inpainted=data["inpainted"],
                original=data["original"],
                indexes=data["indexes"],
            )
            env.reset()
            break

        sam2_pcd = sam2_pcd[0]  # squeeze
        chamfer_distances = []
        target_actions_per_frame = []
        pcds_per_frame = []
        invalid_keypoint = (
            False  # ignore frames with one or more hands with bad depth values
        )
        for i, keypoints2d in enumerate(keypoints_per_frame):
            keypoints2d = keypoints2d[:, :2].astype(int)
            if (
                keypoints2d[:, 0].max() >= depth_map.shape[1]
                or keypoints2d[:, 1].max() >= depth_map.shape[0]
            ):
                invalid_keypoint = True
                continue
            depth_keypoints_map = depth_map
            depth_keypoints_map[keypoints2d[:, 1], keypoints2d[:, 0]] = depth_map[
                keypoints2d[:, 1], keypoints2d[:, 0]
            ]
            # project the keypoints to 3D camera coordinates
            points = np.zeros((len(keypoints2d), 3))
            colors = np.zeros((len(keypoints2d), 3))
            for j, (x, y) in enumerate(keypoints2d):
                z = depth_map[y, x]
                if j in {4, 8}:
                    if z <= 250 or z >= 5000:
                        invalid_keypoint = True
                else:
                    colors[j] = [1, 1, 1]
                points[j] = [x, y, z]
            if invalid_keypoint:
                print("Invalid keypoint detected, skipping frame.")
                continue
            invalid_keypoint = False
            points[:, 0] = (points[:, 0] - cx) * points[:, 2] / fx / 1000.0
            points[:, 1] = (points[:, 1] - cy) * points[:, 2] / fy / 1000.0
            points[:, 2] = points[:, 2] / 1000.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcds_per_frame.append(pcd)

            sam2_pcd = sam2_pcd.voxel_down_sample(voxel_size=0.01)
            chamfer_distance = pcd.compute_point_cloud_distance(sam2_pcd)
            chamfer_distance = np.asarray(chamfer_distance)
            chamfer_distance = np.mean(chamfer_distance)
            chamfer_distances.append(chamfer_distance)

            action = calculate_action(points, extrinsic)
            target_actions_per_frame.append(action)

        if len(chamfer_distances) == 0:
            print("No valid keypoints detected, skipping frame.")
            continue
        elif min(chamfer_distances) > 0.005:
            print("No good matching keypoints, skipping frame.")
            continue
        min_chamfer_distance = min(chamfer_distances)
        min_chamfer_index = chamfer_distances.index(min_chamfer_distance)

        action = target_actions_per_frame[min_chamfer_index]
        pcd = pcds_per_frame[min_chamfer_index]

        keypoints = np.asarray(pcd.points)

        act = {
            "xyzrpy": [
                action[0],
                action[1],
                action[2],
                action[3],
                action[4],
                action[5],
            ],
            "gripper": action[6],
        }
        obs, _, _, _, _ = env.step(act)
        if not env.get_wrapper_attr("robot").get_state().ik_success:
            continue
        joint_state = env.get_wrapper_attr("robot").get_joint_position()
        position_state = env.get_wrapper_attr("robot").get_cartesian_position().xyzrpy()
        gripper_state = obs["gripper"]
        state = np.concatenate(
            (
                joint_state.flatten(),
                position_state.flatten(),
                np.array([gripper_state]).flatten(),
            )
        )
        sim_image = obs["frames"]["orbbec"]["rgb"]
        # show inpainted image
        overlayed_image = overlay_image(
            inpainted, sim_image, (inpainted.shape[1], inpainted.shape[0])
        )
        states_per_episode.append(state)
        images_per_episode.append(overlayed_image)
        actions_per_episode.append(action)
        keypoints_per_episode.append(keypoints)
        inpainted_images_per_episode.append(inpainted)
        originals_per_episode.append(color_frame)
        indexes_per_episode.append(idx)

    print("Batch processing complete.")
