import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils.data_utils import load_keypoints_grouped_by_frame, load_pcds
from utils.depth_utils import load_raw_depth_images
from utils.hw_camera import orbbec_cx, orbbec_cy, orbbec_fx, orbbec_fy
import open3d as o3d
from utils.phantomutils import calculate_action, filter_episode, invert_keypoints, overlay_image
from omegaconf import OmegaConf
import os
from utils.rgb_utils import load_rgb_images
import logging
import mujoco

from utils.samutils import search_folder
from utils.mujoco_utils import (
    initialize_mujoco_sim,
    solve_ik,
    get_robot_state,
    set_gripper,
    render_camera
)
#TODO: revise gripper actions and review script
if __name__ == "__main__":
    # load config
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_dir = search_folder(f"/home/{os.getenv('USER')}/", "phantom-touch")
    phantom_config = OmegaConf.load(f"{parent_directory}/cfg/phantom.yaml")
    paths_cfg = OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
    dataset_root = paths_cfg.dataset_output_directory
    recordings_root = paths_cfg.recordings_directory
    vitpose_root = paths_cfg.vitpose_output_directory
    sam3hand_root = paths_cfg.sam3hand_directory
    inpainting_root = paths_cfg.inpainting_directory
    experiment_name = paths_cfg.metadata.experiment_name

    # Configure logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize MuJoCo simulation environment
    scene_path = f"{repo_dir}/data/fr3_simple_pick_up_digit_hand_wsensor/scene.xml"
    tcp_offset = phantom_config.tcp_offset  # Measured distance from tcp_0(flange) to gripper center for IK in MuJoCo scene
    (model, mj_data, ee_site_id, gripper_actuator_id, 
     finger_joint1_id, finger_joint2_id, camera_id, renderer) = initialize_mujoco_sim(scene_path)
    
    for ep in range(phantom_config.episode_number.start, phantom_config.episode_number.end + 1):
        recordings_root = os.path.join(
            paths_cfg.recordings_directory, f"e{ep}"
        )
        vitpose_root = os.path.join(
            paths_cfg.vitpose_output_directory, f"e{ep}"
        )
        sam3hand_root = os.path.join(
            paths_cfg.sam3hand_directory, f"e{ep}"
        )
        inpainting_root = os.path.join(
            paths_cfg.inpainting_directory, f"e{ep}"
        )
            # === Load Data ===
        # Load color images and their paths
        vitpose_keypoints2d, vitpose_keypoints2d_paths = load_keypoints_grouped_by_frame(
            vitpose_root, prefix="vitpose_", return_path=True
        )
        vitpose_keypoints2d = invert_keypoints(vitpose_keypoints2d,phantom_config)
        numpy_color, color_paths = load_rgb_images(
            recordings_root, prefix="Color_", return_path=True
        )
        inpainted_images = load_rgb_images(inpainting_root, prefix="Color_", return_path=False)
        numpy_depth = load_raw_depth_images(
            recordings_root, shape=[numpy_color.shape[1], numpy_color.shape[2]]
        )

        sam3_pcds, sam3_pcd_paths = load_pcds(sam3hand_root, prefix="Color_", return_path=True)

        extrinsic = np.load(
            f"{repo_dir}/src/cameras/Orbbec/data/robotbase_camera_transform_orbbec_fr4.npy"
        )
        assert (
            numpy_depth.shape[0] == numpy_color.shape[0]
        ), "Number of frames in masks and color images do not match"
        assert numpy_depth.shape[0] == len(
            vitpose_keypoints2d
        ), "Number of keypoints files and depth images do not match"

        assert numpy_depth.shape[0] == len(
            sam3_pcds
        ), "Number of sam3 pcds and depth images do not match"
        logger.info(
            f"Loaded {numpy_depth.shape} frames of depth images, \
            {numpy_color.shape} frames of color images, \
            {len(vitpose_keypoints2d)} files of keypoints, \
            {len(sam3_pcds)} sam segmented pcds, and \
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
        relative_indexes_per_episode = []
        absolute_indexes_per_episode = []

        previous_episode = os.path.basename(os.path.dirname(color_paths[0]))
        
        # start data generation
        for idx, (
            depth_map,
            color_frame,
            color_path,
            keypoints_per_frame,
            sam3_pcd,
            inpainted,
        ) in tqdm(
            enumerate(
                zip(
                    numpy_depth,
                    numpy_color,
                    color_paths,
                    vitpose_keypoints2d,
                    sam3_pcds,
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
                len(sam3_pcds),
            )
            if idx == last_index - 1:
                # discard the last image
                originals_per_episode = originals_per_episode[:-1]
                images_per_episode = images_per_episode[:-1]
                inpainted_images_per_episode = inpainted_images_per_episode[:-1]
                keypoints_per_episode = keypoints_per_episode[:-1]
                states_per_episode = states_per_episode[:-1]
                actions_per_episode = actions_per_episode[1:]
                relative_indexes_per_episode = relative_indexes_per_episode[:-1]
                absolute_indexes_per_episode = absolute_indexes_per_episode[:-1]
                data = {
                    "action": actions_per_episode,
                    "image_0": images_per_episode,
                    "state": states_per_episode,
                    "keypoints": keypoints_per_episode,
                    "inpainted": inpainted_images_per_episode,
                    "original": originals_per_episode,
                    "indexes": relative_indexes_per_episode,
                    "absolute_indexes": absolute_indexes_per_episode
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
                    absolute_indexes=data["absolute_indexes"]
                )
                mujoco.mj_resetData(model, mj_data)
                mujoco.mj_forward(model, mj_data)
                break
                        
            chamfer_distances = []
            target_actions_per_frame = []
            pcds_per_frame = []
            invalid_keypoint = False  # ignore frames with one or more hands with bad depth values
            for i, keypoints2d in enumerate(keypoints_per_frame):
                keypoints2d = keypoints2d[:, :2].astype(int)
                if (
                    keypoints2d[:, 0].max() >= depth_map.shape[1]
                    or keypoints2d[:, 1].max() >= depth_map.shape[0]
                ):
                    # clip out-of-bounds keypoints
                    # keypoints2d[:, 0] = np.clip(keypoints2d[:, 0], 0, depth_map.shape[1] - 1)
                    # keypoints2d[:, 1] = np.clip(keypoints2d[:, 1], 0, depth_map.shape[0] - 1)
                    print(f"Frame {frame_index}: Keypoints out of bounds, skipping frame.")
                    continue
                # project the keypoints to 3D camera coordinates
                points = np.zeros((len(keypoints2d), 3))
                colors = np.zeros((len(keypoints2d), 3))
                for j, (x, y) in enumerate(keypoints2d):
                    z = depth_map[y, x]
                    if j in {4, 8}:  # Thumb tip and index tip
                        if z <= 250 or z >= 5000:
                            invalid_keypoint = True
                            break
                    colors[j] = [1, 0, 0]
                    points[j] = [x, y, z]
                if invalid_keypoint:
                    print(f"Frame {frame_index}: Invalid keypoint detected with depth {z}, skipping frame.")
                    continue
                
                # Convert to 3D camera coordinates
                points[:, 0] = (points[:, 0] - orbbec_cx) * points[:, 2] / orbbec_fx / 1000.0
                points[:, 1] = (points[:, 1] - orbbec_cy) * points[:, 2] / orbbec_fy / 1000.0
                points[:, 2] = points[:, 2] / 1000.0
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points[[4,8], :])
                pcd.colors = o3d.utility.Vector3dVector(colors[[4,8], :])
                pcds_per_frame.append(pcd)
                if isinstance(sam3_pcd, np.ndarray):
                    sam3_pcd=sam3_pcd[0]
                else:
                    sam3_pcd = sam3_pcd
                sam3_pcd = sam3_pcd.voxel_down_sample(voxel_size=0.01)
                chamfer_distance = pcd.compute_point_cloud_distance(sam3_pcd)
                chamfer_distance = np.asarray(chamfer_distance)
                chamfer_distance = np.mean(chamfer_distance)
                chamfer_distances.append(chamfer_distance)

                action = calculate_action(points, extrinsic)
                target_actions_per_frame.append(action)
                invalid_keypoint = False


            if len(chamfer_distances) == 0:
                print(f"Frame {frame_index}: No valid keypoints detected, skipping frame.")
                continue
            elif min(chamfer_distances) > phantom_config.chamfer_distance_threshold:

                print(f"Frame {frame_index}: No good matching keypoints, skipping frame, the min chamfer distance is {min(chamfer_distances)}")
                
                # # Visualize thumb tip and index tip on sam3 pcd for debugging
                # min_chamfer_index_debug = chamfer_distances.index(min(chamfer_distances))
                # pcd_debug = pcds_per_frame[min_chamfer_index_debug]
                # keypoints_3d_debug = np.asarray(pcd_debug.points)
                
                # # Extract thumb tip (index 4) and index tip (index 8)
                # thumb_tip = keypoints_3d_debug[4]
                # index_tip = keypoints_3d_debug[8]
                
                # # Create spheres for visualization
                # thumb_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                # thumb_sphere.translate(thumb_tip)
                # thumb_sphere.paint_uniform_color([1, 0, 0])  # Red for thumb
                
                # index_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                # index_sphere.translate(index_tip)
                # index_sphere.paint_uniform_color([0, 1, 0])  # Green for index
                
                # # Color the sam3_pcd for better visualization
                # sam3_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray
                
                # # Visualize all keypoints in blue
                # all_keypoints_pcd = o3d.geometry.PointCloud()
                # all_keypoints_pcd.points = o3d.utility.Vector3dVector(keypoints_3d_debug)
                # all_keypoints_pcd.paint_uniform_color([0, 0, 1])  # Blue
                
                # print(f"Thumb tip 3D: {thumb_tip}")
                # print(f"Index tip 3D: {index_tip}")
                # print(f"Visualizing frame {idx} with chamfer distance: {min(chamfer_distances):.4f}")
                
                # # Display the 3D scene
                # o3d.visualization.draw_geometries(
                #     [sam3_pcd, thumb_sphere, index_sphere, all_keypoints_pcd],
                #     window_name=f"Frame {idx} - Chamfer: {min(chamfer_distances):.4f}",
                #     width=1024, height=768
                # )
                
                continue
            min_chamfer_distance = min(chamfer_distances)
            min_chamfer_index = chamfer_distances.index(min_chamfer_distance)

            action = target_actions_per_frame[min_chamfer_index]
            pcd = pcds_per_frame[min_chamfer_index]

            keypoints = np.asarray(points)

            # Extract target pose from action
            target_pos = action[:3]
            target_rpy = action[3:6]
            gripper_value = action[6]
            
            # Store initial configuration for potential reset
            initial_qpos = mj_data.qpos[:7].copy()
            
            # Solve inverse kinematics
            ik_success, pos_err, tool_center_pos, tool_center_rpy = solve_ik(
                model, mj_data, ee_site_id, 
                target_pos, target_rpy,
                tcp_offset=tcp_offset,
                alpha=0.3, tol=0.02, max_iter=200, reg=1e-3,
                verbose=False
            )
            
            if not ik_success:
                print(f"\nIK failed for frame {idx}: pos_err={pos_err:.4f}m")
                print(f"  Target: {target_pos}, RPY: {target_rpy}")
                mj_data.qpos[:7] = initial_qpos
                continue
            
            # Set gripper position
            set_gripper(model, mj_data, finger_joint1_id, finger_joint2_id, gripper_value)
            
            # Step simulation multiple times to let actuator respond
            # for _ in range(100):
            #     mujoco.mj_step(model, mj_data)
            
            # Get robot state
            state = get_robot_state(model, mj_data, ee_site_id, finger_joint1_id)
            
            # Render camera image
            sim_image = render_camera(renderer, mj_data, camera_id)
            
            overlayed_image = overlay_image(
                inpainted, sim_image, (inpainted.shape[1], inpainted.shape[0])
            )
            states_per_episode.append(state)
            images_per_episode.append(overlayed_image)
            actions_per_episode.append(action)
            keypoints_per_episode.append(keypoints)
            inpainted_images_per_episode.append(inpainted)
            originals_per_episode.append(color_frame)
            relative_indexes_per_episode.append(idx)
            absolute_indexes_per_episode.append(frame_index)

        print("Batch processing complete.")
