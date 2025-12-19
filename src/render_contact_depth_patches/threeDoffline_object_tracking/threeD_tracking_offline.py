# standard libraries
import cv2
import numpy as np
import os
import torch
import matplotlib
import datetime
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import open3d as o3d
import trimesh
# models
from sam2.build_sam import build_sam2_video_predictor
# repo libraries
from utils.rgb_utils import load_rgb_images
from utils.sam2utils import (
    search_folder,
)
from phantom_touch.preprocessors.split_episodes import Preprocessor
#### Script metadata ####
matplotlib.use("TKAgg")
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"
from utils.depth_utils import load_raw_depth_images

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        centroid_coords.append([event.xdata, event.ydata])
        print(f"Selected centroid: ({event.xdata:.1f}, {event.ydata:.1f})")
        plt.close()  # Close the figure after one click

def depth_to_point_cloud(depth_image, mask, fx, fy, cx, cy):
    """
    Convert depth image to point cloud using camera intrinsics.
    Only include points where mask is non-zero.
    Returns point cloud and centroid.
    """

    breakpoint()
    h_, w_ = depth_image.shape
    points = []

    # Create meshgrid for image coordinates
    v, u_ = np.meshgrid(np.arange(h_), np.arange(w_), indexing='ij')

    # Apply mask
    valid_mask = mask[:, :, 0] > 0  # Use first channel of RGB mask

    # Get valid depth values
    valid_depth = depth_image[valid_mask]
    valid_u = u_[valid_mask] # the x in image coordinates
    valid_v = v[valid_mask] # the y in image coordinates

    # Convert to 3D points
    z = valid_depth.astype(np.float32) / 1000.0  # Convert to meters
    x = (valid_u - cx) * z / fx
    y = (valid_v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1)

    # Calculate centroid
    centroid = np.mean(points, axis=0)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd, centroid

if __name__ == "__main__":
    # meta paths
    OmegaConf.register_new_resolver("phantom-touch", lambda: search_folder("/home", "phantom-touch"))
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = search_folder("/home", "phantom-touch")

    #### Config loading ####
    threed_cfg = OmegaConf.load(f"{current_dir}/cfg/threeD_tracking_offline.yaml")
    paths_cfg = OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
    object_mask_output_dir = paths_cfg.object_masks_dir
    sam2video_cfg= threed_cfg.sam2videoPredictor
    data_dir = paths_cfg.dataset_directory
    dataset_name = paths_cfg.metadata.experiment_name
    dataset_format = paths_cfg.metadata.dataset_format
    depth_data_dir = paths_cfg.recordings_directory
    depth_shape = tuple(threed_cfg.depth_shape)
    threeD_tracking_output_dir = paths_cfg.threeD_tracking_output_dir
    preprocess_cfg=OmegaConf.load(f"{repo_dir}/src/phantom_touch/cfg/preprocessors.yaml")



    episodes = [
        f"{data_dir}/e{i}/{dataset_name}_e{i}.{dataset_format}"
        for i in range(threed_cfg.start_episode, threed_cfg.end_episode + 1)
    ]
    all_depth_data = load_raw_depth_images(depth_data_dir, depth_shape)
    all_rgb_data, all_rgb_paths = load_rgb_images(depth_data_dir, return_path=True)
    for i,episode in enumerate(episodes): # loop over episodes
        try:
            episode_data = np.load(episode)
        except FileNotFoundError:
            print(f"File not found: {episode}")
            continue
        print(f"Loaded {episode} with {len(episode_data['image_0'])} frames")
        # print episodes with less than 10 frames
        if len(episode_data["image_0"]) < 10:
            print(f"Episode {episode} has less than 10 frames")

        # load all episode_images and depth episode_images from  the recordings instead
        data_preprocessor = Preprocessor(preprocess_cfg,paths_cfg=paths_cfg)
        episodes_meta_info = data_preprocessor.read_episodes()

        episode_images = episode_data["image_0"]
        # Create a temporary directory to store episode_images
        temp_dir = "temporary_images_extraction"
        temporary_images_path = os.path.join(data_dir, temp_dir)
        if threed_cfg.mask_objects:
            # # Save episode_images to the temporary directory
            if not os.path.exists(temporary_images_path) or len(os.listdir(temporary_images_path)) != len(episode_images):
                os.makedirs(temporary_images_path, exist_ok=True)
                for idx, img in enumerate(episode_images):
                    cv2.imwrite(os.path.join(data_dir,temp_dir, f"frame_{idx:04d}.png"), img)

            # Show the first image and get a click
            fig, ax = plt.subplots()
            ax.imshow(episode_images[0])  # Show first image

            # image_idx = episodes_meta_info[f"e{i}"][episode_data["indexes"][0]]   # the indexes key in the episode_data gives information about the remaining indexes relative to one episode after filtering

            # look for the image_idx in the paths names
            # for path in all_rgb_paths:
            #     if f"{image_idx:04d}.png" in path:
            #         original_image_path = path
            #         break
            # original_img = cv2.imread(original_image_path)
            # cv2.imshow("First Frame", original_img)
            # cv2.waitKey(0)
            plt.title("Click to select centroid")
            centroid_coords = []

            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

            # Convert to numpy array if a point was selected
            if centroid_coords:
                centroids = np.array(centroid_coords)
            else:
                raise ValueError("No centroid was selected.")

            #### Second workflow component: VIDEO-SAM2 ####
            print("Running VIDEO-SAM2...")
            points = np.array([[centroids[0][0], centroids[0][1]]], dtype=np.float32)
            predictor = build_sam2_video_predictor(sam2video_cfg.model_cfg, sam2video_cfg.sam2_checkpoint, device=device)
            inference_state = predictor.init_state(video_path=temporary_images_path)
            predictor.reset_state(inference_state)

            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=np.array([1], np.int32)
            )

            mask_frames = []
            for _, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                for i, _ in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                    mask = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
                    mask_frames.append(mask)
            mask_frames = np.array(mask_frames)

            #### Store results ####
            print("Saving masks...")
            frame_names = sorted(
                [os.path.join(temporary_images_path, f) for f in os.listdir(temporary_images_path) if f.endswith(".png")],
                key=lambda p: os.path.splitext(os.path.basename(p))[0]
            )
            print(f"Number of frames: {len(frame_names)}")


            os.makedirs(object_mask_output_dir, exist_ok=True)
            for i, frame_path in enumerate(frame_names):
                frame_name = os.path.join(object_mask_output_dir, f"mask_{os.path.basename(frame_path)}")
                print(f"Saving {frame_name}")
                cv2.imwrite(frame_name, mask_frames[i])

        # the indexes key in the episode_data gives information about the remaining indexes relative to one episode after filtering
        episode_depth_data = all_depth_data[episode_data["indexes"].astype(int)]
        print(f"Loaded {len(episode_depth_data)} depth episode_images from {depth_data_dir}")

        # Optional: Load pre-existing masks instead of generating with SAM2
        # Set LOAD_EXISTING_MASKS to True to load from output_dir
        LOAD_EXISTING_MASKS = threed_cfg.load_masks

        if LOAD_EXISTING_MASKS:
            print(f"Loading existing masks from {object_mask_output_dir}...")
            mask_paths = sorted(
                [os.path.join(object_mask_output_dir, f) for f in os.listdir(object_mask_output_dir) if f.startswith("mask_") and f.endswith(".png")],
                key=lambda p: os.path.splitext(os.path.basename(p))[0]
            )
            mask_frames = []
            for mask_path in mask_paths:
                mask = cv2.imread(mask_path)
                mask_frames.append(mask)
            mask_frames = np.array(mask_frames)
            print(f"Loaded {len(mask_frames)} masks with shape {mask_frames[0].shape}")

        # Camera intrinsic parameters - Orbbec Femto Bolt
        # (imported from utils.hw_camera)
        from utils.hw_camera import orbbec_fx, orbbec_fy, orbbec_cx, orbbec_cy

        # ============ Camera to Robot Transformation ============
        # Load the camera-to-robot transformation (camera frame -> robot base frame)
        CAMERA_TO_ROBOT_TRANSFORM_PATH = threed_cfg.CAMERA_TO_ROBOT_TRANSFORM_PATH
        camera_to_robot_transform = np.load(CAMERA_TO_ROBOT_TRANSFORM_PATH, allow_pickle=True)
        print(f"Loaded camera-to-robot transformation from {CAMERA_TO_ROBOT_TRANSFORM_PATH}")
        print(f"Camera to robot transform:\n{camera_to_robot_transform}")

        # Convert depth episode_images and masks to point clouds

        # ============ CAD Model Configuration ============
        # Path to the CAD model (mesh file: .obj, .stl, .ply, etc.)
        CAD_MODEL_PATH = threed_cfg.CAD_MODEL_PATH  # UPDATE THIS PATH

        # Initial pose of the CAD model in robot base frame
        # Position (x, y, z) in meters
        INITIAL_POSITION = np.array([0.0, 0.0, 0.0])  # UPDATE THIS

        # Initial orientation as rotation matrix or use identity
        # You can also specify as quaternion or euler angles and convert
        INITIAL_ORIENTATION = np.eye(3)  # UPDATE THIS if needed

        # Number of points to sample from CAD model
        CAD_SAMPLE_POINTS = 10000

        # Generate point clouds for each frame
        point_clouds = []
        centroids = []
        print("Converting depth episode_images and masks to point clouds...")
        print(f"Depth shape: {episode_depth_data[0].shape}, Mask shape: {mask_frames[0].shape}")

        # Get depth resolution
        depth_height, depth_width = episode_depth_data[0].shape
        mask_height, mask_width = mask_frames[0].shape[:2]

        print(f"Resizing masks from {mask_width}x{mask_height} to {depth_width}x{depth_height}")

        for i in range(min(len(episode_depth_data), len(mask_frames))):
            # Resize mask to match depth image resolution
            mask_resized = cv2.resize(mask_frames[i], (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)

            # Visualize the first frame to verify mask alignment
            # if i == 0:
            #     # Normalize depth for visualization
            #     depth_normalized = cv2.normalize(episode_depth_data[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #     depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            #     # Create overlay: blend depth image with mask
            #     mask_overlay = mask_resized.copy()
            #     overlay = cv2.addWeighted(depth_colored, 0.6, mask_overlay, 0.4, 0)

            #     # Display side by side
            #     comparison = np.hstack([depth_colored, mask_overlay, overlay])

            #     # Show the visualization
            #     cv2.imshow('Depth | Mask | Overlay', comparison)
            #     print("Press any key to continue...")
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            #TODO:: stopped here
            pcd, centroid = depth_to_point_cloud(episode_depth_data[i], mask_resized, orbbec_fx, orbbec_fy, orbbec_cx, orbbec_cy)

            # Transform point cloud from camera frame to robot base frame
            pcd_robot_frame = o3d.geometry.PointCloud(pcd)
            pcd_robot_frame.transform(camera_to_robot_transform)

            # Transform centroid to robot frame
            centroid_homogeneous = np.append(centroid, 1)  # Convert to homogeneous coordinates
            centroid_robot_frame = (camera_to_robot_transform @ centroid_homogeneous)[:3]

            point_clouds.append(pcd_robot_frame)
            centroids.append(centroid_robot_frame)
            print(f"Generated point cloud {i} with {len(pcd_robot_frame.points)} points")
            print(f"  Centroid (camera frame): {centroid}")
            print(f"  Centroid (robot frame): {centroid_robot_frame}")

        # ============ Load CAD Model ============
        print(f"\nLoading CAD model from {CAD_MODEL_PATH}...")
        cad_mesh = trimesh.load(CAD_MODEL_PATH)
        print(f"CAD mesh loaded with {len(cad_mesh.vertices)} vertices")

        # Sample points from the CAD model surface
        cad_points = cad_mesh.sample(CAD_SAMPLE_POINTS)
        print(f"Sampled {len(cad_points)} points from CAD model")

        # Create Open3D point cloud from CAD model
        cad_pcd = o3d.geometry.PointCloud()
        cad_pcd.points = o3d.utility.Vector3dVector(cad_points)

        # Apply initial pose (position and orientation in robot base frame)
        cad_pcd_initial = o3d.geometry.PointCloud(cad_pcd)
        cad_pcd_initial.rotate(INITIAL_ORIENTATION, center=(0, 0, 0))
        cad_pcd_initial.translate(INITIAL_POSITION)

        # Calculate centroid of the CAD model at initial pose
        cad_centroid_initial = np.mean(np.asarray(cad_pcd_initial.points), axis=0)
        print(f"CAD model initial centroid: {cad_centroid_initial}")

        # Center the CAD model for ICP (rotation-only alignment)
        cad_pcd_centered = o3d.geometry.PointCloud(cad_pcd_initial)
        cad_pcd_centered.translate(-cad_centroid_initial)
        # convert to meters
        cad_pcd_centered.scale(0.001, center=(0, 0, 0))

        # Apply ICP between observation point clouds and CAD model reference
        # ICP optimizes only for rotation, translation comes from centroid tracking
        if len(point_clouds) > 0:

            # Downsample CAD model reference for faster processing
            voxel_size = 0.005
            cad_pcd_down = cad_pcd_centered.voxel_down_sample(voxel_size)
            cad_pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )

            print(f"\nCAD reference point cloud has {len(cad_pcd_down.points)} points after downsampling")

            # Store transformation matrices (full 4x4 including translation from centroid)
            # These transforms bring the CAD model from initial pose to observation pose (in robot frame)
            transformations = []
            rotations = []
            translations = []
            positions = []  # Absolute positions (centroids) in robot base frame
            orientations = []  # Absolute orientations (rotation matrices) in robot base frame
            fitness_scores = []  # Track fitness for filtering
            frame_indices = []  # Track which frames were kept

            # Fitness threshold for filtering
            FITNESS_THRESHOLD = 0.5  # Adjust this value based on your episode_data quality needs

            print("\nApplying ICP alignment (rotation only) + centroid tracking (translation)...")
            print("All point clouds are now in robot base frame...")
            print("Transforming CAD model from initial pose to observation poses...")
            print(f"Filtering frames with fitness < {FITNESS_THRESHOLD}")

            for i, (pcd, centroid) in enumerate(zip(point_clouds, centroids)):

                pcd_centered = o3d.geometry.PointCloud(pcd)
                pcd_centered.translate(pcd_centered.get_center() * -1)
                # get information about the min, max and mean of the two point clouds
                x_min, y_min, z_min = np.min(np.asarray(pcd_centered.points), axis=0)
                x_max, y_max, z_max = np.max(np.asarray(pcd_centered.points), axis=0)
                x_mean, y_mean, z_mean = np.mean(np.asarray(pcd_centered.points), axis=0)
                print(f"\nFrame {i}:")
                print(f"  Observation point cloud - x:[{x_min:.3f}, {x_max:.3f}], y:[{y_min:.3f}, {y_max:.3f}], z:[{z_min:.3f}, {z_max:.3f}], mean:({x_mean:.3f}, {y_mean:.3f}, {z_mean:.3f})")
                # do the same for the reference cad model
                x_min_cad, y_min_cad, z_min_cad = np.min(np.asarray(cad_pcd_down.points), axis=0)
                x_max_cad, y_max_cad, z_max_cad = np.max(np.asarray(cad_pcd_down.points), axis=0)
                x_mean_cad, y_mean_cad, z_mean_cad = np.mean(np.asarray(cad_pcd_down.points), axis=0)
                print(f"  CAD reference point cloud - x:[{x_min_cad:.3f}, {x_max_cad:.3f}], y:[{y_min_cad:.3f}, {y_max_cad:.3f}], z:[{z_min_cad:.3f}, {z_max_cad:.3f}], mean:({x_mean_cad:.3f}, {y_mean_cad:.3f}, {z_mean_cad:.3f})")
                # Downsample current point cloud
                pcd_down = pcd_centered.voxel_down_sample(voxel_size)
                pcd_down.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
                )

                # Apply ICP for rotation only (point clouds are centered)
                # This aligns the centered CAD model to the centered observation
                initial_transform = np.eye(4)
                max_correspondence_distance = 0.05

                reg_result = o3d.pipelines.registration.registration_icp(
                    cad_pcd_down, pcd_down,  # CAD model (source) -> observation (target)
                    max_correspondence_distance,
                    init=initial_transform,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6
                    ),
                )

                # Extract rotation from ICP result
                # This is the rotation needed to align CAD orientation to observation orientation
                rotation_icp = reg_result.transformation[:3, :3]

                # Combine with initial orientation to get absolute orientation in robot frame
                orientation = rotation_icp @ INITIAL_ORIENTATION

                # Position comes directly from observation centroid (already in robot frame)
                position = centroid

                # Build full transformation matrix from robot base frame to observation pose
                # This transforms the CAD model to match the observation
                full_transform = np.eye(4)
                full_transform[:3, :3] = orientation
                full_transform[:3, 3] = position

                # Filter based on fitness threshold
                if reg_result.fitness >= FITNESS_THRESHOLD:
                    transformations.append(full_transform)
                    rotations.append(rotation_icp)  # Incremental rotation from ICP
                    orientations.append(orientation)  # Absolute orientation
                    positions.append(position)  # Absolute position
                    translations.append(position - INITIAL_POSITION)  # Translation from initial
                    fitness_scores.append(reg_result.fitness)
                    frame_indices.append(i)

                    print(f"Frame {i}: fitness={reg_result.fitness:.4f}, inlier_rmse={reg_result.inlier_rmse:.4f} [KEPT]")
                    print(f"  Position (centroid): {position}")
                    print(f"  Rotation angles (deg): {np.rad2deg(o3d.geometry.get_rotation_matrix_from_xyz(np.array([0, 0, 0])))}")
                else:
                    print(f"Frame {i}: fitness={reg_result.fitness:.4f}, inlier_rmse={reg_result.inlier_rmse:.4f} [REJECTED - LOW FITNESS]")

            # Save results
            transformations_array = np.array(transformations)
            rotations_array = np.array(rotations)
            orientations_array = np.array(orientations)
            positions_array = np.array(positions)
            translations_array = np.array(translations)
            fitness_scores_array = np.array(fitness_scores)
            frame_indices_array = np.array(frame_indices)

            print(f"\nFiltering summary:")
            print(f"  Total frames processed: {len(point_clouds)}")
            print(f"  Frames kept (fitness >= {FITNESS_THRESHOLD}): {len(frame_indices)}")
            print(f"  Frames rejected: {len(point_clouds) - len(frame_indices)}")
            print(f"  Average fitness of kept frames: {np.mean(fitness_scores_array):.4f}")

            os.makedirs(threeD_tracking_output_dir, exist_ok=True)
            np.save(os.path.join(threeD_tracking_output_dir, "cad_to_observation_transforms.npy"), transformations_array)
            np.save(os.path.join(threeD_tracking_output_dir, "icp_rotations.npy"), rotations_array)
            np.save(os.path.join(threeD_tracking_output_dir, "absolute_orientations.npy"), orientations_array)
            np.save(os.path.join(threeD_tracking_output_dir, "absolute_positions.npy"), positions_array)
            np.save(os.path.join(threeD_tracking_output_dir, "translations_from_initial.npy"), translations_array)
            np.save(os.path.join(threeD_tracking_output_dir, "fitness_scores.npy"), fitness_scores_array)
            np.save(os.path.join(threeD_tracking_output_dir, "frame_indices.npy"), frame_indices_array)

            # Also save CAD model info
            cad_info = {
                'model_path': CAD_MODEL_PATH,
                'initial_position': INITIAL_POSITION,
                'initial_orientation': INITIAL_ORIENTATION,
                'sample_points': CAD_SAMPLE_POINTS,
                'initial_centroid': cad_centroid_initial
            }
            np.save(os.path.join(threeD_tracking_output_dir, "cad_model_info.npy"), cad_info)

            print(f"\nSaved results to {threeD_tracking_output_dir}:")
            print(f"  - cad_to_observation_transforms.npy (4x4 matrices: CAD initial -> observation in robot frame)")
            print(f"  - icp_rotations.npy (3x3 incremental rotation matrices from ICP)")
            print(f"  - absolute_orientations.npy (3x3 absolute orientation matrices in robot frame)")
            print(f"  - absolute_positions.npy (3D absolute position vectors in robot frame)")
            print(f"  - translations_from_initial.npy (3D translation from initial CAD pose)")
            print(f"  - fitness_scores.npy (ICP fitness scores for kept frames)")
            print(f"  - frame_indices.npy (original frame indices of kept frames)")
            print(f"  - cad_model_info.npy (CAD model metadata)")
            print(f"\nNOTE: All positions and orientations are in robot base frame coordinate system")
            print(f"NOTE: Only frames with fitness >= {FITNESS_THRESHOLD} are included in the outputs")

            # Quick trajectory visualization (saved as image)
            print("\nGenerating quick trajectory preview...")
            positions_array = np.array(positions)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Position over time (use actual frame indices for x-axis)
            axes[0].plot(frame_indices_array, positions_array[:, 0], 'r-', label='X', linewidth=2)
            axes[0].plot(frame_indices_array, positions_array[:, 1], 'g-', label='Y', linewidth=2)
            axes[0].plot(frame_indices_array, positions_array[:, 2], 'b-', label='Z', linewidth=2)
            axes[0].set_xlabel('Frame Index', fontsize=12)
            axes[0].set_ylabel('Position (m)', fontsize=12)
            axes[0].set_title(f'Object Position Over Time (fitness >= {FITNESS_THRESHOLD})', fontsize=14)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

            # X-Y trajectory
            axes[1].plot(positions_array[:, 0], positions_array[:, 1], 'b-', linewidth=2)
            axes[1].scatter(positions_array[0, 0], positions_array[0, 1], c='green', s=150, marker='o', label='Start', zorder=5)
            axes[1].scatter(positions_array[-1, 0], positions_array[-1, 1], c='red', s=150, marker='x', label='End', zorder=5)
            axes[1].set_xlabel('X (m)', fontsize=12)
            axes[1].set_ylabel('Y (m)', fontsize=12)
            axes[1].set_title('X-Y Trajectory', fontsize=14)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)
            axes[1].axis('equal')

            plt.tight_layout()
            trajectory_preview_path = os.path.join(threeD_tracking_output_dir, "trajectory_preview.png")
            plt.savefig(trajectory_preview_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved trajectory preview to: {trajectory_preview_path}")

            # Optional: Visualize aligned point clouds
            # Uncomment to visualize the first few aligned frames with CAD model overlay
            # aligned_pcds = []
            # num_frames_to_show = min(5, len(point_clouds))
            # for i in range(num_frames_to_show):
            #     # Transform CAD model to observation pose
            #     cad_transformed = o3d.geometry.PointCloud(cad_pcd)
            #     cad_transformed.transform(transformations[i])
            #     cad_transformed.paint_uniform_color([1, 0, 0])  # Red for CAD model
            #
            #     # Color the observation point cloud
            #     obs_colored = o3d.geometry.PointCloud(point_clouds[i])
            #     obs_colored.paint_uniform_color([0, i/num_frames_to_show, 1-i/num_frames_to_show])  # Blue gradient
            #
            #     aligned_pcds.append(cad_transformed)
            #     aligned_pcds.append(obs_colored)
            #
            # # Add coordinate frame at robot base
            # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # o3d.visualization.draw_geometries(aligned_pcds + [coord_frame])