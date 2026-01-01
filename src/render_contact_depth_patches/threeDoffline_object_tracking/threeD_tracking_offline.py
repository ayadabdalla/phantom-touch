"""3D Object Tracking Offline Pipeline

This module performs offline 3D object tracking by:
1. Loading RGB and depth data from episodes
2. Segmenting objects using SAM3
3. Converting depth to point clouds
4. Aligning CAD models to observations using ICP
5. Tracking object pose over time

REFERENCE FRAME DOCUMENTATION
==============================

This pipeline tracks transformations through multiple coordinate frames.
Understanding these frames is CRITICAL for using the output data correctly.

COORDINATE FRAMES:
------------------

1. CAMERA FRAME
   - Origin: Camera optical center
   - Axes: Standard camera convention
   - Source: Depth sensor intrinsics (orbbec_fx, orbbec_fy, orbbec_cx, orbbec_cy)
   - When: depth_to_point_cloud() converts depth pixels → 3D points

2. ROBOT FRAME
   - Origin: Robot base
   - Axes: Robot's world coordinate system
   - Transform: T_robot_from_camera (loaded from CAMERA_TO_ROBOT_TRANSFORM_PATH)
   - When: Final output transforms are all in this frame

3. CAD MODEL FRAME
   - Origin: Original CAD file's coordinate system
   - When: CAD mesh is first loaded from file

4. CENTERED CAD FRAME
   - Origin: CAD model centroid moved to (0,0,0)
   - When: load_cad_model() centers and scales the model
   - Purpose: ICP alignment (both CAD and observations centered)

PROCESSING PIPELINE:
--------------------

Step 1: Depth → Camera Frame
    depth_to_point_cloud(depth_image, mask)
    → point cloud in CAMERA FRAME
    → centroid in CAMERA FRAME

Step 2: ICP Alignment (in camera frame)
    - Center observation at origin (matches centered CAD)
    - ICP: centered CAD → centered observation
    - Result: R_icp (rotation in centered space)
    - Build: T_camera_from_cad = [R_icp | centroid_camera]

Step 3: Transform to Robot Frame
    - T_robot_from_cad = T_robot_from_camera @ T_camera_from_cad
    - Extract position and rotation in robot frame
    - This is the FINAL OUTPUT frame

OUTPUT DATA KEYS (in saved NPZ files):
--------------------------------------

TRANSFORMS:
  T_robot_from_cad (N,4,4)
    → Transforms centered CAD model to robot frame per frame
    → Use this to visualize CAD model at tracked poses

POSITIONS:
  object_pos_in_camera (N,3)
    → Object centroid in camera frame (intermediate)
  object_pos_in_robot (N,3)
    → Object centroid in robot frame (MAIN OUTPUT)

ROTATIONS:
  R_icp_in_centered_space (N,3,3)
    → ICP output: rotation matrix in camera frame axes
    → Rotates centered CAD to match centered observation
  R_object_in_robot (N,3,3)
    → Same rotation but expressed in robot frame axes
    → WARNING: Depends on CAD file's original orientation

TRAJECTORY:
  displacement_from_start (N,3)
    → Displacement from first tracked position (robot frame)

IMPORTANT NOTES:
----------------
- Each frame is processed INDEPENDENTLY (no temporal tracking)
- ICP always initializes from identity (init=eye(4))
- Bad fitness scores do NOT propagate to next frames
- All final outputs are in ROBOT FRAME for easy use

CAD ORIENTATION DEPENDENCY:
---------------------------
CRITICAL: Rotation outputs depend on the CAD model's orientation in its file!

Background:
  The CAD model is centered at origin but KEEPS its original orientation from
  the file. ICP finds the rotation needed to align this CAD to the observation.
  Different CAD file orientation → Different rotation matrices in output.

What is CAD-orientation-INDEPENDENT:
  ✓ object_pos_in_robot        - Centroid position (no rotation involved)
  ✓ displacement_from_start     - Position changes over time
  ✓ T_robot_from_cad            - WHEN APPLIED to your centered CAD model

What DEPENDS on CAD file orientation:
  ⚠ R_icp_in_centered_space     - Aligns CAD-as-loaded to observation (camera frame)
  ⚠ R_object_in_robot           - Same rotation expressed in robot frame

Why T_robot_from_cad IS independent (when applied):
  Although rotation matrix values differ for different CAD orientations, when
  you apply the full transform to your centered CAD model, you ALWAYS get the
  correct observed pose. The rotation compensates for CAD's initial orientation.

  Example:
    CAD_A (upright in file):   T_A @ CAD_A = Correct observed pose
    CAD_B (rotated in file):   T_B @ CAD_B = Same observed pose!

How to use the rotation data:
  1. For visualization:
     Apply T_robot_from_cad to your centered CAD model (always correct)

  2. For rotation analysis (independent of CAD initial orientation):
     Compute relative rotations between frames (robot frame):
       R_relative[i] = R_object_in_robot[i] @ R_object_in_robot[0].T
     This gives rotation from frame 0 to frame i in robot frame.
     The CAD's initial orientation cancels out in this product.
"""

import re
import cv2
import numpy as np
import os
import torch
import matplotlib
from tqdm import tqdm
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import open3d as o3d
import trimesh
from omegaconf import OmegaConf

from utils.samutils import search_folder
from utils.depth_utils import load_raw_depth_episode
from utils.hw_camera import orbbec_fx, orbbec_fy, orbbec_cx, orbbec_cy

device = "cuda" if torch.cuda.is_available() else "cpu"


class ObjectTracker:
    """Handles 3D object tracking with ICP and point cloud processing."""
    
    def __init__(self, config):
        self.cfg = config
        self.camera_intrinsics = {
            'fx': orbbec_fx, 'fy': orbbec_fy,
            'cx': orbbec_cx, 'cy': orbbec_cy
        }
        self.camera_to_robot_transform = np.load(
            config.CAMERA_TO_ROBOT_TRANSFORM_PATH, allow_pickle=True
        )
        
    
    def depth_to_point_cloud(self, depth_image, mask):
        """Convert depth image to point cloud using camera intrinsics."""
        h, w = depth_image.shape
        v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        valid_mask = mask[:, :, 0] > 0
        valid_depth = depth_image[valid_mask]
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        
        z = valid_depth.astype(np.float32) / 1000.0  # Convert to meters
        x = (valid_u - self.camera_intrinsics['cx']) * z / self.camera_intrinsics['fx']
        y = (valid_v - self.camera_intrinsics['cy']) * z / self.camera_intrinsics['fy']
        
        points = np.stack([x, y, z], axis=-1)
        centroid = np.mean(points, axis=0)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        return pcd, centroid
    
    def transform_to_robot_frame(self, pcd, centroid):
        """Transform point cloud and centroid to robot frame."""
        pcd_robot = o3d.geometry.PointCloud(pcd)
        pcd_robot.transform(self.camera_to_robot_transform)
        
        centroid_homogeneous = np.append(centroid, 1)
        centroid_robot = (self.camera_to_robot_transform @ centroid_homogeneous)[:3]
        
        return pcd_robot, centroid_robot
    
    def generate_masks_with_sam3(self, images, dataset_indexes, temp_dir, predictor_cfg, language_prompt=None):
        """Generate object masks using SAM3."""
        from sam3.model_builder import build_sam3_video_predictor
        
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save temporary images
        for dataset_index, img in zip(dataset_indexes, images):
            cv2.imwrite(os.path.join(temp_dir, f"Color_{dataset_index}.png"), img)
        
        print(f"Initializing SAM3 video predictor...")
        video_predictor = build_sam3_video_predictor()
        
        # Start SAM3 session
        print(f"Starting SAM3 session with video path: {temp_dir}")
        response = video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=temp_dir
            )
        )
        session_id = response["session_id"]
        print(f"Session started with ID: {session_id}")
        
        # Add text prompt (use language_prompt if provided, otherwise default)
        text_prompt = language_prompt if language_prompt is not None else "object"
        print(f"Adding text prompt '{text_prompt}' at frame 0")
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=text_prompt,
            )
        )
        
        print("SAM3 propagating masks through video...")
        
        # Propagate masks through all frames
        outputs_per_frame = {}
        for response in video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
                
        
        # Extract masks from outputs
        masks = []
        for i in range(len(images)):
            if i in outputs_per_frame and "out_binary_masks" in outputs_per_frame[i] and outputs_per_frame is not []:
                mask = outputs_per_frame[i]["out_binary_masks"][0]
                # Convert to uint8 and scale to 0-255
                mask_img = (mask.astype(np.uint8) * 255)
                # Convert grayscale to RGB for consistency
                mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
                masks.append(mask_rgb)
            else:
                # Create empty mask if none found
                empty_mask = np.zeros_like(images[i])
                masks.append(empty_mask)
                print(f"Warning: No mask found for frame {i}")
        
        # Close SAM3 session
        video_predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )
        print(f"SAM3 session ended. Generated {len(masks)} masks")
        
        return np.array(masks)
    
    def visualize_alignment(self, rgb_image, depth_image, mask, depth_width, depth_height):
        """Visualize mask-depth alignment for verification."""
        rgb_resized = cv2.resize(rgb_image, (depth_width, depth_height), 
                                interpolation=cv2.INTER_NEAREST)
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, 
                                        cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        rgb_overlay = cv2.addWeighted(rgb_resized, 0.6, mask, 0.4, 0)
        rgbd = cv2.addWeighted(rgb_resized, 0.6, depth_colored, 0.4, 0)
        depth_overlay = cv2.addWeighted(depth_colored, 0.6, mask, 0.4, 0)
        
        # comparison = np.hstack([rgb_overlay, rgbd, depth_overlay])
        breakpoint()
        cv2.imshow('RGB+Mask', rgb_overlay)
        cv2.waitKey(0)

        cv2.imshow('RGB+Depth', rgbd)
        cv2.imshow('Depth+Mask', depth_overlay)
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def process_point_clouds(self, depth_images, masks, rgb_images=None):
        """
        Convert depth images and masks to point clouds in CAMERA FRAME.

        Keeps point clouds in camera frame for ICP alignment.
        Final transformation to robot frame happens in align_with_icp().
        """
        depth_height, depth_width = depth_images[0].shape
        point_clouds, centroids = [], []

        for i, (depth_img, mask) in enumerate(zip(depth_images, masks)):
            mask_resized = cv2.resize(mask, (depth_width, depth_height),
                                     interpolation=cv2.INTER_NEAREST)

            # if i == 0 and rgb_images is not None:
            #     self.visualize_alignment(rgb_images[i], depth_img, mask_resized,
            #                            depth_width, depth_height)

            pcd_camera, pcd_centroid_in_camera = self.depth_to_point_cloud(depth_img, mask_resized) #in camera frame

            point_clouds.append(pcd_camera)
            centroids.append(pcd_centroid_in_camera)

            if i % 10 == 0:
                print(f"Processed point cloud {i}/{len(depth_images)}")

        return point_clouds, centroids
    
    def load_cad_model(self, cad_path, num_samples, scale=0.001):
        """Load and prepare CAD model."""
        cad_mesh = trimesh.load(cad_path)
        cad_points = cad_mesh.sample(num_samples)
        
        cad_pcd = o3d.geometry.PointCloud()
        cad_pcd.points = o3d.utility.Vector3dVector(cad_points)
        
        # Apply initial pose and centering
        cad_centroid = np.mean(np.asarray(cad_pcd.points), axis=0)
        cad_pcd.translate(-cad_centroid)
        cad_pcd.scale(scale, center=(0, 0, 0))
        
        return cad_pcd, cad_mesh
    
    def align_with_icp(self, point_clouds, centroids, cad_pcd, voxel_size=0.005):
        """
        Align CAD model to observations using ICP.

        REFERENCE FRAME TRACKING:
        =========================
        Input:
          - point_clouds: In CAMERA FRAME
          - centroids: In CAMERA FRAME
          - cad_pcd: Centered at origin in its own frame

        Processing:
          - ICP performed in CAMERA FRAME (with both source and target centered at origin)
          - Each frame processed INDEPENDENTLY (no temporal tracking)

        Output:
          - All transforms and positions in ROBOT FRAME
          - Clear naming distinguishes camera vs robot frame data

        Args:
            point_clouds: List of point clouds in CAMERA FRAME
            centroids: List of centroids (3D) in CAMERA FRAME
            cad_pcd: CAD point cloud centered at origin
            voxel_size: Voxel size for downsampling

        Returns:
            Dictionary with:
              - Transforms in ROBOT FRAME
              - ICP rotations in CENTERED/CAMERA FRAME
              - Positions in both CAMERA and ROBOT FRAME
        """
        # Store CAD centroid (should be near zero since centered)
        cad_centroid_in_cad_frame = np.mean(np.asarray(cad_pcd.points), axis=0)

        # Downsample CAD for ICP
        cad_down = cad_pcd.voxel_down_sample(voxel_size)
        cad_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
        )

        results = {
            # --- Transforms in ROBOT FRAME (final output) ---
            'T_robot_from_cad': [],         # (N, 4, 4) Transform: centered CAD → robot frame

            # --- Positions ---
            'object_pos_in_camera': [],     # (N, 3) Object centroid in camera frame
            'object_pos_in_robot': [],      # (N, 3) Object centroid in robot frame

            # --- Rotations ---
            'R_icp_in_centered_space': [],  # (N, 3, 3) ICP rotation: centered CAD → centered obs
            'R_object_in_robot': [],        # (N, 3, 3) Object orientation in robot frame

            # --- Trajectory (in robot frame) ---
            'displacement_from_start': [],  # (N, 3) Translation from first tracked position

            # --- Quality metrics ---
            'icp_fitness': [],              # (N,) ICP fitness score per frame
            'frame_idx': [],                # (N,) Frame index in episode
        }

        # Track first position in robot frame for trajectory
        first_pos_robot = None

        print(f"\nApplying ICP alignment (camera frame → robot frame output)")

        for i, (pcd_camera, centroid_camera) in enumerate(zip(point_clouds, centroids)):
            # ===== STEP 1: ICP in centered space (camera frame) =====
            # Center observation at origin (same as CAD)
            pcd_centered = o3d.geometry.PointCloud(pcd_camera)
            pcd_centered.translate(-pcd_centered.get_center())

            # Downsample for ICP
            pcd_down = pcd_centered.voxel_down_sample(voxel_size)
            pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
            )

            # Run ICP: centered CAD → centered observation
            # Always starts from identity (independent per frame)
            reg_result = o3d.pipelines.registration.registration_icp(
                cad_down, pcd_down,
                max_correspondence_distance=0.05,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6
                )
            )

            # Extract rotation from ICP (in centered/camera space)
            R_icp = reg_result.transformation[:3, :3]

            # ===== STEP 2: Build transform in camera frame =====
            # T_camera = [R_icp | centroid_camera]
            #            [  0   |        1       ]
            T_camera_from_cad = np.eye(4)
            T_camera_from_cad[:3, :3] = R_icp
            T_camera_from_cad[:3, 3] = centroid_camera

            # ===== STEP 3: Transform to robot frame =====
            # T_robot = T_robot_from_camera @ T_camera_from_cad
            T_robot_from_cad = self.camera_to_robot_transform @ T_camera_from_cad

            # Extract position and rotation in robot frame
            pos_robot = T_robot_from_cad[:3, 3]
            R_robot = T_robot_from_cad[:3, :3]

            # Track first position for trajectory
            if first_pos_robot is None:
                first_pos_robot = pos_robot.copy()

            # ===== STEP 4: Store results =====
            results['T_robot_from_cad'].append(T_robot_from_cad)
            results['object_pos_in_camera'].append(centroid_camera)
            results['object_pos_in_robot'].append(pos_robot)
            results['R_icp_in_centered_space'].append(R_icp) #(source:cad, target:camera obs)
            results['R_object_in_robot'].append(R_robot)
            results['displacement_from_start'].append(pos_robot - first_pos_robot)
            results['icp_fitness'].append(reg_result.fitness)
            results['frame_idx'].append(i)

        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])

        # Add metadata
        results['cad_centroid_in_cad_frame'] = cad_centroid_in_cad_frame
        results['first_pos_robot'] = first_pos_robot

        print(f"\nTracked {len(results['frame_idx'])}/{len(point_clouds)} frames "
              f"(avg fitness: {np.mean(results['icp_fitness']):.4f})")

        return results
    
    def save_results(self, results, cad_info, output_dir, episode_num):
        """
        Save all tracking results to a single NPZ file per episode.

        Output file contains clearly named arrays with reference frame annotations.
        All spatial data explicitly indicates whether it's in camera or robot frame.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Combine all data with CLEAR, UNAMBIGUOUS key names
        tracking_data = {
            # ========== TRANSFORMS IN ROBOT FRAME ==========
            'T_robot_from_cad': results['T_robot_from_cad'],  # (N,4,4) Centered CAD → robot

            # ========== POSITIONS ==========
            'object_pos_in_camera': results['object_pos_in_camera'],  # (N,3) Camera frame
            'object_pos_in_robot': results['object_pos_in_robot'],    # (N,3) Robot frame

            # ========== ROTATIONS ==========
            'R_icp_in_centered_space': results['R_icp_in_centered_space'],  # (N,3,3) ICP result
            'R_object_in_robot': results['R_object_in_robot'],              # (N,3,3) Robot frame

            # ========== TRAJECTORY (Robot frame) ==========
            'displacement_from_start': results['displacement_from_start'],  # (N,3) From first position

            # ========== QUALITY METRICS ==========
            'icp_fitness': results['icp_fitness'],  # (N,) ICP fitness scores
            'frame_idx': results['frame_idx'],      # (N,) Frame indices

            # ========== CAD MODEL INFO ==========
            'cad_model_path': cad_info['model_path'],                       # str: Path to CAD file
            'cad_num_sample_points': cad_info['sample_points'],            # int: Points sampled
            'cad_centroid_in_cad_frame': results['cad_centroid_in_cad_frame'],  # (3,) CAD centroid
            'cad_scale_factor': cad_info.get('scale', 0.001),              # float: Scale applied

            # ========== REFERENCE TRANSFORMS ==========
            'T_robot_from_camera': self.camera_to_robot_transform,  # (4,4) Camera→robot transform
            'first_pos_robot': results['first_pos_robot'],          # (3,) First tracked position

            # ========== METADATA ==========
            'episode_number': episode_num,                                  # int
            'num_frames_tracked': len(results['frame_idx']),               # int
            'num_frames_total': results.get('total_frames', len(results['frame_idx']))  # int
        }

        # Save as single NPZ file
        output_file = os.path.join(output_dir, f"episode_{episode_num:02d}_tracking.npz")
        np.savez(output_file, **tracking_data)

        print(f"\nSaved tracking data to: {output_file}")
        print(f"  - {len(results['frame_idx'])} frames tracked")
        print(f"  - Average fitness: {np.mean(results['icp_fitness']):.4f}")

        return output_file
    
    def plot_trajectory(self, positions, frame_indices, output_dir, fitness_threshold):
        """Generate and save trajectory visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Position over time
        axes[0].plot(frame_indices, positions[:, 0], 'r-', label='X', linewidth=2)
        axes[0].plot(frame_indices, positions[:, 1], 'g-', label='Y', linewidth=2)
        axes[0].plot(frame_indices, positions[:, 2], 'b-', label='Z', linewidth=2)
        axes[0].set_xlabel('Frame Index', fontsize=12)
        axes[0].set_ylabel('Position (m)', fontsize=12)
        axes[0].set_title(f'Object Position Over Time (fitness >= {fitness_threshold})', 
                         fontsize=14)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # X-Y trajectory
        axes[1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2)
        axes[1].scatter(positions[0, 0], positions[0, 1], c='green', s=150, 
                       marker='o', label='Start', zorder=5)
        axes[1].scatter(positions[-1, 0], positions[-1, 1], c='red', s=150, 
                       marker='x', label='End', zorder=5)
        axes[1].set_xlabel('X (m)', fontsize=12)
        axes[1].set_ylabel('Y (m)', fontsize=12)
        axes[1].set_title('X-Y Trajectory', fontsize=14)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].axis('equal')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, "trajectory_preview.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved trajectory preview to: {save_path}")
    
    def load_masks(self,directory):
        # get all png files from the given directory
        masks_paths =[]
        for root, dirs, files in os.walk(os.path.join(directory)):
            for file in files:
                if file.startswith("mask_") and file.endswith(".png"):
                    masks_paths.append(os.path.join(root, file))
        masks_paths = np.asarray(masks_paths)
        def natural_key(string_):
            """Helper to sort strings like humans expect."""
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string_)]
        masks_paths = sorted(masks_paths, key=natural_key)  # <--- natural sort
        print(masks_paths)
        masks = [cv2.imread(p) for p in tqdm(masks_paths, desc="reading images")]
        return masks


def main():
    """Main execution pipeline."""
    # Load configurations
    OmegaConf.register_new_resolver("phantom-touch", 
                                   lambda: search_folder("/home", "phantom-touch"))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = search_folder("/home", "phantom-touch")
    threed_cfg = OmegaConf.load(f"{current_dir}/cfg/threeD_tracking_offline.yaml")
    paths_cfg = OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
    
    # Initialize tracker
    tracker = ObjectTracker(threed_cfg)
    
    # Generate episode paths
    episodes = [
        f"{paths_cfg.dataset_directory}/e{i}/{paths_cfg.metadata.experiment_name}_e{i}.{paths_cfg.metadata.dataset_format}"
        for i in range(threed_cfg.start_episode, threed_cfg.end_episode + 1)
    ]
    
    # Process each episode
    for episode_idx, episode_path in enumerate(episodes):
        episode_num = episode_idx + threed_cfg.start_episode
        print(f"\n{'='*60}\nProcessing Episode {episode_num}\n{'='*60}")
        try:
            episode_data = np.load(episode_path)
            if episode_data['image_0'].shape[0] < 10: # skip short episodes
                print(f"Episode {episode_num} is empty, skipping...")
                continue
            depth_images_per_episode = load_raw_depth_episode(
                paths_cfg.recordings_directory, episode_num, tuple(threed_cfg.depth_shape)
            )
            depth_images_per_episode = depth_images_per_episode[episode_data['indexes']] # remove filtered episode frames
            print(f"Loaded {len(depth_images_per_episode)} depth frames after applying epsisode filter from the dataset")
        except FileNotFoundError:
            print(f"File not found: {episode_path}")
            continue
        
        # prepare for object segmentation per episode
        rgb_images_per_episode = episode_data["original"]
        dataset_indexes_per_episode = episode_data['absolute_indexes']
        temp_dir = os.path.join(paths_cfg.dataset_directory, "temporary_images_extraction", f"e{episode_num}")
        # Clean temp directory for this episode if it exists
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate masks with SAM3 using text prompt
        if threed_cfg.mask_objects:
            language_prompt = threed_cfg.get('language_prompt', default_value='object')
            masks_per_episode = tracker.generate_masks_with_sam3(
                rgb_images_per_episode, dataset_indexes_per_episode,temp_dir, threed_cfg.get('sam3videoPredictor', {}),
                language_prompt=language_prompt
            )
            # Save masks in episode-specific folder
            episode_mask_dir = os.path.join(paths_cfg.object_masks_dir, f"e{episode_num}")
            os.makedirs(episode_mask_dir, exist_ok=True)
            for dataset_index, mask in zip(dataset_indexes_per_episode, masks_per_episode):
                cv2.imwrite(
                    os.path.join(episode_mask_dir, f"mask_frame_{dataset_index}.png"),
                    mask
                )
            print(f"Saved {len(masks_per_episode)} masks to {episode_mask_dir}")
        ############################################################################################
        
        
        ### second phase: object pose tracking
        # load saved masks
        if threed_cfg.load_masks:
            episode_mask_dir = os.path.join(paths_cfg.object_masks_dir, f"e{episode_num}")
            print(f"Loading masks from {episode_mask_dir}...")
            masks_per_episode = tracker.load_masks(episode_mask_dir)
            print(f"Loaded {len(masks_per_episode)} masks")
        
        # Process point clouds
        print("\nConverting depth to point clouds...")
        point_clouds_in_camera, centroids_in_camera = tracker.process_point_clouds(
            depth_images_per_episode, masks_per_episode, rgb_images_per_episode
        )
        # Load CAD model
        print(f"\nLoading CAD model from {threed_cfg.CAD_MODEL_PATH}...")
        cad_pcd, cad_mesh = tracker.load_cad_model(
            threed_cfg.CAD_MODEL_PATH, 
            num_samples=threed_cfg.get('cad_sample_points', 10000),
            scale=threed_cfg.get('cad_scale', 0.001)
        )
        print(f"CAD model loaded with {len(cad_mesh.vertices)} vertices")
        
        # Align with ICP
        icp_cfg = threed_cfg.get('icp', {})
        results = tracker.align_with_icp(
            point_clouds_in_camera, centroids_in_camera, cad_pcd,
            voxel_size=icp_cfg.get('voxel_size', 0.005)
        )
        
        # Store total frames for metadata
        results['total_frames'] = len(point_clouds_in_camera)
        
        # Save results
        cad_info = {
            'model_path': threed_cfg.CAD_MODEL_PATH,
            'sample_points': threed_cfg.get('cad_sample_points', 10000),
            'scale': threed_cfg.get('cad_scale', 0.001)
        }

        tracker.save_results(results, cad_info, paths_cfg.threeD_tracking_output_dir, episode_num)

        # Generate trajectory plot (use robot frame positions)
        if len(results['object_pos_in_robot']) > 0:
            tracker.plot_trajectory(
                results['object_pos_in_robot'], results['frame_idx'],
                paths_cfg.threeD_tracking_output_dir,
                fitness_threshold=icp_cfg.get('fitness_threshold', 0.5)
            )
        
        print(f"\n✓ Episode {episode_num} completed successfully")


if __name__ == "__main__":
    main()
