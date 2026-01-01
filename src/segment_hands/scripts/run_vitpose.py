# global imports
import os
import sys
from omegaconf import OmegaConf
import cv2
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import torch
import hamer
from pathlib import Path
import numpy as np
from hamer.vitpose_model import ViTPoseModel
import re

# local imports
from utils.rgb_utils import load_rgb_images
from utils.samutils import search_folder
from tqdm import tqdm

if __name__ == "__main__":
    # script setup: configs, paths, and device
    user = os.getenv("USER")
    repo_dir = search_folder(f"/home/{user}/", "phantom-touch")
    vitpose_config = OmegaConf.load(f"{repo_dir}/src/segment_hands/cfg/vitpose_segmentation.yaml")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    detectron2_cfg_path = (
        Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    )
    detectron2_cfg = LazyConfig.load(str(detectron2_cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"

    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    vitpose_model = ViTPoseModel(device)

    # Processing mode: True = process all images directly from experiment folder
    #                  False = process pre-split episode directories (not implemented)
    process_continuous_data = True
    
    if process_continuous_data:
        # Helper function for natural/human sorting (e.g., img1, img2, img10 instead of img1, img10, img2)
        def natural_key(string_):
            """Sort strings with numbers in natural order (1, 2, 10 not 1, 10, 2)."""
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string_)]
        
        # Gather all image files from the experiment folder matching configured file types
        print(f"Searching for images in: {vitpose_config.img_folder}")
        img_paths = [img for end in vitpose_config.file_type for img in Path(vitpose_config.img_folder).glob(end)]
        img_paths = [str(path) for path in img_paths]  # Convert Path objects to strings
        img_paths = sorted(img_paths, key=natural_key)  # Sort naturally by number
        print(f"Found {len(img_paths)} images to process")


        # Process each image: detect humans → detect hand keypoints → save
        for i, img_path in tqdm(enumerate(img_paths), desc="Processing images", total=len(img_paths)):
            # Load image from disk
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not load image {img_path}")
                continue
            
            if vitpose_config.crop.flag:
                # Crop image to region of interest (removes irrelevant parts)
                image_crop = image[vitpose_config.crop.y1:vitpose_config.crop.y2, 
                                  vitpose_config.crop.x1:vitpose_config.crop.x2]
            else:
                image_crop = image
            
            # Show crop preview on first image for verification
            if i == 0 and vitpose_config.crop.show_crop:
                print("\n" + "="*60)
                print("PREVIEW: First image crop verification")
                print("="*60)
                cv2.imshow("Original Image (Full Frame)", image)
                cv2.imshow("Cropped Image (ROI for Processing)", image_crop)
                print(f"Image shape: {image.shape}")
                print(f"Crop region: y[{vitpose_config.crop.y1}:{vitpose_config.crop.y2}], x[{vitpose_config.crop.x1}:{vitpose_config.crop.x2}]")
                print(f"Cropped shape: {image_crop.shape}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                user_input = input("\nDoes the crop look correct? Continue processing? (y/n): ").strip().lower()
                if user_input != 'y':
                    print("Processing stopped by user. Check crop settings in config.")
                    break
                print("="*60 + "\n")

            # Resize image to standard resolution for detection
            img_resized = cv2.resize(image_crop, 
                                    (vitpose_config.resolution.width, vitpose_config.resolution.height))
            
            # Step 1: Detect humans using Detectron2 (finds bounding boxes around people)
            detection_output = detector(img_resized)
            img_rgb = img_resized.copy()[:, :, ::-1]  # Convert BGR to RGB for VitPose

            # Filter detections: only keep humans (class 0) with confidence > 0.5
            det_instances = detection_output["instances"]
            is_human = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            human_bboxes = det_instances.pred_boxes.tensor[is_human].cpu().numpy()
            human_scores = det_instances.scores[is_human].cpu().numpy()

            # Step 2: Detect hand keypoints for each detected person using VitPose
            vitposes_out = vitpose_model.predict_pose(
                img_rgb,
                [np.concatenate([human_bboxes, human_scores[:, None]], axis=1)],
            )
            
            # Skip this frame if no poses detected
            if len(vitposes_out) == 0:
                continue

            # Step 3: Extract and save hand keypoints for each detected person
            for person_idx, pose_data in enumerate(vitposes_out):
                # VitPose keypoints structure: body keypoints + left hand (21) + right hand (21)
                all_keypoints = pose_data["keypoints"]
                left_hand_keypoints = all_keypoints[-42:-21]   # Last 42 points are hands: first 21 = left
                right_hand_keypoints = all_keypoints[-21:]      # Last 21 points = right hand
                
                # Only save right hand if we have confident detections
                # Each keypoint has format [x, y, confidence]
                confidence_scores = right_hand_keypoints[:, 2]
                confident_keypoints = confidence_scores > 0.5
                
                # Require at least 4 confident keypoints to save (minimum for valid hand)
                if sum(confident_keypoints) > 3:
                    # Generate output filename based on input image name
                    original_filename = os.path.basename(img_path)  # e.g., "Color_0034.png"
                    base_name = os.path.splitext(original_filename)[0]  # Remove extension
                    output_filename = f"vitpose_keypoints_2d_{base_name}_right_{person_idx}.npy"
                    
                    # Save keypoints as numpy array
                    output_dir = vitpose_config.vitpose_out_folder
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, output_filename)
                    
                    np.save(output_path, right_hand_keypoints)
                    # print(f"✓ Saved: {output_filename}")  # Uncomment for verbose output
    else:
        # This branch would handle pre-split episode directories
        print("ERROR: Episode-based processing not implemented.")
        print("Please set process_continuous_data = True to process all images.")
        sys.exit(1)
