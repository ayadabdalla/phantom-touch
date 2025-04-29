# global imports
import os
from omegaconf import OmegaConf
import cv2
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import torch
import hamer
from pathlib import Path
import numpy as np
from hamer.vitpose_model import ViTPoseModel

# local imports
from utils.rgb_utils import load_rgb_images
from utils.sam2utils import search_folder
from tqdm import tqdm

# script setup: configs, paths, and device
user = os.getenv("USER")
repo_dir = search_folder(f"/home/{user}/", "phantom-touch")
config = OmegaConf.load(f"{repo_dir}/src/hamer/conf/hamer_segmentation.yaml")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cfg_path = (
    Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
)
detectron2_cfg = LazyConfig.load(str(cfg_path))
detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"

for i in range(3):
    detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
detector = DefaultPredictor_Lazy(detectron2_cfg)
images,img_paths = load_rgb_images(config.img_folder, prefix="Color_", return_path=True)
print(f"Image shape: {images.shape}")
cpm = ViTPoseModel(device)

for i,img_path in tqdm(enumerate(img_paths), desc="Processing images"):
    img_cv2 = cv2.imread(str(img_path))

    # Detect humans in image
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person

    vitposes_out = cpm.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    # Use hands based on hand keypoint detections
    for idx,vitposes in enumerate(vitposes_out):
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]
        # Rejecting not confident detections
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            # === Save ===
            original_path = img_paths[i]  # Full path
            episode_name = os.path.basename(os.path.dirname(original_path))  # e.g., "e0"
            original_filename = os.path.basename(original_path)  # e.g., "Color_0034.png"
            save_name = "vitpose_keypoints_2d_" + os.path.splitext(original_filename)[0] + f"_right_{idx}.npy"
            episode_output_dir = os.path.join(config.vitpose_out_folder, episode_name)
            os.makedirs(episode_output_dir, exist_ok=True)  # Create folder if it doesn't exist
            save_path = os.path.join(episode_output_dir, save_name)
            np.save(save_path, right_hand_keyp)
