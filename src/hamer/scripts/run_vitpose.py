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
img_paths = [
    img for end in config.file_type for img in Path(config.img_folder).glob(end)
]
cpm = ViTPoseModel(device)

for img_path in tqdm(img_paths, desc="Processing images"):
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
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]
        # Rejecting not confident detections
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            right_string = f"{img_path.stem}_right"
            keypoints_2d_path = f"{config.out_folder}/vitpose_keypoints_2d_{right_string}"
            np.save(keypoints_2d_path, right_hand_keyp)
