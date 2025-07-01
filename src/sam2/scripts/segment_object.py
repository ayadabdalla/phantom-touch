import sys
import sieve
import cv2
import os
import numpy as np
import torch
import matplotlib
import datetime
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2_video_predictor
from utils.rgb_utils import load_rgb_images
from utils.sam2utils import (
    extract_centroid,
    save_mp4video,
    filelist_to_mp4sieve,
    search_folder,
    sievesamzip_to_numpy,
)

#### Script metadata ####
matplotlib.use("TKAgg")
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

#### Config loading ####
OmegaConf.register_new_resolver("phantom-touch", lambda: search_folder("/home", "phantom-touch"))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = OmegaConf.load(f"{parent_dir}/conf/sam2_object.yaml")
sam2_sieve_cfg = cfg.sam2sieve
sam2video_cfg = cfg.sam2videoPredictor

os.makedirs(sam2_sieve_cfg.output_dir, exist_ok=True)
os.makedirs(sam2video_cfg.output_dir, exist_ok=True)

# Use the flat directory (no episodes)
images_path = sam2_sieve_cfg.images_path  # directly set in config
video_name = f"color_video_compiled_for_sieve_{now}.mp4"

#### First workflow component: SAM2-SIEVE ####
# print("Running SAM2-SIEVE...")
# sam = sieve.function.get("sieve/text-to-segment")
# input_video = filelist_to_mp4sieve(
#     images_path,
#     prefix="frame_0",
#     output_path=os.path.join(sam2_sieve_cfg.output_dir, video_name),
#     episodes=False  # Use flat directory structure
# )
# sam_out = sam.run(input_video, sam2_sieve_cfg.text_prompt)
# original_masks = sievesamzip_to_numpy(sam_out)

# centroids = [extract_centroid(mask) for mask in original_masks if extract_centroid(mask) is not None]
# centroids = np.array(centroids)

images = load_rgb_images(base_dir=images_path, return_path=False, episodes=False, BGR=False)
centroids = np.array([[300, 480-130]])

#### Second workflow component: VIDEO-SAM2 ####
print("Running VIDEO-SAM2...")
points = np.array([[centroids[0][0], centroids[0][1]]], dtype=np.float32)
predictor = build_sam2_video_predictor(sam2video_cfg.model_cfg, sam2video_cfg.sam2_checkpoint, device=device)

inference_state = predictor.init_state(video_path=images_path)
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
    [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".png")],
    key=lambda p: os.path.splitext(os.path.basename(p))[0]
)
print(f"Number of frames: {len(frame_names)}")

output_dir = sam2video_cfg.output_dir

os.makedirs(output_dir, exist_ok=True)
for i, frame_path in enumerate(frame_names):
    frame_name = os.path.join(output_dir, f"mask_{os.path.basename(frame_path)}")
    print(f"Saving {frame_name}")
    cv2.imwrite(frame_name, mask_frames[i])
