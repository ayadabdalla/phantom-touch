# workflow 2 sam2-sieve=>sam2-videoPredictor
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
from utils.sam2utils import (
    extract_centroid,
    save_mp4video,
    filelist_to_mp4sieve,
    search_folder,
    sievesamzip_to_numpy,
)

# script metadata
matplotlib.use("TKAgg")
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize configurations
OmegaConf.register_new_resolver(
    "phantom-touch", lambda: search_folder("/home", "phantom-touch"))

# get parent directory of the current script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = OmegaConf.load(f"{parent_dir}/conf/sam2_segmentation.yaml")
sam2_sieve_cfg = cfg.sam2sieve
if not os.path.exists(sam2_sieve_cfg.output_dir):
    os.makedirs(sam2_sieve_cfg.output_dir)
output_path = cfg.sam2videoPredictor.output_dir
if not os.path.exists(output_path):
    os.makedirs(output_path)
# First workflow component: SAM2-SIEVE
video_name=f"color_video_compiled_for_sieve_{now}.mp4"
sam = sieve.function.get("sieve/text-to-segment")
print(sam2_sieve_cfg.images_path)
input_video = filelist_to_mp4sieve(
    sam2_sieve_cfg.images_path,
    prefix="Color_",
    output_path=f"{sam2_sieve_cfg.output_dir}/{video_name}",
)
sam_out = sam.run(input_video, sam2_sieve_cfg.text_prompt)
original_masks = sievesamzip_to_numpy(sam_out)
# integration interface with teh next workflow component: VIDEO-SAM2
centroids = []
for mask in original_masks:
    centroid = extract_centroid(mask)
    if centroid is not None:
        centroids.append(centroid)
centroids = np.array(centroids)

# Second workflow component: VIDEO-SAM2
sam2video_cfg = cfg.sam2videoPredictor
points = np.array(
    [[centroids[0][0], centroids[0][1]]], dtype=np.float32
)  # interaction points from previous workflow component
predictor = build_sam2_video_predictor(
    sam2video_cfg.model_cfg, sam2video_cfg.sam2_checkpoint, device=device
)

frame_names = []
for root, dirs, files in os.walk(os.path.join(sam2video_cfg.video_frames_dir)):
    if os.path.basename(root).startswith("e"):
        for file in files:
            if file.startswith('Color_') and file.endswith(".png"):
                frame_names.append(os.path.join(root, file))
frame_names.sort(key=lambda p: os.path.splitext(p)[0])

inference_state = predictor.init_state(video_path=sam2video_cfg.video_frames_dir)
predictor.reset_state(inference_state)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0, # the first frame
    obj_id=1, # the object id we are interested in
    points=points,
    labels=np.array(
    [1], np.int32
)   # here we assume one hand(object) per frame, so we only need one label
)
mask_frames = []  # store the masked video for later use
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
    inference_state
):
    video_frame = np.zeros(
        (out_mask_logits.shape[2], out_mask_logits.shape[3], 3), dtype=np.uint8
    )
    for i, out_obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0] # squeeze the first dimension
        mask = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
    mask_frames.append(mask)
mask_frames = np.array(mask_frames)

# save each frame as a separate image
for i in range(mask_frames.shape[0]):
    frame_name = os.path.join(output_path, f"{frame_names[i]}")
    # check for the word recordings in the filename and replace it with the word output
    frame_name = frame_name.replace("recordings", "output")
    frame_name = frame_name.replace("Color_", "Mask_")
    frame_name = frame_name.replace("episodes", "sam2-vid_output/episodes")
    os.makedirs(os.path.dirname(frame_name), exist_ok=True)
    print(f"Saving {frame_name}")
    cv2.imwrite(frame_name, mask_frames[i])