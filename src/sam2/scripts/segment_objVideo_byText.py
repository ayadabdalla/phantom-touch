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

#### script metadata ####
matplotlib.use("TKAgg")
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    #### initialize configurations ####
    OmegaConf.register_new_resolver(
        "phantom-touch", lambda: search_folder("/home", "phantom-touch"))
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sam2_text_cfg = OmegaConf.load(f"{parent_dir}/cfg/sam2_object_by_text.yaml")
    sam2_sieve_cfg = sam2_text_cfg.sam2sieve
    if not os.path.exists(sam2_sieve_cfg.output_dir):
        os.makedirs(sam2_sieve_cfg.output_dir)
    output_path = sam2_text_cfg.sam2videoPredictor.output_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get episodes addresses
    episodes = []
    for root, dirs, files in os.walk(sam2_sieve_cfg.images_path):
        if os.path.basename(root).startswith("e") and os.path.basename(root)[1:].isdigit():
            episodes.append(os.path.basename(root)) # episodes addresses
    episodes.sort(key=lambda x: int(x[1:]))  # sort by episode number


    os.makedirs(os.path.join(output_path, episodes[0]), exist_ok=True)
    print(f"Processing episode {episodes[0]}...")


    #### First workflow component: SAM2-SIEVE####
    video_name=f"color_video_compiled_for_sieve_{now}.mp4"
    sam = sieve.function.get("sieve/text-to-segment")
    sieve_images_path = os.path.join(sam2_sieve_cfg.images_path, episodes[0])
    input_video = filelist_to_mp4sieve(
        sieve_images_path,
        prefix="Color_",
        output_path=f"{sam2_sieve_cfg.output_dir}/{video_name}",
    )
    sam_out = sam.run(input_video, sam2_sieve_cfg.text_prompt)
    original_masks = sievesamzip_to_numpy(sam_out)
    #### integration interface with the next workflow component: VIDEO-SAM2####
    centroids = []
    for mask in original_masks:
        centroid = extract_centroid(mask)
        if centroid is not None:
            centroids.append(centroid)
    centroids = np.array(centroids)
    
    ## prepare for sam2 second workflow
    images_paths = []
    for episode in episodes:
        # collect all episodes images paths
        images_paths.append(os.path.join(sam2_sieve_cfg.images_path, episode))


    #### Second workflow component: VIDEO-SAM2 ####
    sam2video_cfg = sam2_text_cfg.sam2videoPredictor
    points = np.array(
        [[centroids[0][0], centroids[0][1]]], dtype=np.float32
    )  # interaction points from previous workflow component
    predictor = build_sam2_video_predictor(
        sam2video_cfg.model_cfg, sam2video_cfg.sam2_checkpoint, device=device
    )

    inference_state = predictor.init_state(video_path=images_paths)
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
    mask_frames = []
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


    #### store results ####
    frame_names = []
    images_paths = [os.path.join(sam2_sieve_cfg.images_path, ep) for ep in episodes]
    
    # save all masks and then split them per episode as the original images
    mask_idx = 0
    for episode in episodes:
        episode_input_dir = os.path.join(sam2_sieve_cfg.images_path, episode)
        episode_output_dir = os.path.join(output_path, episode)
        os.makedirs(episode_output_dir, exist_ok=True)
        
        # Get all color frames in this episode
        color_frames = sorted([f for f in os.listdir(episode_input_dir) if f.startswith("Color_")])
        
        # Save masks for this episode
        for frame_name in color_frames:
            if mask_idx < len(mask_frames):
                mask_output_path = os.path.join(episode_output_dir, frame_name.replace("Color_", "mask_"))
                cv2.imwrite(mask_output_path, mask_frames[mask_idx])
                mask_idx += 1
        
        print(f"Saved {len(color_frames)} masks for {episode}")