# workflow 2 sam2-sieve=>sam2-videoPredictor
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
    cfg = OmegaConf.load(f"{parent_dir}/cfg/sam2_object_by_text.yaml")
    sam2_sieve_cfg = cfg.sam2sieve
    if not os.path.exists(sam2_sieve_cfg.output_dir):
        os.makedirs(sam2_sieve_cfg.output_dir)
    output_path = cfg.sam2videoPredictor.output_dir
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Check if data is split into episodes or is unsplit
    episodes = []
    for root, dirs, files in os.walk(sam2_sieve_cfg.images_path):
        if os.path.basename(root).startswith("e") and os.path.basename(root)[1:].isdigit():
            episodes.append(os.path.basename(root))

    # If no episodes found, treat the entire directory as one unsplit dataset
    if len(episodes) == 0:
        print("No episode directories found. Processing unsplit data...")
        episodes = ["unsplit"]  # Dummy episode name for unsplit data
    else:
        episodes.sort(key=lambda x: int(x[1:]))  # sort by episode number

    for episode in episodes:
        print(f"Processing {'unsplit data' if episode == 'unsplit' else f'episode {episode}'}...")

        # Determine images path for this episode
        if episode == "unsplit":
            images_path = sam2_sieve_cfg.images_path
        else:
            images_path = os.path.join(sam2_sieve_cfg.images_path, episode)
            # Check if output already exists and skip if complete
            episode_output_path = os.path.join(output_path, episode)
            if os.path.exists(episode_output_path):
                input_file_count = len([f for f in os.listdir(images_path) if f.startswith('Color_')])
                output_file_count = len(os.listdir(episode_output_path))
                if output_file_count == int(input_file_count / 2 + 1):
                    print(f"Output directory {episode_output_path} already exists and is complete. Skipping...")
                    continue

        #### First workflow component: SAM2-SIEVE####
        # Run text-to-segment for this specific episode
        video_name = f"color_video_compiled_for_sieve_{episode}_{now}.mp4"
        sam = sieve.function.get("sieve/text-to-segment")
        input_video = filelist_to_mp4sieve(
            images_path,
            prefix="Color_",
            output_path=f"{sam2_sieve_cfg.output_dir}/{video_name}",
        )
        sam_out = sam.run(input_video, sam2_sieve_cfg.text_prompt)
        original_masks = sievesamzip_to_numpy(sam_out)

        #### integration interface with the next workflow component: VIDEO-SAM2####
        # Extract centroids for this specific episode
        centroids = []
        for mask in original_masks:
            centroid = extract_centroid(mask)
            if centroid is not None:
                centroids.append(centroid)
        centroids = np.array(centroids)

        if len(centroids) == 0:
            print(f"No centroids found for {episode}, skipping...")
            continue

        #### Second workflow component: VIDEO-SAM2 ####
        sam2video_cfg = cfg.sam2videoPredictor
        points = np.array(
            [[centroids[0][0], centroids[0][1]]], dtype=np.float32
        )  # interaction points from previous workflow component
        predictor = build_sam2_video_predictor(
            sam2video_cfg.model_cfg, sam2video_cfg.sam2_checkpoint, device=device
        )

        # Handle both split and unsplit data
        if episode == "unsplit":
            video_frames_path = sam2video_cfg.video_frames_dir
        else:
            video_frames_path = os.path.join(sam2video_cfg.video_frames_dir, episode)

        inference_state = predictor.init_state(video_path=video_frames_path)
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
        if episode == "unsplit":
            search_path = sam2_sieve_cfg.images_path
        else:
            search_path = os.path.join(sam2_sieve_cfg.images_path, episode)

        os.makedirs(search_path, exist_ok=True)

        # For unsplit data, find Color_ files directly in the path
        # For split data, look in episode subdirectories
        if episode == "unsplit":
            # Direct files in the directory
            for file in sorted(os.listdir(search_path)):
                if file.startswith('Color_') and file.endswith(".png"):
                    frame_names.append(os.path.join(search_path, file))
        else:
            # Files in episode subdirectory - look directly in the episode folder
            for file in sorted(os.listdir(search_path)):
                if file.startswith('Color_') and file.endswith(".png"):
                    frame_names.append(os.path.join(search_path, file))

        frame_names.sort(key=lambda p: os.path.splitext(p)[0])

        # now we have frame_names and mask_frames, next is to store them
        for i in range(mask_frames.shape[0]):
            frame_name = frame_names[i]
            # Generate output path
            if episode == "unsplit":
                # For unsplit data, save directly to output directory
                output_filename = os.path.basename(frame_name).replace("Color_", "Mask_")
                output_file_path = os.path.join(output_path, output_filename)
            else:
                # For split data, maintain episode structure
                output_filename = os.path.basename(frame_name).replace("Color_", "Mask_")
                episode_output_path = os.path.join(output_path, episode)
                output_file_path = os.path.join(episode_output_path, output_filename)

            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            print(f"Saving {output_file_path}")
            cv2.imwrite(output_file_path, mask_frames[i])