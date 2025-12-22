#standard libraries
import numpy as np
from omegaconf import OmegaConf
import os
import glob
import json
import shutil
import re
from tqdm import tqdm
from utils.sam2utils import search_folder


repo_dir = search_folder("/home", "phantom-touch")

class Preprocessor: 
    def __init__(self,main_cfg, paths_cfg):
        self.paths_cfg = paths_cfg
        self.config = main_cfg
        self.split_threshold = main_cfg.split_threshold
        self.minimum_frames = main_cfg.minimum_frames
        self.recording_path = os.path.dirname(paths_cfg.recordings_directory)
        self.output_path = paths_cfg.recordings_directory
        self.episodes = {}

    def split_episodes(self,path):
        files = glob.glob(path + "/*.npy") # get all files in the directory
        frame_numbers = [f.split("_")[-3].split(".")[0] for f in files]  # get the frame numbers from the file names
        frame_numbers = sorted(set(int(f) for f in frame_numbers)) # Ensure frame_numbers are integers and sorted
        episodes = {}
        current_episode = [frame_numbers[0]]
        j=0
        for i in range(1, len(frame_numbers)):
            # split if you didn't get hands for two consecutive frames
            if frame_numbers[i] - frame_numbers[i - 1] <= self.config.split_threshold:
                current_episode.append(frame_numbers[i])
            else:
                # TODO: count the episodes that are saved
                if len(current_episode) > self.config.minimum_frames: # minimum episode length
                    episodes[f"e{j}"] = current_episode
                    j+=1
                else:
                    # If the current episode is too short, discard it
                    print("Discarding short episode:", len(current_episode))
                current_episode = [frame_numbers[i]]
        if len(current_episode) > self.config.minimum_frames:
            episodes[f'e{j}']=current_episode
        self.episodes = episodes
        return episodes

    def get_episodes(self):
        """ a getter for the episodes dictionary """            
        return self.episodes
    
    def move_recordings(self):
        if self.episodes is not None:
            output_path = self.output_path

            j = 0
            for i, episode in enumerate(tqdm(self.episodes.values(), desc="Processing episodes")):
                # create a new folder for each episode
                output_folder = os.path.join(output_path, f"e{j}")
                os.makedirs(output_folder, exist_ok=True)
                # copy the files to the new folder
                for frame_number in tqdm(episode, desc=f"Copying frames for episode {j}", leave=False):
                    # get all files that include the frame number
                    frame_number = str(frame_number).zfill(5)
                    files = glob.glob(os.path.join(self.recording_path, f"*{frame_number}*"))
                    for file in files:
                        shutil.move(file, output_folder)
                j += 1
    
    def move_vitpose_outputs(self):
        if self.episodes is not None:
            output_path = self.paths_cfg.vitpose_output_directory

            j = 0
            for i, episode in enumerate(tqdm(self.episodes.values(), desc="Processing episodes")):
                # create a new folder for each episode
                output_folder = os.path.join(output_path, f"e{j}")
                os.makedirs(output_folder, exist_ok=True)
                # copy the files to the new folder
                for frame_number in tqdm(episode, desc=f"Copying vitpose outputs for episode {j}", leave=False):
                    # get all files that include the frame number
                    frame_number = str(frame_number).zfill(5)
                    files = glob.glob(os.path.join(paths_cfg.vitpose_output_directory, f"*{frame_number}*"))
                    for file in files:
                        shutil.move(file, output_folder)
                j += 1

    def move_sam2_hand_masks(self):
        """Move SAM2 hand masks to episode folders"""
        if self.episodes is not None:
            # SAM2 hand masks are saved to sam2-vid_output (unsplit)
            # Need to get parent of experiment dir and look for sam2-vid_output
            experiment_dir = os.path.dirname(self.paths_cfg.recordings_directory)
            sam2_hand_source = os.path.join(experiment_dir, "sam2-vid_output")
            sam2_hand_output = self.paths_cfg.sam2hand_output_directory

            if not os.path.exists(sam2_hand_source):
                print(f"SAM2 hand masks source directory not found: {sam2_hand_source}")
                return

            j = 0
            for i, episode in enumerate(tqdm(self.episodes.values(), desc="Processing SAM2 hand mask episodes")):
                # create a new folder for each episode
                output_folder = os.path.join(sam2_hand_output, f"e{j}")
                os.makedirs(output_folder, exist_ok=True)
                # move the files to the new folder
                for frame_number in tqdm(episode, desc=f"Moving SAM2 hand masks for episode {j}", leave=False):
                    # get all files that include the frame number
                    frame_number = str(frame_number).zfill(5)
                    # Look in sam2-vid_output for unsplit mask files
                    files = glob.glob(os.path.join(sam2_hand_source, f"*{frame_number}*"))
                    for file in files:
                        if os.path.isfile(file):  # Only move files, not directories
                            shutil.move(file, output_folder)
                j += 1

    def move_sam2_object_masks(self):
        """Move SAM2 object masks to episode folders"""
        if self.episodes is not None:
            # Object masks are in root object_masks_dir (unsplit)
            # After split, they should go to object_masks_dir/e{j}/
            object_masks_dir = self.paths_cfg.object_masks_dir

            if not os.path.exists(object_masks_dir):
                print(f"SAM2 object masks directory not found: {object_masks_dir}")
                return

            # Check if there are any mask files directly in the root directory
            mask_files = glob.glob(os.path.join(object_masks_dir, "Mask_*.png"))
            if len(mask_files) == 0:
                print(f"No mask files found in {object_masks_dir}")
                return

            j = 0
            for i, episode in enumerate(tqdm(self.episodes.values(), desc="Processing SAM2 object mask episodes")):
                # create a new folder for each episode
                output_folder = os.path.join(object_masks_dir, f"e{j}")
                os.makedirs(output_folder, exist_ok=True)
                # move the files to the new folder
                for frame_number in tqdm(episode, desc=f"Moving SAM2 object masks for episode {j}", leave=False):
                    # get all files that include the frame number
                    frame_number = str(frame_number).zfill(5)
                    # Look in root object_masks_dir for unsplit mask files
                    files = glob.glob(os.path.join(object_masks_dir, f"*{frame_number}*"))
                    for file in files:
                        if os.path.isfile(file):  # Only move files, not directories
                            shutil.move(file, output_folder)
                j += 1

    def read_episodes(self):
        """read episodes as save in move_recordings"""
        recording_path = self.paths_cfg.recordings_directory
        episode_dirs = [d for d in os.listdir(recording_path) if os.path.isdir(os.path.join(recording_path, d)) and d.startswith("e")]
        episodes = {}
        for episode_dir in episode_dirs:
            episode_path = os.path.join(recording_path, episode_dir)
            files = glob.glob(os.path.join(episode_path, "*.raw"))
            frame_numbers = [f.split("_")[-1].split(".")[0] for f in files]
            frame_numbers = sorted(set(int(f) for f in frame_numbers))
            episodes[episode_dir] = frame_numbers

        episode_mappings = {}
        for episode_name, frame_numbers in episodes.items():
            mapping = {idx:frame_number for idx, frame_number in enumerate(frame_numbers)}
            episode_mappings[episode_name] = mapping
        return episode_mappings

if __name__ == "__main__":
    paths_cfg=OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
    preprocess_cfg=OmegaConf.load(f"{repo_dir}/src/phantom_touch/cfg/preprocessors.yaml")
    data_preprocessor = Preprocessor(preprocess_cfg,paths_cfg=paths_cfg)
    if preprocess_cfg.split_episodes:
        episodes = data_preprocessor.split_episodes(paths_cfg.vitpose_output_directory)
        data_preprocessor.move_recordings()
        data_preprocessor.move_vitpose_outputs()

        # Move SAM2 hand masks if they exist
        experiment_dir = os.path.dirname(paths_cfg.recordings_directory)
        sam2_hand_source = os.path.join(experiment_dir, "sam2-vid_output")
        if os.path.exists(sam2_hand_source):
            print("Moving SAM2 hand masks...")
            data_preprocessor.move_sam2_hand_masks()

        # Move SAM2 object masks if they exist
        if os.path.exists(paths_cfg.object_masks_dir):
            print("Moving SAM2 object masks...")
            data_preprocessor.move_sam2_object_masks()

    print(len(episodes))