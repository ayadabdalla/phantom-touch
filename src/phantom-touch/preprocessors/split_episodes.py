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
paths_cfg=OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
preprocess_cfg=OmegaConf.load(f"{repo_dir}/src/phantom-touch/cfg/preprocessors.yaml")
path=f"{paths_cfg.vitpose_output_directory}/no_episode_split"

class Preprocessor: 
    def __init__(self,config):
        self.config = config
        self.split_threshold = config.split_threshold
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
            if frame_numbers[i] - frame_numbers[i - 1] <= 2:
                current_episode.append(frame_numbers[i])
            else:
                # TODO: count the episodes that are saved
                if len(current_episode) > 10: # minimum episode length
                    episodes[f"e{j}"] = current_episode
                    j+=1
                else:
                    # If the current episode is too short, discard it
                    print("Discarding short episode:", len(current_episode))
                current_episode = [frame_numbers[i]]
        if len(current_episode) > 10:
            episodes[f'e{j}']=current_episode
        return episodes

    def get_episodes(self):
        """ a getter for the episodes dictionary """
        return self.episodes
    
    def split_recordings(self, recordings_path):
        if self.episodes is not None:
            output_path = recordings_path
            j = 0
            for i, episode in enumerate(tqdm(self.episodes.values(), desc="Processing episodes")):
                # create a new folder for each episode
                folder_name = os.path.join(output_path, f"e{j}")
                os.makedirs(folder_name, exist_ok=True)
                # copy the files to the new folder
                for frame_number in tqdm(episode, desc=f"Copying frames for episode {j}", leave=False):
                    # get all files that include the frame number
                    frame_number = str(frame_number).zfill(5)
                    files = glob.glob(os.path.join(recordings_path, f"*{frame_number}*"))
                    for file in files:
                        shutil.move(file, folder_name)
                j += 1
        

if __name__ == "__main__":
    data_preprocessor = Preprocessor(preprocess_cfg)
    if preprocess_cfg.split_episodes:
        episodes = data_preprocessor.split_episodes(path)
    else:
        episodes = data_preprocessor.get_episodes()
print(len(episodes))
print(episodes[-1][-1])
# recordings path
# recordings_path=f'{paths_cfg.vitpose_output_directory}/no_episode_split'
# output_path = paths_cfg.vitpose_output_directory
# j = 0
# for i, episode in enumerate(tqdm(episodes, desc="Processing episodes")):
#     # create a new folder for each episode
#     folder_name = os.path.join(output_path, f"e{j}")
#     os.makedirs(folder_name, exist_ok=True)
#     # copy the files to the new folder
#     for frame_number in tqdm(episode, desc=f"Copying frames for episode {j}", leave=False):
#         # get all files that include the frame number
#         frame_number = str(frame_number).zfill(5)
#         files = glob.glob(os.path.join(recordings_path, f"*{frame_number}*"))
#         for file in files:
#             shutil.move(file, folder_name)
#     j += 1