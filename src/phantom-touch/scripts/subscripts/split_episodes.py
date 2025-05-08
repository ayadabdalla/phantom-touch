import numpy as np

path="/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/vitpose_output/episodes/no_episode_split"
# get all files in the directory
import os
import glob
import json
import shutil
import re
from tqdm import tqdm

def split_episodes(path):
    # get all files in the directory
    files = glob.glob(path + "/*.npy")
    # get the number before _digit.npy
    frame_numbers = [f.split("_")[-3].split(".")[0] for f in files]
    # Ensure frame_numbers are integers and sorted
    frame_numbers = sorted(set(int(f) for f in frame_numbers))
    episodes = []
    current_episode = [frame_numbers[0]]

    for i in range(1, len(frame_numbers)):
        if frame_numbers[i] - frame_numbers[i - 1] <= 2:
            current_episode.append(frame_numbers[i])
        else:
            if len(current_episode) > 10:
                # Append the current episode to the list
                episodes.append(current_episode)
            else:
                # If the current episode is too short, discard it
                print("Discarding short episode:", len(current_episode))
            current_episode = [frame_numbers[i]]
    if len(current_episode) > 10:
        episodes.append(current_episode)
    return episodes
episodes = split_episodes(path)

print(len(episodes))
print(episodes[-1][-1])
# recordings path
recordings_path='/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/vitpose_output/episodes/no_episode_split'
output_path = '/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/vitpose_output'
j = 0
for i, episode in enumerate(tqdm(episodes, desc="Processing episodes")):
    # create a new folder for each episode
    folder_name = os.path.join(output_path, "episodes", f"e{j}")
    os.makedirs(folder_name, exist_ok=True)
    # copy the files to the new folder
    for frame_number in tqdm(episode, desc=f"Copying frames for episode {j}", leave=False):
        # get all files that include the frame number
        frame_number = str(frame_number).zfill(5)
        files = glob.glob(os.path.join(recordings_path, f"*{frame_number}*"))
        for file in files:
            shutil.move(file, folder_name)
    j += 1