
import glob
from pathlib import Path
import cv2
import numpy as np
episodes = [
    f"/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e{i}/handover_collection_0_e{i}.npz"
    for i in range(0,112)
]
raw_episodes = [ f"/mnt/dataset_drive/ayad/phantom-touch/data/recordings/handover_collection_0/episodes/e{i}"
                    for i in range(0,112) ]

i=1

def load_raw_frames(folder):
    """Return a list of cv2-loaded images from a raw episode folder."""
    # accept png/jpg; sort numerically if filenames contain indices
    pattern = str(Path(folder) / "*.png")
    files = sorted(glob.glob(pattern))
    if not files:
        pattern = str(Path(folder) / "*.jpg")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    frames = [cv2.imread(f) for f in files]
    if any(f is None for f in frames):
        raise IOError(f"Corrupt image detected in {folder}")
    return frames
# for episode in episodes:
#     try:
#         data = np.load(episode)
#     except FileNotFoundError:
#         print(f"File not found: {episode}")
#         i+=1
#         continue
#     # print(f"Loaded {episode} with {len(data['image_0'])} frames")
#     # print episodes with less than 10 frames
#     # print all keys in the data

#     if len(data["image_0"]) < 10:
#         print(f"Episode {episode} has less than 10 frames")
#         i+=1
# # print(i)
#     # replay the episode from image_0
#     for i,image in enumerate(data["inpainted"]):
#         cv2.imshow("image", image)
#         breakpoint()
#         if i == 0:
#             cv2.waitKey(0)
#         else:
#             cv2.waitKey(1)
i = 0
for raw_episode in raw_episodes:
    try:
        frames = load_raw_frames(raw_episode)
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading frames from {raw_episode}: {e}")
        continue
    # print(f"Loaded {raw_episode} with {len(frames)} frames")
    if len(frames) < 10:
        print(f"Raw episode {raw_episode} has less than 10 frames")
        continue
    # replay the raw episode
    for j, frame in enumerate(frames):
        # resize frame to 240 height and 432 width
        frame = cv2.resize(frame, (432, 240))
        cv2.imshow("frame", frame)
        if i == 0:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)
    i += 1