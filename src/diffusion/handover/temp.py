import cv2
import numpy as np
episodes = [
    f"/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e{i}/handover_collection_0_e{i}.npz"
    for i in range(112)
]
for episode in episodes:
    try:
        data = np.load(episode)
    except FileNotFoundError:
        print(f"File not found: {episode}")
        continue
    print(f"Loaded {episode} with {len(data['image_0'])} frames")
    # replay the episode from image_0
    for image in data["image_0"]:
        cv2.imshow("image", image)
        cv2.waitKey(1)