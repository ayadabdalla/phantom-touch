
import cv2
import numpy as np
episodes = [
    f"/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e{i}/handover_collection_0_e{i}.npz"
    for i in range(0,112)
]
i=1
for episode in episodes:
    try:
        data = np.load(episode)
    except FileNotFoundError:
        print(f"File not found: {episode}")
        i+=1
        continue
    # print(f"Loaded {episode} with {len(data['image_0'])} frames")
    # print episodes with less than 10 frames
    if len(data["image_0"]) < 10:
        print(f"Episode {episode} has less than 10 frames")
        i+=1
print(i)
    # # replay the episode from image_0
    # for image in data["image_0"]:
    #     cv2.imshow("image", image)
    #     cv2.waitKey(1)