
import cv2
import numpy as np
episodes = [
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_temp/dataset/e0/handover_collection_temp_e0.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e1/handover_collection_1_e1.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e2/handover_collection_1_e2.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e3/handover_collection_1_e3.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e4/handover_collection_1_e4.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e5/handover_collection_1_e5.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e6/handover_collection_1_e6.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e7/handover_collection_1_e7.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e8/handover_collection_1_e8.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e9/handover_collection_1_e9.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e10/handover_collection_1_e10.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e11/handover_collection_1_e11.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e12/handover_collection_1_e12.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e13/handover_collection_1_e13.npz",
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_1/dataset/e14/handover_collection_1_e14.npz",
]
# for episode in episodes:
#     data = np.load(episode)
#     print(f"Loaded {episode} with {len(data['image_0'])} frames")
#     # replay the episode from image_0
#     for image in data["image_0"]:
#         cv2.imshow("image", image)
#         cv2.waitKey(100)

for episode in episodes:
    data = np.load(episode)
    for key in data.keys():
        print(f"{episode} {key}: {data[key].shape}")