
import cv2
import numpy as np
episodes = [
    "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e56/handover_collection_0_e56.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e1/handover_collection_0_e1.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e2/handover_collection_0_e2.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e3/handover_collection_0_e3.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e4/handover_collection_0_e4.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e5/handover_collection_0_e5.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e6/handover_collection_0_e6.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e7/handover_collection_0_e7.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e8/handover_collection_0_e8.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e9/handover_collection_0_e9.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e10/handover_collection_0_e10.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e11/handover_collection_0_e11.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e12/handover_collection_0_e12.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e13/handover_collection_0_e13.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e14/handover_collection_0_e14.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e15/handover_collection_0_e15.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e16/handover_collection_0_e16.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e17/handover_collection_0_e17.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e18/handover_collection_0_e18.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e19/handover_collection_0_e19.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e20/handover_collection_0_e20.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e21/handover_collection_0_e21.npz",
    # "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_collection_0/dataset/e38/handover_collection_0_e38.npz",
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