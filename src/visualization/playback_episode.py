
import cv2
import numpy as np
from omegaconf import OmegaConf
import os

from utils.samutils import search_folder
repo_dir = search_folder(f"/home/{os.getenv('USER')}/", "phantom-touch")
def overlay_images(mujoco_image, image_to_overlay):
    # Convert images to float32 and normalize to [0, 1]
    mujoco_image = mujoco_image.astype(np.float32) / 255.0
    image_to_overlay = image_to_overlay.astype(np.float32) / 255.0

    # Create alpha channel for mujoco image based on white background
    white_background = np.ones_like(mujoco_image)
    diff = np.abs(mujoco_image - white_background)
    alpha_channel = 1.0 - np.clip(np.max(diff, axis=2, keepdims=True) * 3, 0, 1)

    # Add alpha channel to mujoco image
    mujoco_image_with_alpha = np.concatenate((mujoco_image, alpha_channel), axis=2)

    # Add full opacity alpha channel to inpainted image
    color_image_with_alpha = np.concatenate(
        (image_to_overlay, np.ones((*image_to_overlay.shape[:2], 1), dtype=np.float32)), axis=2
    )
    # Perform the overlay operation
    overlayed_image = (
        color_image_with_alpha[:, :, :3] * (1 - mujoco_image_with_alpha[:, :, 3:])
        + mujoco_image_with_alpha[:, :, :3] * mujoco_image_with_alpha[:, :, 3:]
    )
    # Convert back to 8-bit for display
    overlayed_image = (overlayed_image * 255).astype(np.uint8)
    return overlayed_image

if __name__ == "__main__":
    visualization_cfg = OmegaConf.load(f"{repo_dir}/src/visualization/cfg/visualization.yaml")
    paths_cfg = OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
    episodes = [
        f"{paths_cfg.dataset_output_directory}/e{i}/{paths_cfg.metadata.experiment_name}_e{i}.npz"
        for i in range(visualization_cfg.start_episode, visualization_cfg.end_episode + 1)
    ]
    
    for j,episode in enumerate(episodes):
        try:
            data = np.load(episode)
        except FileNotFoundError:
            print(f"File not found: {episode}")
            continue
        if len(data["image_0"]) < 2:
            print(f"Episode {episode} {j} has less than 2 frames, length: {len(data['image_0'])}, skipping...")
            continue
        else:
            print(f"Loaded episode {episode} {j} with {len(data['image_0'])} frames.")
        # replay the episode from image_0
        for i,data_image in enumerate(data["image_0"]):
            # overlay data images on originals
            overlayed = overlay_images(data_image,data["original"][i])

            cv2.imshow(f"data image_e{j}", data_image)
            cv2.imshow("overlayed_data_image_on_original", overlayed)
            cv2.imshow("original",data["original"][i])
            # cv2.imshow(f"original_e{j}", data["original"][i])
            if i == len(data["image_0"])//2:
                cv2.imwrite("data_image_and_original.png", np.vstack((data_image, data["original"][i])))
                cv2.imwrite("overlayed_data_image_on_original.png", overlayed)
            cv2.waitKey(10)