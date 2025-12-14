
import cv2
import numpy as np
from omegaconf import OmegaConf
import os

from utils.sam2utils import search_folder
repo_dir = search_folder(f"/home/{os.getenv('USER')}/", "phantom-touch")
def overlay_images(mujoco_image, inpainted_image):
    # Convert images to float32 and normalize to [0, 1]
    mujoco_image = mujoco_image.astype(np.float32) / 255.0
    inpainted_image = inpainted_image.astype(np.float32) / 255.0

    # Create alpha channel for mujoco image based on white background
    white_background = np.ones_like(mujoco_image)
    diff = np.abs(mujoco_image - white_background)
    alpha_channel = 1.0 - np.clip(np.max(diff, axis=2, keepdims=True) * 5.0, 0, 1)

    # Add alpha channel to mujoco image
    mujoco_image_with_alpha = np.concatenate((mujoco_image, alpha_channel), axis=2)

    # Add full opacity alpha channel to inpainted image
    color_image_with_alpha = np.concatenate(
        (inpainted_image, np.ones((*inpainted_image.shape[:2], 1), dtype=np.float32)), axis=2
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
    cfg = OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
    episodes = [
        f"{cfg.dataset_output_directory}/e{i}/{cfg.metadata.experiment_name}_e{i}.npz"
        for i in range(visualization_cfg.start_episode, visualization_cfg.end_episode + 1)
    ]
    
    for episode in episodes:
        try:
            data = np.load(episode)
        except FileNotFoundError:
            print(f"File not found: {episode}")
            continue
        # print(f"Loaded {episode} with {len(data['image_0'])} frames")
        # print episodes with less than 10 frames
        if len(data["image_0"]) < 10:
            print(f"Episode {episode} has less than 10 frames")
    # print(i)
        # replay the episode from image_0
        for i,image in enumerate(data["image_0"]):
            cv2.imshow("image", image)
            # cv2.imshow("original", data["original"][i])
            # save the image and the corresponding inpainted original vertically over each other
            # cv2.imshow("inpainted", np.vstack((image, data["original"][i])))
            # save the image
            # overlay both images
            overlayed = overlay_images(image,data["original"][i])
            cv2.imshow("overlayed", overlayed)
            if i == len(data["image_0"])//2:
                cv2.imwrite("inpainted.png", np.vstack((image, data["original"][i])))
                cv2.imwrite("original.png", data["original"][i])
                cv2.imwrite("datapoint.png", image)
                # cv2.imwrite("overlayed.png", overlayed)
            cv2.waitKey(1)

            
    print(i)