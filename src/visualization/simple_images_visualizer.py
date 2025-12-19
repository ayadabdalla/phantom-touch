import cv2
import numpy as np
from omegaconf import OmegaConf
import os

from utils.sam2utils import search_folder

def load_simple_images(image_paths):
    """Load a list of images from given file paths.

    Args:
        image_paths (list of str): List of file paths to the images.
    Returns:
        list of np.ndarray: List of loaded images.
    """
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            images.append(image)
        else:
            print(f"Warning: Could not load image at {path}")
    return images
def visualize_simple_images(image_list, window_name="Image Visualization", wait_time=0):
    """Visualize a list of images in a single window.

    Args:
        image_list (list of np.ndarray): List of images to visualize.
        window_name (str): Name of the display window.
        wait_time (int): Time in milliseconds to wait for a key event. 0 means wait indefinitely.
    """
    if not image_list:
        print("No images to display.")
        return

    # Concatenate images horizontally
    for i in range(len(image_list)):
        if image_list[i].ndim == 2:  # Grayscale image
            image_list[i] = cv2.cvtColor(image_list[i], cv2.COLOR_GRAY2BGR)
        image = image_list[i]
        # Display the image
        cv2.imshow(window_name, image)
        cv2.waitKey(wait_time)

if __name__ == "__main__":
    repo_dir= search_folder(f"/home/{os.getenv('USER')}/", "phantom-touch")
    visualization_cfg = OmegaConf.load(f"{repo_dir}/src/visualization/cfg/visualization.yaml")
    paths_cfg = OmegaConf.load(f"{repo_dir}/cfg/paths.yaml")
    image_folder = f"{paths_cfg.recordings_directory}/e6"
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    # sort image files to natural order
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))))
    images = load_simple_images(image_files)
    visualize_simple_images(images, wait_time=100)  # Display for 5 seconds