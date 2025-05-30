import re
import cv2
import os
import numpy as np
import glob

from tqdm import tqdm

def fetch_rgb_video(video_path):
    # Load the mask video
    cap_vid = cv2.VideoCapture(video_path)
    if not cap_vid.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    return cap_vid


def natural_key(string_):
    """Helper to sort strings like humans expect."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string_)]


def load_rgb_images(base_dir, prefix="Color_", return_path=False, episodes=True):
    # image_paths = glob.glob(os.path.join(base_dir, "e*", f"{prefix}*.png"))
    image_paths = []
    for root, dirs, files in os.walk(os.path.join(base_dir)):
        if os.path.basename(root).startswith("e") and episodes:
            for file in files:
                if file.startswith(prefix) and file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))
        elif not episodes:
            for file in files:
                if file.startswith(prefix) and file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))

    image_paths = sorted(image_paths, key=natural_key)  # <--- natural sort
    images = [cv2.imread(p) for p in tqdm(image_paths, desc="reading images")]
    if return_path:
        return np.stack(images, axis=0), image_paths
    else:
        return np.stack(images, axis=0)