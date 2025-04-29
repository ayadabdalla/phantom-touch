import re
import cv2
import os
import numpy as np
import glob

def fetch_rgb_video(video_path):
    # Load the mask video
    cap_vid = cv2.VideoCapture(video_path)
    if not cap_vid.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    return cap_vid

# def load_rgb_images(rgb_directory_path, prefix=None):
#     rgb_images = []
#     for filename in sorted(os.listdir(rgb_directory_path)):
#         if filename.endswith(".jpg") or filename.endswith(".png"):
#             img_path = os.path.join(rgb_directory_path, filename)
#             img = cv2.imread(img_path)
#             rgb_images.append(img)
#     return np.array(rgb_images)


def natural_key(string_):
    """Helper to sort strings like humans expect."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', string_)]


def load_rgb_images(base_dir, prefix="Color_", return_path=False):
    # image_paths = glob.glob(os.path.join(base_dir, "e*", f"{prefix}*.png"))
    image_paths = []
    for root, dirs, files in os.walk(os.path.join(base_dir)):
        if os.path.basename(root).startswith("e"):
            for file in files:
                if file.startswith(prefix) and file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))

    image_paths = sorted(image_paths, key=natural_key)  # <--- natural sort
    images = [cv2.imread(p) for p in image_paths]
    if return_path:
        return np.stack(images, axis=0), image_paths
    else:
        return np.stack(images, axis=0)