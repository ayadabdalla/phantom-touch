import cv2
import os
import numpy as np
def fetch_rgb_video(video_path):
    # Load the mask video
    cap_vid = cv2.VideoCapture(video_path)
    if not cap_vid.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    return cap_vid

def load_rgb_images(rgb_directory_path, prefix=None):
    rgb_images = []
    for filename in sorted(os.listdir(rgb_directory_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(rgb_directory_path, filename)
            img = cv2.imread(img_path)
            rgb_images.append(img)
    return np.array(rgb_images)