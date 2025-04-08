import cv2


def fetch_rgb_video(video_path):
    # Load the mask video
    cap_vid = cv2.VideoCapture(video_path)
    if not cap_vid.isOpened():
        raise Exception(f"Could not open video file: {video_path}")
    return cap_vid