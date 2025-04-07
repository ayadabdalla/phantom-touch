import sys
import cv2
import os
from utils.sam2utils import search_folder

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        # get frame data type to kjnow if it is 8-bit or 16-bit
        frame_dtype = frame.dtype
        cv2.imwrite(frame_filename, frame )
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    # get the current repo directory
    current_repo_dir = search_folder("/home", "phantom-touch")
    video_path = f"{current_repo_dir}/data/output/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/sam2-sieve_output/color_video_compiled_for_sieve_2025-04-07_11-06-04.mp4"
    output_folder = f"{current_repo_dir}/data/recordings/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/jpg_frames"
    extract_frames(video_path, output_folder)