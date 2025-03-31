import sys
import cv2
import os

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
        print(f"Frame {frame_count}: {frame.shape}, dtype: {frame_dtype}")
        sys.exit()
        cv2.imwrite(frame_filename, frame )
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {output_folder}")

if __name__ == "__main__":
    video_path = "/home/abdullah/utn/phantom-human-videos/assets/d405-depth_J9ifnWP9.avi"
    output_folder = "extracted_frames"
    extract_frames(video_path, output_folder)