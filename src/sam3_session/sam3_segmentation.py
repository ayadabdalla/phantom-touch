import cv2
import torch

from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "/mnt/dataset_drive/ayad/data/recordings/soft_strawberries/Strawberry_1/extracted_recording_1760633035/orbbec_color_images" # a JPEG folder or an MP4 video file
# Start a session
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=10, # Arbitrary frame index
        text="strawberry",
    )
)
output = response["outputs"]
import numpy as np

cv2.imwrite("strawberry_mask.png",output['out_binary_masks'][0].astype(np.uint8)*255)