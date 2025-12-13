import sys
import sieve
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import torch
import matplotlib
import datetime
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2_video_predictor
from utils.rgb_utils import load_rgb_images
from utils.sam2utils import (
    extract_centroid,
    save_mp4video,
    filelist_to_mp4sieve,
    search_folder,
    sievesamzip_to_numpy,
)

#### Script metadata ####
matplotlib.use("TKAgg")
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

#### Config loading ####
OmegaConf.register_new_resolver("phantom-touch", lambda: search_folder("/home", "phantom-touch"))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cfg = OmegaConf.load(f"{parent_dir}/conf/sam2_object.yaml")
sam2video_cfg = cfg.sam2videoPredictor

os.makedirs(sam2video_cfg.output_dir, exist_ok=True)

# Use the flat directory (no episodes)
images_path = sam2video_cfg.video_frames_dir  # directly set in config

images = load_rgb_images(base_dir=images_path, return_path=False, episodes=False, BGR=False)
# Show the first image and get a click
fig, ax = plt.subplots()
ax.imshow(images[0])  # Show first image
plt.title("Click to select centroid")
centroid_coords = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        centroid_coords.append([event.xdata, event.ydata])
        print(f"Selected centroid: ({event.xdata:.1f}, {event.ydata:.1f})")
        plt.close()  # Close the figure after one click

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Convert to numpy array if a point was selected
if centroid_coords:
    centroids = np.array(centroid_coords)
else:
    raise ValueError("No centroid was selected.")
# centroids = np.array([[300, 480-130]])

#### Second workflow component: VIDEO-SAM2 ####
print("Running VIDEO-SAM2...")
points = np.array([[centroids[0][0], centroids[0][1]]], dtype=np.float32)
predictor = build_sam2_video_predictor(sam2video_cfg.model_cfg, sam2video_cfg.sam2_checkpoint, device=device)

inference_state = predictor.init_state(video_path=images_path)
predictor.reset_state(inference_state)

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=0,
    obj_id=1,
    points=points,
    labels=np.array([1], np.int32)
)

mask_frames = []
for _, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    for i, _ in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
        mask = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2RGB)
        mask_frames.append(mask)
mask_frames = np.array(mask_frames)

#### Store results ####
print("Saving masks...")
frame_names = sorted(
    [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".png")],
    key=lambda p: os.path.splitext(os.path.basename(p))[0]
)
frame_names = frame_names  # Adjust to match the processed frames
print(f"Number of frames: {len(frame_names)}")

output_dir = sam2video_cfg.output_dir

os.makedirs(output_dir, exist_ok=True)
for i, frame_path in enumerate(frame_names):
    frame_name = os.path.join(output_dir, f"mask_{os.path.basename(frame_path)}")
    print(f"Saving {frame_name}")
    cv2.imwrite(frame_name, mask_frames[i])
