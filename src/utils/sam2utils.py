import cv2
import zipfile
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
import sieve

def sievesamzip_to_mp4_video(sam_out, output_path=None):
    """
    convert zip file of frames to an mp4
    """
    # convert zip file of frames to an mp4
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(sam_out["masks"].path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        images = [img for img in os.listdir(temp_dir) if img.endswith(".png")]
        images = sorted(images, key=lambda x: int(x.split("_")[1]))

        first_frame = cv2.imread(os.path.join(temp_dir, images[0]))
        height, width, layers = first_frame.shape
        frame_size = (width, height)

        # Define the codec and create VideoWriter object
        out = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, frame_size
        )

        # Loop through the images and write them to the video
        for image in images:
            img_path = os.path.join(temp_dir, image)
            frame = cv2.imread(img_path)
            out.write(frame)
    out.release()


def sievesamzip_to_numpy(sam_out):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(sam_out["masks"].path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        images = [img for img in os.listdir(temp_dir) if img.endswith(".png")]
        images = sorted(images, key=lambda x: int(x.split("_")[1]))
        original_masks = []
        for image in images:
            mask = cv2.imread(os.path.join(temp_dir, image), cv2.IMREAD_ANYDEPTH)
            original_masks.append(mask)
    original_masks = np.array(original_masks)
    return original_masks


def filelist_to_mp4sieve(frames_dir, output_path=None):
    # convert list of images to mp4
    images = [
        img
        for img in os.listdir(frames_dir)
        if img.endswith(".png") and img.startswith("png_output_Color")
    ]
    images = sorted(images)
    print(f"Number of loaded images: {len(images)}")
    first_frame = cv2.imread(os.path.join(frames_dir, images[0]))
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, frame_size)

    # Loop through the images and write them to the video
    for image in images:
        img_path = os.path.join(frames_dir, image)
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")
    return sieve.File(path=output_path)


def save_mp4video(masks, output_path=None):
    """
    save refined masks in a video
    """
    # save final masks in a video
    height, width, layers = masks[0].shape
    frame_size = (width, height)
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, frame_size)
    # Loop through the images and write them to the video
    for mask in masks:
        frame = mask
        out.write(frame)
    out.release()


def extract_centroid(mask):
    """
    Extract the centroid of the mask.
    """
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate the centroid
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

def search_folder(start_path, target_folder):
    for dirpath, dirnames, filenames in os.walk(start_path):
        if target_folder in dirnames:
            return os.path.join(dirpath, target_folder)
    return None