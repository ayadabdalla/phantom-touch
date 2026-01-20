# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import importlib
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation
import torch
import re
from omegaconf import OmegaConf

from core.utils import to_tensors
torch.backends.cudnn.deterministic = True
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.__version__)

# Load configuration
if len(sys.argv) < 2:
    print("Usage: python test.py <config.yaml>")
    sys.exit(1)

config_path = sys.argv[1]
if not os.path.exists(config_path):
    print(f"Config file not found: {config_path}")
    sys.exit(1)

cfg = OmegaConf.load(config_path)
print("Configuration loaded:")
print(OmegaConf.to_yaml(cfg))

# Extract config values
ref_length = cfg.model.step
num_ref = cfg.model.num_ref
neighbor_stride = cfg.model.neighbor_stride
default_fps = cfg.model.savefps

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
    images = [cv2.imread(p) for p in tqdm(image_paths, desc="reading images")]
    if return_path:
        return np.stack(images, axis=0), image_paths
    else:
        return np.stack(images, axis=0)

# sample reference frames from the whole video
def get_ref_index(f, neighbor_ids, length, num_ref, ref_length):
    ref_index = []
    if num_ref == -1:
        for i in range(0, length, ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, f - ref_length * (num_ref // 2))
        end_idx = min(length, f + ref_length * (num_ref // 2))
        for i in range(start_idx, end_idx + 1, ref_length):
            if i not in neighbor_ids:
                if len(ref_index) > num_ref:
                    break
                ref_index.append(i)
    return ref_index


# read frame-wise masks
def read_mask(mpath, size):
    masks = []
    mnames = mpath
    mnames.sort()
    for mp in mnames:
        m = Image.open(mp)
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                       iterations=4)
        masks.append(Image.fromarray(m * 255))
    return masks


#  read frames from video
def read_frame_from_videos(vname, use_mp4):
    frames = []
    if use_mp4:
        vidcap = cv2.VideoCapture(vname)
        success, image = vidcap.read()
        count = 0
        while success:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
            success, image = vidcap.read()
            count += 1
    else:
        lst = os.listdir(vname)
        lst.sort()
        fr_lst = [vname + '/' + name for name in lst]
        for fr in fr_lst:
            image = cv2.imread(fr)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frames.append(image)
    return frames


# resize frames
def resize_frames(frames, size=None):
    # convert to list if not list
    if not isinstance(frames, list):
        # convert numpy array to list
        frames = [Image.fromarray(f) for f in frames]
    if size is not None:
        frames = [f.resize(size) for f in frames]
    else:
        size = frames[0].size
    return frames, size


def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine size from config
    if cfg.model.name == "e2fgvi" and not cfg.model.set_size:
        size = (432, 240)
    elif cfg.model.set_size:
        size = (cfg.model.width, cfg.model.height)
    else:
        size = None

    net = importlib.import_module('model.' + cfg.model.name)
    model = net.InpaintGenerator().to(device)
    data = torch.load(cfg.model.checkpoint, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {cfg.model.checkpoint}')
    model.eval()

    # prepare dataset
    print(f'Processing episodes {cfg.data.episode_start} to {cfg.data.episode_end - 1}')
    print(f'RGB path: {cfg.data.rgb_base_path}')
    print(f'Mask path: {cfg.data.mask_base_path}')
    
    for n in range(cfg.data.episode_start, cfg.data.episode_end):
        rgb_path = os.path.join(cfg.data.rgb_base_path, f"e{n}")
        print(f"\nProcessing episode {n}: {rgb_path}")
        frames, paths = load_rgb_images(rgb_path, prefix=cfg.data.rgb_prefix, return_path=True)
        frames, size = resize_frames(frames, size)
        h, w = size[1], size[0]
        video_length = len(frames)
        imgs = to_tensors()(frames).unsqueeze(0) * 2 - 1
        frames = [np.array(f).astype(np.uint8) for f in frames]
        masks_path = os.path.join(cfg.data.mask_base_path, f"e{n}")
        print(f"Loading masks from: {masks_path}")
        masks, paths = load_rgb_images(masks_path, prefix=cfg.data.mask_prefix, return_path=True)
        masks = read_mask(paths, size)
        binary_masks = [
            np.expand_dims((np.array(m) != 0).astype(np.uint8), 2) for m in masks
        ]
        masks = to_tensors()(masks).unsqueeze(0)
        imgs, masks = imgs.to(device), masks.to(device)
        comp_frames = [None] * video_length

        # completing holes by e2fgvi
        print(f'Start inpainting for episode {n}...')
        for f in tqdm(range(0, video_length, neighbor_stride), desc=f"Episode {n}"):

            if cfg.model.verbose:
                print(f"Frame {f}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")
                print(f"Frame {f}: {torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")

            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, num_ref, ref_length)
            selected_imgs = imgs[:1, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks[:1, neighbor_ids + ref_ids, :, :, :]
            with torch.no_grad():
                masked_imgs = selected_imgs * (1 - selected_masks)
                mod_size_h = 60
                mod_size_w = 108
                h_pad = (mod_size_h - h % mod_size_h) % mod_size_h
                w_pad = (mod_size_w - w % mod_size_w) % mod_size_w
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [3])],
                    3)[:, :, :, :h + h_pad, :]
                masked_imgs = torch.cat(
                    [masked_imgs, torch.flip(masked_imgs, [4])],
                    4)[:, :, :, :, :w + w_pad]
                pred_imgs, _ = model(masked_imgs, len(neighbor_ids))
                pred_imgs = pred_imgs[:, :, :h, :w]
                pred_imgs = (pred_imgs + 1) / 2
                pred_imgs = pred_imgs.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_imgs[i]).astype(
                        np.uint8) * binary_masks[idx] + frames[idx] * (
                            1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(
                            np.float32) * 0.5 + img.astype(np.float32) * 0.5

        frame_names = []
        output_episode_path = os.path.join(cfg.data.output_path, f"e{n}")
        # make directory if not exist
        os.makedirs(output_episode_path, exist_ok=True)
        print(f"Output path: {output_episode_path}")
        
        for root, dirs, files in os.walk(rgb_path):
            if os.path.basename(root).startswith("e"):
                for file in files:
                    if file.startswith(cfg.data.rgb_prefix) and file.endswith(".png"):
                        frame_names.append(os.path.join(root, file))
        frame_names.sort(key=lambda p: os.path.splitext(p)[0])
        
        for i in range(len(comp_frames)):
            # Generate output filename maintaining original structure
            original_name = os.path.basename(frame_names[i])
            output_frame_path = os.path.join(output_episode_path, original_name)
            os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)
            if cfg.model.verbose:
                print(f"Saving {output_frame_path}")
            cv2.imwrite(output_frame_path, comp_frames[i])
        
        print(f"Episode {n} completed. Saved {len(comp_frames)} frames to {output_episode_path}")

    # saving videos
    # print('Saving videos...')
    # save_dir_name = 'results'
    # ext_name = '_results.mp4'
    # save_base_name = args.video.split('/')[-1]
    # save_name = save_base_name.replace(
    #     '.mp4', ext_name) if args.use_mp4 else save_base_name + ext_name
    # if not os.path.exists(save_dir_name):
    #     os.makedirs(save_dir_name)
    # save_path = os.path.join(save_dir_name, save_name)
    # writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
    #                          default_fps, size)
    # for f in range(video_length):
    #     comp = comp_frames[f].astype(np.uint8)
    #     writer.write(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    # writer.release()
    # print(f'Finish test! The result video is saved in: {save_path}.')

    # # show results
    # print('Let us enjoy the result!')
    # fig = plt.figure('Let us enjoy the result')
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.axis('off')
    # ax1.set_title('Original Video')
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.axis('off')
    # ax2.set_title('Our Result')
    # imdata1 = ax1.imshow(frames[0])
    # imdata2 = ax2.imshow(comp_frames[0].astype(np.uint8))

    # def update(idx):
    #     imdata1.set_data(frames[idx])
    #     imdata2.set_data(comp_frames[idx].astype(np.uint8))

    # fig.tight_layout()
    # anim = animation.FuncAnimation(fig,
    #                                update,
    #                                frames=len(frames),
    #                                interval=50)
    # plt.show()


if __name__ == '__main__':
    main_worker()
