import numpy as np
import os
import glob

from utils.rgb_utils import natural_key
from tqdm import tqdm
def load_raw_depth_images(base_dir, shape):
    # depth_paths = glob.glob(os.path.join(base_dir, "e*", "RawDepth_*.raw"))
    depth_paths = []
    for root, dirs, files in os.walk(os.path.join(base_dir)):
        if os.path.basename(root).startswith("e"):
            for file in files:
                if file.startswith("RawDepth_") and file.endswith(".raw"):
                    depth_paths.append(os.path.join(root, file))
    depth_paths = sorted(depth_paths, key=natural_key)  # <--- natural sort
    depth_images = []
    for path in tqdm(depth_paths, desc="reading depth images"):
        with open(path, "rb") as f:
            raw = np.fromfile(f, dtype=np.uint16)
            raw = raw[:shape[0]*shape[1]]
        depth_image = raw.reshape(shape)
        depth_images.append(depth_image)
    return np.stack(depth_images, axis=0)
