import os

import cv2
import numpy as np
import open3d as o3d
from utils.rgb_utils import natural_key


import os
import numpy as np
import re
from collections import defaultdict

def natural_key(string_):
    """Sort helper that handles numbers in filenames correctly."""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def load_keypoints_grouped_by_frame(base_dir, prefix="vitpose_", return_path=False):
    frame_to_paths = defaultdict(list)

    # Walk through directories
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root).startswith("e"):
            for file in files:
                if file.startswith(prefix) and file.endswith(".npy"):
                    full_path = os.path.join(root, file)
                    # Extract frame id: the part after 'eXXXXX_' and before '_right'
                    match = re.search(r'\d+_(\d+)_right', file)
                    if match:
                        frame_id = match.group(1)
                        frame_to_paths[frame_id].append(full_path)

    # Sort frames naturally
    sorted_frame_ids = sorted(frame_to_paths.keys(), key=natural_key)

    keypoints_grouped = []
    paths_grouped = []

    for frame_id in sorted_frame_ids:
        # Sort the paths for each frame naturally (e.g., right_0, right_1, etc.)
        paths = sorted(frame_to_paths[frame_id], key=natural_key)
        keypoints = [np.load(p) for p in paths]
        keypoints_grouped.append(np.stack(keypoints, axis=0))  # Shape: (num_views, ...)
        paths_grouped.append(paths)

    if return_path:
        return keypoints_grouped, paths_grouped
    else:
        return keypoints_grouped

def load_pcds(base_dir, prefix="Color_", return_path=False):
    """
    Load PCD files from a directory, grouped by frame.
    Args:
        base_dir (str): Base directory containing the PCD files.
        prefix (str): Prefix for the PCD files to load.
        return_path (bool): If True, return the paths of the loaded files.
    Returns:
        list: List of loaded PCD files, each as a numpy array.
        list: List of paths to the loaded PCD files (if return_path is True).
    """ 
    pcds_paths=[]
    # Walk through directories
    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root).startswith("e"):
            for file in files:
                if file.startswith(prefix) and file.endswith(".ply"):
                    full_path = os.path.join(root, file)
                    pcds_paths.append(full_path)
    # Sort frames naturally
    sorted_pcd_paths = sorted(pcds_paths, key=natural_key)
    # load the pcds through o3d
    pcds = []
    for pcd_path in sorted_pcd_paths:
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcds.append(pcd)
    pcds = np.stack(pcds, axis=0)  # Shape: (num_views, ...)
    pcds = np.reshape(pcds, (pcds.shape[0], 1))
    if return_path:
        return pcds, pcds_paths
    else:
        return pcds

def retrieve_data_sample_path(
    data_source,
    experiment_name,
    experiment_specifics,
    data_ontology,
    sample_id,
    sub_data_sample_id=None,
    data_sample_name=None,
    data_extension=None,
):
    """
    Function to retrieve the data sample from the specified path.
    """
    # Construct the path to the data sample
    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_sample_name = (
        f"{data_sample_name}_{sample_id}_{sub_data_sample_id}"
        if sub_data_sample_id
        else f"{data_sample_name}_{sample_id}"
    )
    data_extension = "obj"

    path = os.path.join(
        repository_root,
        "assets",
        "data",
        data_source,
        experiment_name,
        experiment_specifics,
        data_ontology,
        f"{data_sample_name}.{data_extension}",
    )

    return path