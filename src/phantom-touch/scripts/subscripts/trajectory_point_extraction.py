import os
import numpy as np
from omegaconf import OmegaConf
from scipy import linalg

def calculate_target_position_and_orientation(hand_keypoints):
    """
    Calculate the target position as the midpoint between thumb and index fingertips,
    and the target orientation by fitting a plane through thumb and index finger points.
    
    Parameters:
    -----------
    hand_keypoints : numpy.ndarray
        Array of shape (21, 3) containing 3D coordinates of 21 hand keypoints
        Keypoints are organized in the following order:
        0: Wrist
        1-4: Thumb (from base to tip)
        5-8: Index finger (from base to tip)
        9-12: Middle finger (from base to tip)
        13-16: Ring finger (from base to tip)
        17-20: Pinky finger (from base to tip)
    
    Returns:
    --------
    tuple: (target_position, normal_vector, plane_points)
        target_position: numpy.ndarray of shape (3,) - midpoint between thumb and index tips
        normal_vector: numpy.ndarray of shape (3,) - unit normal vector of the fitted plane
        plane_points: numpy.ndarray of shape (8, 3) - points used for plane fitting
    """
    # Validate input
    if not isinstance(hand_keypoints, np.ndarray):
        hand_keypoints = np.array(hand_keypoints)
    
    if hand_keypoints.shape != (21, 3):
        raise ValueError(f"Expected hand keypoints of shape (21, 3), got {hand_keypoints.shape}")
    
    # Extract index finger and thumb keypoints
    index_finger = hand_keypoints[5:9]  # indices 5-8
    thumb = hand_keypoints[1:5]         # indices 1-4
    
    # 1. Calculate target position (pt) - midpoint between thumb tip and index tip
    thumb_tip = thumb[3]     # The 4th point of thumb (index 4 in original array)
    index_tip = index_finger[3]  # The 4th point of index finger (index 8 in original array)
    target_position = (thumb_tip + index_tip) / 2
    
    # 2. Calculate target orientation by fitting a plane through thumb and index finger points
    # Combine all points from thumb and index finger
    plane_points = np.vstack((thumb, index_finger))
    
    # Calculate centroid
    centroid = np.mean(plane_points, axis=0)
    
    # Center the points
    centered_points = plane_points - centroid
    
    # Singular Value Decomposition
    u, s, vh = linalg.svd(centered_points)
    
    # The normal vector to the plane is the last singular vector
    normal_vector = vh[2, :]
    
    
    # Normalize the vector to unit length
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    return target_position, normal_vector, plane_points

# Example usage
if __name__ == "__main__":
    hand_keypoints_pcd = np.load("hand_keypoints_pcd.npz")
    hand_keypoints = hand_keypoints_pcd['points']
    print(f"hand keypoints pcd length: {len(hand_keypoints)}")
    repository_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = OmegaConf.load(f"{repository_directory}/conf/3d_projection.yaml")
    output_directory = os.path.join(config.trajectory_directory, "target_position_orientation.npz")
    # Calculate target position and orientation
    target_position, normal_vector, plane_points = calculate_target_position_and_orientation(hand_keypoints)
    
    # save the target position and orientation to a file
    np.savez_compressed(output_directory, target_position=target_position, normal_vector=normal_vector)
    print(f"Target position (midpoint between tips):\n{target_position}")
    print(f"\nPlane normal vector (target orientation):\n{normal_vector}")    
