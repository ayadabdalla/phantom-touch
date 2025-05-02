import numpy as np
from scipy import linalg


def calculate_target_position_and_orientation(hand_keypoints):
    """
    Calculate the target position as the midpoint between the thumb tip and index finger tip,
    and the target orientation by fitting a plane through the points of the thumb and index finger.

    Parameters:
    -----------
    hand_keypoints : numpy.ndarray or list
        Array or list of shape (21, 3) containing 3D coordinates of 21 hand keypoints.
        Keypoints are organized in the following order:
        0: Wrist
        1-4: Thumb (from base to tip)
        5-8: Index finger (from base to tip)
        9-12: Middle finger (from base to tip)
        13-16: Ring finger (from base to tip)
        17-20: Pinky finger (from base to tip)

    Returns:
    --------
    tuple: (target_position, normal_vector, plane_points, thumb_axis)
        target_position: numpy.ndarray of shape (3,) - midpoint between the thumb tip and index finger tip.
        normal_vector: numpy.ndarray of shape (3,) - unit normal vector of the fitted plane.
        plane_points: numpy.ndarray of shape (8, 3) - points used for plane fitting (thumb and index finger points).
        thumb_axis: numpy.ndarray of shape (3,) - unit vector representing the principal axis of the thumb.
    """
    # Validate input
    if not isinstance(hand_keypoints, np.ndarray):
        hand_keypoints = np.array(hand_keypoints)

    if hand_keypoints.shape != (21, 3):
        raise ValueError(
            f"Expected hand keypoints of shape (21, 3), got {hand_keypoints.shape}"
        )

    # Extract index finger and thumb keypoints
    index_finger = hand_keypoints[5:9]  # indices 5-8
    thumb = hand_keypoints[1:5]  # indices 1-4

    # 1. Calculate target position (pt) - midpoint between thumb tip and index tip
    thumb_tip = thumb[3]  # The 4th point of thumb (index 4 in original array)
    index_tip = index_finger[
        3
    ]  # The 4th point of index finger (index 8 in original array)
    target_position = (thumb_tip + index_tip) / 2

    # 2. Calculate target orientation by fitting a plane through thumb and index finger points
    # Combine all points from thumb and index finger
    plane_points = np.vstack((thumb, index_finger))

    # Fit a plane to the points using Singular Value Decomposition (SVD)
    centroid = np.mean(plane_points, axis=0)
    _, _, vh = np.linalg.svd(plane_points - centroid)
        # Center the points
    centered_points = plane_points - centroid

    # Singular Value Decomposition
    u, s, vh = linalg.svd(centered_points)

    # The normal vector to the plane is the last singular vector
    normal_vector = vh[2, :]

    # Normalize the vector to unit length
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # ensure the normal vector points in the negative z direction
    if normal_vector[2] > 0:
        normal_vector = -normal_vector
    # the principal axis is the thumb, fit a line through the first 4 points using pca
    thumb_axis = thumb[3] - thumb[0]
    thumb_axis = thumb_axis / np.linalg.norm(thumb_axis)

    return target_position, normal_vector, plane_points, thumb_axis
