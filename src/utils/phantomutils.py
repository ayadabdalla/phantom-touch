import cv2
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R


# def calculate_target_position_and_orientation(hand_keypoints):
#     """
#     Calculate the target position as the midpoint between the thumb tip and index finger tip,
#     and the target orientation by fitting a plane through the points of the thumb and index finger.

#     Parameters:
#     -----------
#     hand_keypoints : numpy.ndarray or list
#         Array or list of shape (21, 3) containing 3D coordinates of 21 hand keypoints.
#         Keypoints are organized in the following order:
#         0: Wrist
#         1-4: Thumb (from base to tip)
#         5-8: Index finger (from base to tip)
#         9-12: Middle finger (from base to tip)
#         13-16: Ring finger (from base to tip)
#         17-20: Pinky finger (from base to tip)

#     Returns:
#     --------
#     tuple: (target_position, normal_vector, plane_points, thumb_axis)
#         target_position: numpy.ndarray of shape (3,) - midpoint between the thumb tip and index finger tip.
#         normal_vector: numpy.ndarray of shape (3,) - unit normal vector of the fitted plane.
#         plane_points: numpy.ndarray of shape (8, 3) - points used for plane fitting (thumb and index finger points).
#         thumb_axis: numpy.ndarray of shape (3,) - unit vector representing the principal axis of the thumb.
#     """
#     # Validate input
#     if not isinstance(hand_keypoints, np.ndarray):
#         hand_keypoints = np.array(hand_keypoints)

#     if hand_keypoints.shape != (21, 3):
#         raise ValueError(
#             f"Expected hand keypoints of shape (21, 3), got {hand_keypoints.shape}"
#         )

#     # Extract index finger and thumb keypoints
#     index_finger = hand_keypoints[5:9]  # indices 5-8
#     thumb = hand_keypoints[1:5]  # indices 1-4

#     # 1. Calculate target position (pt) - midpoint between thumb tip and index tip
#     thumb_tip = thumb[3]  # The 4th point of thumb (index 4 in original array)
#     index_tip = index_finger[
#         3
#     ]  # The 4th point of index finger (index 8 in original array)
#     target_position = (thumb_tip + index_tip) / 2

#     # 2. Calculate target orientation by fitting a plane through thumb and index finger points
#     # Combine all points from thumb and index finger
#     plane_points = np.vstack((thumb, index_finger))

#     # Fit a plane to the points using Singular Value Decomposition (SVD)
#     centroid = np.mean(plane_points, axis=0)
#     _, _, vh = np.linalg.svd(plane_points - centroid)
#         # Center the points
#     centered_points = plane_points - centroid

#     # Singular Value Decomposition
#     u, s, vh = linalg.svd(centered_points)

#     # The normal vector to the plane is the last singular vector
#     normal_vector = vh[2, :]

#     # Normalize the vector to unit length
#     normal_vector = normal_vector / np.linalg.norm(normal_vector)

#     # ensure the normal vector points in the negative z direction
#     if normal_vector[2] > 0:
#         normal_vector = -normal_vector
#     # the principal axis is the thumb, fit a line through the first 4 points using pca
#     thumb_axis = thumb[3] - thumb[0]
#     thumb_axis = thumb_axis / np.linalg.norm(thumb_axis)

#     return target_position, normal_vector, plane_points, thumb_axis


def calculate_action(hand_keypoints,extrinsics):
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

    rot_matrix = normal_principal_to_rotation_matrix(
        normal_vector, thumb_axis
    )
    # convert to robot base
    target_position = np.dot(
        extrinsics[:3,:3], target_position
    ) + extrinsics[:3, 3]
    rot_matrix = np.dot(extrinsics[:3, :3],rot_matrix)

    r = R.from_matrix(rot_matrix)
    target_orn = r.as_euler(
        "xyz", degrees=False
    )
    grips = np.linalg.norm(thumb_tip - index_tip, axis=0)
    grips = grips / 0.1 # 10 cm is the max grip
    action = np.concatenate(
        [
            target_position.flatten(),
            target_orn.flatten(),
            np.array([grips]).flatten(),
        ]
    )

    return action

def overlay_image(color_image, mujoco_image, size=(432, 240)):

    # Normalize the color image and add an alpha channel
    color_image_normalized = color_image.astype(np.float32) / 255.0
    color_image_with_alpha = np.concatenate(
        [color_image_normalized, np.ones_like(color_image_normalized[:, :, :1])], axis=2
    )
    # Normalize the Mujoco image
    # # resize image to 240 * 432
    mujoco_image = cv2.resize(mujoco_image, size)
    mujoco_image_normalized = mujoco_image.astype(np.float32) / 255.0

    # Create alpha channel for Mujoco image - black pixels should be transparent
    # Assuming black is [0, 0, 0] in RGB
    mujoco_alpha = np.any(mujoco_image_normalized > 0, axis=2, keepdims=True).astype(
        np.float32
    )
    mujoco_image_with_alpha = np.concatenate(
        [mujoco_image_normalized, mujoco_alpha], axis=2
    )

    # Perform the overlay operation
    overlayed_image = (
        color_image_with_alpha[:, :, :3] * (1 - mujoco_image_with_alpha[:, :, 3:])
        + mujoco_image_with_alpha[:, :, :3] * mujoco_image_with_alpha[:, :, 3:]
    )

    # Convert back to 8-bit and BGR for display
    overlayed_image = (overlayed_image * 255).astype(np.uint8)
    return overlayed_image

def filter_trajectories(trajectories):
    # do exponential average on only the z axis
    filtered_positions = []
    filtered_normals = []
    filtered_thumbs = []
    filtered_grips = []
    for trajectory in trajectories:
        data = np.load(trajectory)
        positions = data["positions"]
        normals = data["normals"]
        thumbs = data["thumb_vectors"]
        keypoints = data["keypoints"]
        grips = np.linalg.norm(positions - keypoints[:, 8, :3], axis=1)
        # normalize the grips
        grips = grips / np.max(grips)
        filtered_z = np.zeros_like(positions[:, 2])
        # expect the next 2 steps and compare to the actual and if it's too far ignore it
        indeces = []
        for i in range(0, len(positions) - 2):
            filtered_z[i] = 0.8 * positions[i + 2, 2] + 0.2 * positions[i + 1, 2]
            # Check if the current z is too far from the expected z
            if abs(positions[i, 2] - filtered_z[i]) > 0.1:
                # store index of the filtered z
                indeces.append(i)
        positions = np.delete(positions, indeces, axis=0)
        normals = np.delete(normals, indeces, axis=0)
        thumbs = np.delete(thumbs, indeces, axis=0)
        grips = np.delete(grips, indeces, axis=0)
        print(f"Filtered out: {len(indeces)}")
        filtered_positions.append(positions)
        filtered_thumbs.append(thumbs)
        filtered_normals.append(normals)
        filtered_grips.append(grips)
    return (
        filtered_positions,
        filtered_normals,
        filtered_thumbs,
        indeces,
        filtered_grips,
    )


def normal_principal_to_rotation_matrix(normal, principal=None, eps=1e-6):
    # Normalize the input normal vector
    normal = -normal / np.linalg.norm(normal, 2)

    # Set the z-axis as the provided normal vector
    basis_three = principal

    # Calculate a perpendicular vector to the z-axis for the x-axis
    # basis_one = np.cross(np.array([0, 0, 1]), basis_three)
    basis_one = normal
    basis_one = basis_one / np.linalg.norm(basis_one, 2)

    # if basis one is zero vector use x-axis
    if np.linalg.norm(basis_one, 2) == 0:
        print("basis one is zero vector")
        basis_one = np.array([1, 0, 0])

    # Calculate the y-axis as the cross product of the z-axis and x-axis
    basis_two = np.cross(basis_three, basis_one)
    basis_two = basis_two / np.linalg.norm(basis_two, 2)

    # Construct the rotation matrix
    rot = np.vstack((basis_one, basis_two, basis_three)).T
    return rot
