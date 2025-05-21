import cv2
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R

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
    thumb_axis = thumb[3] - thumb[0]
    thumb_axis = thumb_axis / np.linalg.norm(thumb_axis)

    # # convert to robot base frame
    normal_vector = np.concatenate([normal_vector, np.zeros((1))], axis=0)
    thumb_axis = np.concatenate([thumb_axis, np.zeros((1))], axis=0)
    thumb_axis = np.dot(extrinsics, thumb_axis)
    normal_vector = np.dot(extrinsics, normal_vector)
    thumb_axis = thumb_axis[:3]
    normal_vector = normal_vector[:3]
    target_position = np.concatenate([target_position, np.ones((1))], axis=0)
    target_position = np.dot(
        extrinsics, target_position
    )


    target_position = target_position[:3]
    rot_matrix = normal_principal_to_rotation_matrix(
        normal_vector, thumb_axis
    )
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

def filter_episode(npz_file):
    # do exponential average on only the z axis
    action=np.array(npz_file["action"]).reshape(-1, 7)
    image_0 = np.array(npz_file["image_0"]).reshape(-1, 240, 432, 3)
    state = np.array(npz_file["state"]).reshape(-1, 14)
    keypoints = np.array(npz_file["keypoints"]).reshape(-1, 21, 3)
    inpainted = np.array(npz_file["inpainted"]).reshape(-1, 240, 432, 3)
    filtered_xyz = np.zeros_like(action[:, :3])
    filtered_euler = np.zeros_like(action[:, 3:6])
    indeces = set()
    for i in range(0, len(action) - 2):
        # Smooth position (x, y, z)
        filtered_xyz[i] = 0.5 * action[i + 2, :3] + 0.5 * action[i + 1, :3]

        # Smooth Euler angles with wrap-around consideration
        raw_euler = np.stack([action[i + 1, 3:6], action[i + 2, 3:6]], axis=0)
        filtered_euler[i] = np.arctan2(np.sin(raw_euler).mean(axis=0), np.cos(raw_euler).mean(axis=0))

        # Check for position anomalies
        pos_diffs = np.abs(action[i, :3] - filtered_xyz[i])
        
        # Check for angle anomalies using shortest angular distance
        angle_diffs = np.abs(np.arctan2(np.sin(action[i, 3:6] - filtered_euler[i]),
                                        np.cos(action[i, 3:6] - filtered_euler[i])))

        if np.any(pos_diffs > 0.2) or np.any(angle_diffs > 1):
            indeces.add(i)
    indeces = sorted(indeces)
    action = np.delete(action, indeces, axis=0)
    image_0 = np.delete(image_0, indeces, axis=0)
    state = np.delete(state, indeces, axis=0)
    keypoints = np.delete(keypoints, indeces, axis=0)
    inpainted = np.delete(inpainted, indeces, axis=0)

    print(f"Filtered out: {len(indeces)}")
    data ={
        "action": action,
        "image_0": image_0,
        "state": state,
        "keypoints": keypoints,
        "inpainted": inpainted
    }
    return (
        data
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


def smooth_quaternions(quaternions, window_size=5):
    pad = window_size // 2
    padded = np.pad(quaternions, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.zeros_like(quaternions)

    for i in range(len(quaternions)):
        window = padded[i:i+window_size]

        # Ensure continuity: flip quaternions to the same hemisphere
        for j in range(1, window.shape[0]):
            if np.dot(window[j], window[j - 1]) < 0:
                window[j] = -window[j]

        avg = np.mean(window, axis=0)
        avg /= np.linalg.norm(avg)
        smoothed[i] = avg

    return smoothed



def moving_average(data, window_size=5):
    pad_width = window_size // 2
    if data.ndim == 1:
        padded = np.pad(data, (pad_width,), mode="edge")
        kernel = np.ones(window_size) / window_size
        return np.convolve(padded, kernel, mode="valid")
    else:
        padded = np.pad(data, ((pad_width, pad_width), (0, 0)), mode="edge")
        kernel = np.ones(window_size) / window_size
        return np.array([
            np.convolve(padded[:, i], kernel, mode="valid")
            for i in range(data.shape[1])
        ]).T


def invert_keypoints(vitpose_keypoints2d, config):
    inverted_keypoints = []

    for keypoints in vitpose_keypoints2d:
        # Get cropping and resizing info
        x_crop, y_crop, w_crop, h_crop, w_resize, h_resize = get_crop_and_resize_params(config)

        # Extract x', y'
        xy_prime = keypoints[..., :2]  # shape (n_i, 21, 2)

        # Inverse resize
        xy_c = xy_prime.copy()
        xy_c[..., 0] = (xy_prime[..., 0] / w_resize) * w_crop  # x_c
        xy_c[..., 1] = (xy_prime[..., 1] / h_resize) * h_crop  # y_c

        # Inverse crop
        xy_orig = xy_c.copy()
        xy_orig[..., 0] += x_crop
        xy_orig[..., 1] += y_crop

        # Recombine with score
        scores = keypoints[..., 2:]  # shape (n_i, 21, 1)
        keypoints_orig = np.concatenate([xy_orig, scores], axis=-1)  # (n_i, 21, 3)

        inverted_keypoints.append(keypoints_orig)

    return inverted_keypoints

def get_crop_and_resize_params(config):
    x1_crop, y1_crop, x2_crop, y2_crop = config.crop.x1, config.crop.y1, config.crop.x2, config.crop.y2
    w_resize, h_resize = config.resolution.width, config.resolution.height
    x_crop = x1_crop
    y_crop = y1_crop
    w_crop = x2_crop - x1_crop
    h_crop = y2_crop - y1_crop
    return x_crop, y_crop, w_crop, h_crop, w_resize, h_resize