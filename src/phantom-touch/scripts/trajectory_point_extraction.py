import numpy as np
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
    
    # Ensure the normal vector points "outward" from the palm
    # A simple heuristic: make sure it has a positive Z component (assuming Z is forward)
    if normal_vector[2] < 0:
        normal_vector = -normal_vector
    
    # Normalize the vector to unit length
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    return target_position, normal_vector, plane_points

# Example usage
if __name__ == "__main__":
    # Generate dummy 3D hand keypoints for demonstration
    # In a real application, you would get these from HAMER or other hand tracking model
    np.random.seed(42)  # For reproducible results
    dummy_keypoints = np.random.randn(21, 3)  # 21 keypoints with x, y, z coordinates
    
    # Calculate target position and orientation
    target_position, normal_vector, plane_points = calculate_target_position_and_orientation(dummy_keypoints)
    
    print(f"Target position (midpoint between tips):\n{target_position}")
    print(f"\nPlane normal vector (target orientation):\n{normal_vector}")
    
    # Visualize the results (commented out - uncomment if you have matplotlib)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all hand keypoints
    ax.scatter(dummy_keypoints[:, 0], dummy_keypoints[:, 1], dummy_keypoints[:, 2], 
               c='gray', alpha=0.5, label='All keypoints')
    
    # Plot thumb points
    ax.scatter(dummy_keypoints[1:5, 0], dummy_keypoints[1:5, 1], dummy_keypoints[1:5, 2], 
               c='blue', label='Thumb')
    
    # Plot index finger points
    ax.scatter(dummy_keypoints[5:9, 0], dummy_keypoints[5:9, 1], dummy_keypoints[5:9, 2], 
               c='green', label='Index finger')
    
    # Plot target position
    ax.scatter(target_position[0], target_position[1], target_position[2], 
               c='red', s=100, label='Target position')
    
    # Plot normal vector
    arrow_scale = 0.2
    ax.quiver(target_position[0], target_position[1], target_position[2],
              normal_vector[0] * arrow_scale, normal_vector[1] * arrow_scale, normal_vector[2] * arrow_scale,
              color='red', arrow_length_ratio=0.3, label='Normal vector')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Target Position and Orientation')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
