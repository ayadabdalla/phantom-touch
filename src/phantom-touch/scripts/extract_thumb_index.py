import numpy as np

def extract_index_and_thumb(hand_keypoints):
    """
    Extract the index finger and thumb keypoints from a set of 21 3D hand keypoints.
    
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
    tuple: (index_finger, thumb)
        index_finger: numpy.ndarray of shape (4, 3) containing index finger keypoints
        thumb: numpy.ndarray of shape (4, 3) containing thumb keypoints
    """
    # Validate input
    if not isinstance(hand_keypoints, np.ndarray):
        hand_keypoints = np.array(hand_keypoints)
    
    if hand_keypoints.shape != (21, 3):
        raise ValueError(f"Expected hand keypoints of shape (21, 3), got {hand_keypoints.shape}")
    
    # Extract index finger keypoints (indices 5-8)
    index_finger = hand_keypoints[5:9]
    
    # Extract thumb keypoints (indices 1-4)
    thumb = hand_keypoints[1:5]
    
    return index_finger, thumb

# Example usage
if __name__ == "__main__":
    # Generate dummy 3D hand keypoints for demonstration
    # In a real application, you would get these from HAMER or other hand tracking model
    dummy_keypoints = np.random.randn(21, 3)  # 21 keypoints with x, y, z coordinates
    
    # Extract index finger and thumb
    index_finger, thumb = extract_index_and_thumb(dummy_keypoints)
    
    print(f"Index finger keypoints shape: {index_finger.shape}")
    print(f"Thumb keypoints shape: {thumb.shape}")
    
    print("\nIndex finger keypoints:")
    for i, point in enumerate(index_finger):
        joint_name = ["MCP", "PIP", "DIP", "TIP"][i]
        print(f"  {joint_name}: {point}")
    
    print("\nThumb keypoints:")
    for i, point in enumerate(thumb):
        joint_name = ["CMC", "MCP", "IP", "TIP"][i]
        print(f"  {joint_name}: {point}")