import numpy as np

def extract_index_and_thumb_batch(hand_keypoints_batch):
    """
    Extract index finger and thumb keypoints from a batch of 3D hand keypoints.

    Parameters:
    -----------
    hand_keypoints_batch : numpy.ndarray
        Array of shape (n, 21, 3), where n is the number of hand samples.
    
    Returns:
    --------
    tuple: (index_fingers, thumbs)
        index_fingers: numpy.ndarray of shape (n, 4, 3) containing index finger keypoints
        thumbs: numpy.ndarray of shape (n, 4, 3) containing thumb keypoints
    """
    # Validate input
    if not isinstance(hand_keypoints_batch, np.ndarray):
        hand_keypoints_batch = np.array(hand_keypoints_batch)
    
    if hand_keypoints_batch.ndim != 3 or hand_keypoints_batch.shape[1:] != (21, 3):
        raise ValueError(f"Expected shape (n, 21, 3), got {hand_keypoints_batch.shape}")
    
    # Extract index finger and thumb keypoints
    index_fingers = hand_keypoints_batch[:, 5:9, :]
    thumbs = hand_keypoints_batch[:, 1:5, :]
    
    return index_fingers, thumbs

# Example usage
if __name__ == "__main__":
    n_samples = 5  # Number of samples
    dummy_keypoints_batch = np.random.randn(n_samples, 21, 3)
    dummy_keypoints_batch = np.load(
        "/mnt/dataset_drive/ayad/phantom-touch/data/output/handover_experiment_1/sam2-vid_output/keypoints_3d.npy",
        allow_pickle=True,
    )
    index_fingers, thumbs = extract_index_and_thumb_batch(dummy_keypoints_batch)

    for i in range(n_samples):
        print(f"\nSample {i+1}")
        print("Index finger keypoints:")
        for j, point in enumerate(index_fingers[i]):
            joint_name = ["MCP", "PIP", "DIP", "TIP"][j]
            print(f"  {joint_name}: {point}")
        
        print("Thumb keypoints:")
        for j, point in enumerate(thumbs[i]):
            joint_name = ["CMC", "MCP", "IP", "TIP"][j]
            print(f"  {joint_name}: {point}")
