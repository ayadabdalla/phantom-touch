"""
SAM3-based video segmentation workflow
Replaces the SAM2-sieve + SAM2 video predictor workflow with SAM3's direct text-to-mask approach.
"""
import sys
import cv2
import os
import numpy as np
import torch
import datetime
from omegaconf import OmegaConf
from sam3.model_builder import build_sam3_video_predictor

#### script metadata ####
now = datetime.datetime.now()
now = now.strftime("%Y-%m-%d_%H-%M-%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

def search_folder(start_path, folder_name):
    """Search for a folder by name starting from start_path"""
    for root, dirs, files in os.walk(start_path):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

if __name__ == "__main__":
    #### initialize configurations ####
    OmegaConf.register_new_resolver(
        "phantom-touch", lambda: search_folder("/home", "phantom-touch"))
    
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sam3_cfg = OmegaConf.load(f"{parent_dir}/cfg/sam3_object_by_text.yaml")
    sam3_config = sam3_cfg.sam3
    
    # Create output directory
    if not os.path.exists(sam3_config.output_dir):
        os.makedirs(sam3_config.output_dir)
    
    output_path = sam3_config.output_dir
        
    #### Initialize SAM3 Video Predictor ####
    print("Initializing SAM3 video predictor...")
    video_predictor = build_sam3_video_predictor()
    
    # Collect all episodes paths
    episodes = []
    for root, dirs, files in os.walk(sam3_config.images_path):
        if os.path.basename(root).startswith("e") and os.path.basename(root)[1:].isdigit():
            episodes.append(os.path.basename(root))
    episodes.sort(key=lambda x: int(x[1:]))
    if not episodes:
        print(f"No episodes found in {sam3_config.images_path}")
        sys.exit(1)
    print(f"Found {len(episodes)} episodes: {episodes}")
    episode_paths = []
    for episode in episodes:
        episode_path = os.path.join(sam3_config.images_path, episode)
        episode_paths.append(episode_path)
    
    print(f"Processing {len(episode_paths)} episodes with SAM3...")
    
    #### SAM3 Workflow: Start session and add text prompt ####
    # SAM3 can handle multiple directories as a video sequence
    print(f"Starting SAM3 session with video path: {sam3_config.images_path}")

    for episode_path,episode in zip(episode_paths, episodes):
        try:
            # Start a session - SAM3 will process all frames in the directory structure
            response = video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path= episode_path
                )
            )
            session_id = response["session_id"]
            print(f"Session started with ID: {session_id}")
            
            # Add text prompt to identify the object (e.g., "human hand and arm")
            # Use the configured frame index for the prompt
            print(f"Adding text prompt '{sam3_config.text_prompt}' at frame {sam3_config.frame_index_for_prompt}")
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=sam3_config.frame_index_for_prompt,
                    text=sam3_config.text_prompt,
                )
            )
            
            print("SAM3 propagating masks through video...")
            
            # Get all outputs
            # We need to get masks for all frames
            all_masks = []
            mask_idx = 0
            
            # Process each episode separately to save masks
            episode_input_dir = os.path.join(sam3_config.images_path, episode)
            episode_output_dir = os.path.join(output_path, episode)
            os.makedirs(episode_output_dir, exist_ok=True)
            
            # Get all color frames in this episode
            color_frames = sorted([f for f in os.listdir(episode_input_dir) 
                                if f.startswith("Color_") and f.endswith(".png")])
            
            print(f"Processing episode {episode} with {len(color_frames)} frames...")
            
            # propagate in video
            outputs_per_frame = propagate_in_video(video_predictor, session_id)
            for i,frame_response in enumerate(outputs_per_frame.values()):
                frame_name = color_frames[i]
                try:
                    if "out_binary_masks" in frame_response:
                        mask = frame_response["out_binary_masks"][0]
                        # Convert to uint8 and scale to 0-255
                        mask_img = (mask.astype(np.uint8) * 255)
                        
                        # Convert grayscale to RGB for consistency with SAM2 output
                        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
                        
                        # Save mask with the same naming convention as SAM2
                        mask_output_path = os.path.join(episode_output_dir, 
                                                    frame_name.replace("Color_", "mask_"))
                        cv2.imwrite(mask_output_path, mask_rgb)
                    else:
                        print(f"Warning: No mask found for frame {mask_idx} ({frame_name})")
                        # Create empty mask if none found
                        sample_img = cv2.imread(os.path.join(episode_input_dir, frame_name))
                        if sample_img is not None:
                            empty_mask = np.zeros_like(sample_img)
                            mask_output_path = os.path.join(episode_output_dir, 
                                                        frame_name.replace("Color_", "mask_"))
                            cv2.imwrite(mask_output_path, empty_mask)
                    
                except Exception as e:
                    print(f"Error processing frame {mask_idx} ({frame_name}): {e}")
                    # Create empty mask on error
                    sample_img = cv2.imread(os.path.join(episode_input_dir, frame_name))
                    if sample_img is not None:
                        empty_mask = np.zeros_like(sample_img)
                        mask_output_path = os.path.join(episode_output_dir, 
                                                    frame_name.replace("Color_", "mask_"))
                        cv2.imwrite(mask_output_path, empty_mask)
                
                mask_idx += 1
            
            print(f"Saved {len(color_frames)} masks for {episode}")
            
            # End session
            video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            print("SAM3 session ended successfully")
            
        except Exception as e:
            print(f"Error during SAM3 processing: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        print(f"\n✓ SAM3 segmentation complete!")
        print(f"  Output directory: {output_path}")
        print(f"  Processed {len(episodes)} episodes")
