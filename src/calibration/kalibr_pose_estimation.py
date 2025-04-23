#!/usr/bin/env python

import sys
import cv2
import numpy as np
import argparse
import yaml
import os
from omegaconf import OmegaConf
from pyorbbecsdk import Config
from pyorbbecsdk import OBError
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile

def get_image_from_stream(pipeline,i=0):
    if i==0:
        frames: FrameSet = pipeline.wait_for_frames(5000)
        # skip first 10 frames for stability
        for _ in range(20):
            frames = pipeline.wait_for_frames(100)
    else:
        frames: FrameSet = pipeline.wait_for_frames(1000)
    if frames is None:
        return None
    color_frame = frames.get_color_frame()
    if color_frame is None:
        return None
    color_image = color_frame.get_data()
    color_image = color_image.reshape((1080, 1920, 3))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
    return color_image
def initialize_stream():
    """
    Initialize the Orbbec stream and return the pipeline object.
    """
    config = Config()
    pipeline = Pipeline()
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(1920, 0, OBFormat.RGB, 15)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            print("color profile: ", color_profile)
        config.enable_stream(color_profile)
    except Exception as e:
        print(e)
        return
    pipeline.start(config)
    print("stream started")
    return pipeline
class KalibrDetector:
    """
    A custom detector for Kalibr calibration targets.
    This detector can be adapted for AprilTag, checkerboard, or CircleGrid targets.
    """
    
    def __init__(self, target_type=None, target_params=None, camera_matrix=None, dist_coeffs=None):
        """
        Initialize the detector
        
        Args:
            target_type: Type of calibration target ("checkerboard", "aprilgrid", "circlegrid")
            target_params: Parameters for the specified target type
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
        """
        self.target_type = target_type
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.target_params = target_params
            
        # Initialize detector specific components
        if target_type == "aprilgrid":
            try:
                import apriltag
                options_dict = {
                    "families": "tag36h11",
                    "border": 1,
                    "nthreads": 4,
                    "quad_decimate": 1.0,
                    "quad_sigma": 0.0,
                    "refine_edges": 1,
                    "decode_sharpening": 0.25,
                    "debug": 0
                }
                options = apriltag.DetectorOptions(
                    options_dict['families'],
                    options_dict['border'],
                    options_dict['nthreads'],
                    options_dict['quad_decimate'],
                    options_dict['quad_sigma'],
                    options_dict['refine_edges'],
                    options_dict['decode_sharpening'],
                    options_dict['debug']
                )

                self.apriltag_detector = apriltag.Detector(options=options)
            except ImportError:
                print("Warning: apriltag module not found. AprilTag detection will not work.")
                self.apriltag_detector = None
    

    def detect(self, image):
        """
        Detect the calibration target in the provided image
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            detected: Boolean indicating if target was detected
            corners: List of detected corner points
            ids: IDs for each detected corner (for AprilTags)
        """
        # Ensure image is grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        detected = False
        corners = []
        ids = None
            
        if self.target_type == "aprilgrid" and self.apriltag_detector is not None:
            # Detect AprilTags
            detections = self.apriltag_detector.detect(gray)
            
            if len(detections) > 0:
                detected = True
                ids = []
                
                # Extract corners and IDs
                for det in detections:
                    # Get tag ID
                    tag_id = det.tag_id
                    ids.append(tag_id)
                    
                    # Get corner coordinates (clockwise from top-left)
                    tag_corners = det.corners.astype(np.float32)
                    corners.append(tag_corners)
                
                # Convert to numpy array
                corners = np.array(corners, dtype=np.float32)
                ids = np.array(ids, dtype=np.int32)
                
        return detected, corners, ids
    
    def pose_estimation(self, corners, ids=None, image=None, index=None):
        """
        Estimate the pose (rotation and translation) of the calibration target
        relative to the camera. Requires camera matrix and distortion coefficients.

        Args:
            corners: Detected corner points
            ids: IDs for each detected corner (for AprilTags)
            
        Returns:
            success: Boolean indicating if pose estimation was successful
            rvec: Rotation vector
            tvec: Translation vector
            transformation_matrix: 4x4 transformation matrix from camera to board
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            print("Camera matrix or distortion coefficients not provided. Cannot estimate pose.")
            return False, None, None, None

        # Initialize object points based on target type
        objp = None
        imgp = None
            
        if self.target_type == "aprilgrid" and ids is not None:
            # For AprilGrid, we need to organize the corners based on their tag IDs
            tag_size = self.target_params["tagSize"]  # Size of a tag (without border)
            tag_spacing = self.target_params["tagSpacing"]  # Spacing as a fraction of tag size
            tag_cols = self.target_params["tagCols"]  # Number of tags per row
            
            # Calculate the total spacing (distance between tag centers)
            # If tagSpacing = 0.3, then tags are separated by 0.3 * tagSize
            actual_spacing = tag_size * (1 + tag_spacing)
            
            # Create empty lists for object and image points
            objp = []
            imgp = []

            for i in range(len(ids)):
                tag_id = ids[i]  # Extract the tag ID value
                tag_corners = corners[i].reshape(4, 2)  # Get the 4 corners of this tag
                
                # Calculate tag position in the grid (bottom-left origin)
                row = tag_id // tag_cols   # Row 0 = bottom row
                col = tag_id % tag_cols    # Column 0 = leftmost column

                # Tag CENTER position in world coordinates (X-right, Y-up)
                tag_center_x = col * actual_spacing
                tag_center_y = row * actual_spacing

                # Top-left corner of the tag (world frame)
                tag_origin_x = tag_center_x
                tag_origin_y = tag_center_y + tag_size  # Y increases upwards

                # Corners in clockwise order from top-left (world frame)
                tag_corners_obj = np.array([
                    [tag_origin_x, tag_origin_y, 0],                         # Top-left
                    [tag_origin_x + tag_size, tag_origin_y, 0],               # Top-right
                    [tag_origin_x + tag_size, tag_origin_y - tag_size, 0],    # Bottom-right
                    [tag_origin_x, tag_origin_y - tag_size, 0]                # Bottom-left
                ], dtype=np.float32)
                # Add the points to our lists
                objp.extend(tag_corners_obj)
                imgp.extend(tag_corners)
            # Convert to numpy arrays
            objp = np.array(objp, dtype=np.float32)
            imgp = np.array(imgp, dtype=np.float32)
        if objp is None or imgp is None or len(objp) < 4 or len(imgp) < 4:
            print("Invalid points for pose estimation")
            return False, None, None, None

        # Ensure objp and imgp have the same number of points
        if len(objp) != len(imgp):
            print(f"Mismatch in point count: objp={len(objp)}, imgp={len(imgp)}")
            return False, None, None, None
            
        # Estimate pose
        try:
            success, rvec, tvec = cv2.solvePnP(
                objp, imgp, self.camera_matrix, self.dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return False, None, None, None
        
            # Convert rotation vector to rotation matrix
            rmat, _ = cv2.Rodrigues(rvec)
            
            # Create transformation matrix (4x4)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rmat
            transformation_matrix[:3, 3] = tvec.flatten()
            return success, rvec, tvec, transformation_matrix
                            
        except cv2.error as e:
            print(f"OpenCV error during pose estimation: {e}")
            return False, None, None, None
    
    def draw_pose_axes(self, image, rvec, tvec, axis_length=0.1):
        """
        Draw 3D axes on the image to visualize the pose
        
        Args:
            image: Input image
            rvec: Rotation vector
            tvec: Translation vector
            axis_length: Length of the axes to draw
            
        Returns:
            image: Image with drawn axes
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return image
        
        # Define axes points in 3D (origin, X, Y, Z)
        axis_points = np.float32([
            [0, 0, 0], 
            [axis_length, 0, 0], 
            [0, axis_length, 0], 
            [0, 0, axis_length]
        ])
        
        # Project 3D points to image plane
        imgpts, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        imgpts = np.int32(imgpts).reshape(-1, 2)
        
        # Draw axes
        origin = tuple(imgpts[0])
        image = cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis (red)
        image = cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y-axis (green)
        image = cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z-axis (blue)
        
        return image
    
    def draw_detections(self, image, corners, ids=None):
        """
        Draw the detected calibration target on the image
        
        Args:
            image: Input image
            corners: Detected corner points
            ids: IDs for each detected corner (for AprilTags)
            
        Returns:
            image: Image with drawn detections
        """
        # Create a copy of the image to draw on
        output = image.copy()
        
        
        if self.target_type == "aprilgrid" and corners is not None:
            # Draw AprilTag detections
            if len(corners.shape) == 3:  # Multiple detections
                for i in range(corners.shape[0]):
                    # Draw the perimeter of the tag
                    points = corners[i].astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(output, [points], True, (0, 255, 0), 2)
                    
                    # Draw the tag ID
                    if ids is not None:
                        tag_id = ids[i]
                        center = np.mean(corners[i], axis=0).astype(int)
                        cv2.putText(output, str(tag_id), tuple(center), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:  # Single detection
                points = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(output, [points], True, (0, 255, 0), 2)
                
        return output
    
    def export_to_kalibr_yaml(self, output_path):
        """
        Export the detector configuration to a Kalibr-compatible YAML file
        
        Args:
            output_path: Path to save the YAML file
        """
        target_config = {}
        
        if self.target_type == "aprilgrid":
            target_config = {
                "target_type": "aprilgrid",
                "tagCols": self.target_params["tagCols"],
                "tagRows": self.target_params["tagRows"],
                "tagSize": self.target_params["tagSize"],
                "tagSpacing": self.target_params["tagSpacing"]
            }
        
        # Write to YAML file
        with open(output_path, 'w') as f:
            yaml.dump(target_config, f, default_flow_style=False)
        
        print(f"Configuration saved to {output_path}")


def main():
    args = OmegaConf.load('conf/kalibr.yaml')
    # Initialize target parameters from args
    target_params = {}
    if args.target_type == "aprilgrid":
        if args.cols:
            target_params["tagCols"] = args.cols
        if args.rows:
            target_params["tagRows"] = args.rows
        if args.size:
            target_params["tagSize"] = args.size
        if args.tag_spacing:
            target_params["tagSpacing"] = args.tag_spacing
    
    # construct camera matrix from fx, fy, cx and cy
    camera_matrix = np.array([[args.fx, 0, args.cx],
                                [0, args.fy, args.cy],
                                [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([args.k1, args.k2, args.p1, args.p2], dtype=np.float32)
    # Create detector
    detector = KalibrDetector(
        args.target_type, 
        target_params if target_params else None,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs
    )
    print(f"Detector initialized with target type: {args.target_type}")
    # Export configuration to YAML
    detector.export_to_kalibr_yaml(args.output_yaml)
    
    # Get image from the Orbbec stream
    pipeline=initialize_stream()
    for i in range(100):
        cv2.waitKey(1)
        print(f"Processing frame {i}")
        image = get_image_from_stream(pipeline,i)
        if image is None:
            print("Error: Failed to retrieve image from Orbbec stream")
            return
            
        # Detect targets
        detected, corners, ids = detector.detect(image)

        if detected:
            print(f"Target detected with {len(corners)} corners")
            
            # Draw detections
            result = detector.draw_detections(image, corners, ids)
            
            # Estimate pose if camera matrix is available
            if camera_matrix is not None:
                success, rvec, tvec, transformation_matrix = detector.pose_estimation(corners, ids,image,i)
                if success:
                    print("Pose estimation successful")
                    print("Translation vector (camera to board):")
                    print(tvec)
                    print("Rotation vector (camera to board):")
                    print(rvec)
                    print("Transformation matrix (camera to board):")
                    print(transformation_matrix)
                    
                    # Draw pose axes
                    result = detector.draw_pose_axes(result, rvec, tvec)
                    
                    # Save transformation matrix to file
                    np.save(os.path.splitext(args.output)[0] + f'_transform_{i}.npy', transformation_matrix)
                else:
                    print("Pose estimation failed")
            
            if args.show:
                cv2.imshow('Detection Result', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Save result
            output_path = os.path.splitext(args.output)[0] + f'_detected_{i}.png'
            cv2.imwrite(output_path, result)
            print(f"Detection result saved to {output_path}")
        else:
            print("No target detected in the image")


if __name__ == "__main__":
    main()