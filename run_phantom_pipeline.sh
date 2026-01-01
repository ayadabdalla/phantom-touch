#!/bin/bash

###############################################################################
# Phantom-Touch Pipeline Automation Script
# 
# PURPOSE:
#   Automates the complete phantom-touch data processing pipeline, transforming
#   raw Orbbec camera recordings of human demonstrations into a touch-aware
#   robot manipulation dataset with 3D object tracking and contact information.
#
# PIPELINE STAGES (8 steps):
#   0. Update Configuration     - Set experiment paths in config files
#   1. VitPose Segmentation     - Detect and segment human hands
#   2. Split Episodes           - Organize data into individual episodes
#   3. SAM3 Hand Masks          - Generate precise hand masks using text prompts
#   4. 3D Hand Projection       - Project 2D hand masks to 3D coordinates
#   5. Inpainting               - Remove hands from images to isolate objects
#   6. Phantom Data Creation    - Assemble final dataset structure
#   7. 3D Object Tracking       - Track object pose using ICP algorithm
#   8. Render Contact Patches   - Generate depth patches for tactile learning
#
# PREREQUISITES:
#   1. Orbbec data collection completed (using OBSaveToDisk)
#   2. Virtual environment activated: source .phantom-touch/bin/activate
#   3. All required Python packages installed (see requirements/)
#
# USAGE:
#   Interactive mode (with pauses between steps):
#     ./run_phantom_pipeline.sh [experiment_name] [data_dir] --interactive
#
#   Automated mode:
#     ./run_phantom_pipeline.sh [experiment_name] [data_dir]
#
#   Skip specific steps:
#     ./run_phantom_pipeline.sh experiment_0 /data --skip-step 1 --skip-step 2
#
# EXAMPLES:
#   ./run_phantom_pipeline.sh pick_and_place /home/epon04yc --interactive
#   ./run_phantom_pipeline.sh experiment_0 /home/epon04yc --skip-step 7
#
###############################################################################

set -e  # Exit immediately if any command fails
set -u  # Exit if undefined variable is referenced

# Color codes for terminal output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script execution mode
INTERACTIVE_MODE=false  # Set to true for step-by-step debugging with pauses

# Default configuration values
EXPERIMENT_NAME="${1:-example_2}"  #@user: change to your experiment name
DATA_DIR="${2:-/home/epon04yc}"
MODEL_DIR="/mnt/dataset_drive/ayad/data/phantom-touch" #@user: change to your model directory
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shift the first two positional parameters so we can parse additional flags
shift 2

# CAD model configuration (adjust as needed)
CAD_MODEL_PATH="/mnt/dataset_drive/ayad/scenes_and_cad/cad_models/Strawberry.obj" #@user: change to your CAD model path
CAD_SCALE="0.001" #@user: change to your CAD model scale
TEXT_PROMPT="human"  #@user: text prompt for hand segmentation
OBJECT_TEXT_PROMPT="green strawberry" #@user: text prompt for object segmentation

# Episode range
START_EPISODE=0
END_EPISODE=0  # Auto-detected after Step 2 (split_episodes) - no need to set manually

# Camera transform path
CAMERA_TRANSFORM_PATH="${REPO_DIR}/src/cameras/Orbbec/data/robotbase_camera_transform_orbbec_fr4.npy"

# Directories
EXPERIMENT_DIR="${DATA_DIR}/${EXPERIMENT_NAME}"
EPISODES_DIR="${EXPERIMENT_DIR}/episodes"
VITPOSE_OUTPUT_DIR="${EXPERIMENT_DIR}/vitpose_output/episodes"
SAM3HAND_OUTPUT_DIR="${EXPERIMENT_DIR}/sam3hand_output/episodes"
SAM3VID_OUTPUT_DIR="${EXPERIMENT_DIR}/sam3_output/episodes"
INPAINTING_OUTPUT_DIR="${EXPERIMENT_DIR}/inpainting_output/episodes"
DATASET_DIR="${EXPERIMENT_DIR}/dataset"
TRACKING_OUTPUT_DIR="${EXPERIMENT_DIR}/threeD_tracking_offline"
OBJECT_MASKS_DIR="${EXPERIMENT_DIR}/object_masks"

# Array to store step numbers that should be skipped
SKIP_STEPS=()

###############################################################################
# Helper Functions - Output Formatting and User Interaction
###############################################################################

# Display a prominent section header
print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Print success message with green checkmark
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message with red X (typically exits after this)
print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

# Print warning message with yellow triangle
print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

# Print informational message with cyan icon
print_info() {
    echo -e "${CYAN}ℹ INFO: $1${NC}"
}

# Print debug information in magenta
print_debug() {
    echo -e "${MAGENTA}🔍 DEBUG: $1${NC}"
}

# Pause execution and wait for user confirmation (interactive mode only)
pause_for_verification() {
    if [[ "${INTERACTIVE_MODE}" == "true" ]]; then
        echo ""
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${YELLOW}⏸  PAUSED FOR VERIFICATION${NC}"
        echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "   Please verify the output above looks correct."
        echo -e "   Press ENTER to continue to next step, or Ctrl+C to abort..."
        read -r
    fi
}

# Verify that output files/directories were created successfully
verify_output() {
    local path=$1
    local description=$2
    
    if [[ -e "$path" ]]; then
        # Count files if it's a directory
        if [[ -d "$path" ]]; then
            local file_count=$(find "$path" -type f 2>/dev/null | wc -l)
            print_success "${description} exists with ${file_count} files"
        else
            print_success "${description} exists"
        fi
        return 0
    else
        print_error "${description} not found at: $path"
        return 1
    fi
}

should_skip_step() {
    local step=$1
    for skip in "${SKIP_STEPS[@]}"; do
        if [[ "$skip" == "$step" ]]; then
            return 0
        fi
    done
    return 1
}

# Check if a directory exists, exit with error if not found
check_directory() {
    if [[ ! -d "$1" ]]; then
        print_error "Required directory not found: $1"
        print_info "Please verify the path exists and try again"
        exit 1
    fi
    print_debug "Directory verified: $1"
}

# Create directory if it doesn't exist, no error if already present
create_directory() {
    if [[ ! -d "$1" ]]; then
        mkdir -p "$1"
        print_success "Created directory: $1"
    else
        print_debug "Directory already exists: $1"
    fi
}

# Update a value in a YAML configuration file
# Args: file path, key name, new value
update_yaml_value() {
    local file=$1
    local key=$2
    local value=$3
    
    if [[ -f "$file" ]]; then
        # Use sed to update YAML values (simple key: value replacement)
        if grep -q "^${key}:" "$file"; then
            sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
            print_debug "Updated ${key} in $(basename $file)"
        else
            print_warning "Key '${key}' not found in $(basename $file)"
        fi
    else
        print_error "Config file not found: $file"
        exit 1
    fi
}

###############################################################################
# Parse Command-Line Arguments
###############################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-step)
            SKIP_STEPS+=("$2")
            print_info "Will skip step: $2"
            shift 2
            ;;
        --interactive|-i)
            INTERACTIVE_MODE=true
            print_info "Interactive mode enabled - will pause between steps"
            shift
            ;;
        *)
            print_warning "Unknown argument: $1"
            shift
            ;;
    esac
done

###############################################################################
# Pipeline Configuration and Prerequisites Check
###############################################################################

print_header "Phantom-Touch Pipeline Configuration"
echo -e "${CYAN}Experiment Settings:${NC}"
echo "  • Experiment Name:    ${EXPERIMENT_NAME}"
echo "  • Data Directory:     ${DATA_DIR}"
echo "  • Model Directory:    ${MODEL_DIR}"
echo "  • Repository:         ${REPO_DIR}"
echo "  • Episode Range:      ${START_EPISODE} to ${END_EPISODE}"
echo "  • Interactive Mode:   ${INTERACTIVE_MODE}"
echo ""

# Display CAD and segmentation settings
echo -e "${CYAN}Segmentation Settings:${NC}"
echo "  • Hand Prompt:        ${TEXT_PROMPT}"
echo "  • Object Prompt:      ${OBJECT_TEXT_PROMPT}"
echo "  • CAD Model:          ${CAD_MODEL_PATH}"
echo "  • CAD Scale:          ${CAD_SCALE}"
echo ""

# Verify prerequisites
print_info "Checking prerequisites..."

# Check if we're in the correct repository directory
if [[ ! -f "${REPO_DIR}/setup.py" ]]; then
    print_error "Not in phantom-touch repository root"
    print_info "Expected to find setup.py in: ${REPO_DIR}"
    exit 1
fi
print_debug "Repository root verified"

# Check if virtual environment is activated (recommended but not required)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    print_warning "Virtual environment not detected"
    print_info "Recommended: source .phantom-touch/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_debug "Virtual environment active: ${VIRTUAL_ENV}"
fi

# Verify experiment directory exists
check_directory "${EXPERIMENT_DIR}"

print_success "Prerequisites check complete"
echo ""

###############################################################################
# Step 0: Update Main Paths Configuration
###############################################################################
# PURPOSE: Update the main configuration file (cfg/paths.yaml) with experiment
#          paths so all subsequent steps know where to find/save data.
# INPUT:   cfg/paths.yaml (template configuration)
# OUTPUT:  cfg/paths.yaml (updated with experiment-specific paths)
# NOTES:   Creates timestamped backup before modifying
###############################################################################

if ! should_skip_step 0; then
    print_header "Step 0: Updating Main Configuration (cfg/paths.yaml)"
    
    PATHS_CONFIG="${REPO_DIR}/cfg/paths.yaml"
    
    # Create timestamped backup of original configuration
    BACKUP_FILE="${PATHS_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "${PATHS_CONFIG}" "${BACKUP_FILE}"
    print_info "Backup created: $(basename ${BACKUP_FILE})"
    
    # Update main configuration paths
    print_info "Updating experiment name to: ${EXPERIMENT_NAME}"
    update_yaml_value "${PATHS_CONFIG}" "  experiment_name" "${EXPERIMENT_NAME}"
    
    print_info "Updating data directory to: ${DATA_DIR}"
    update_yaml_value "${PATHS_CONFIG}" "  data_dir" "${DATA_DIR}"
    
    print_success "Main configuration updated successfully"
    print_info "Modified: cfg/paths.yaml"
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 0: Update Configuration"
fi

###############################################################################
# Step 1: VitPose Hand Segmentation
###############################################################################
# PURPOSE: Detect and segment human hands in RGB images using VitPose model.
#          This identifies which pixels belong to hands for later removal.
# INPUT:   Raw RGB images from episodes directory
# OUTPUT:  Hand keypoints and segmentation masks in vitpose_output/
# MODEL:   VitPose - Vision Transformer for pose estimation
# NOTES:   Processes all frames across all episodes
###############################################################################

if ! should_skip_step 1; then
    print_header "Step 1: VitPose Hand Segmentation"
    
    VITPOSE_CONFIG="${REPO_DIR}/src/segment_hands/cfg/vitpose_segmentation.yaml"
    
    # Create timestamped backup and update configuration
    cp "${VITPOSE_CONFIG}" "${VITPOSE_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    print_info "Configuring VitPose for experiment: ${EXPERIMENT_NAME}"
    update_yaml_value "${VITPOSE_CONFIG}" "experiment_name" "${EXPERIMENT_NAME}"
    update_yaml_value "${VITPOSE_CONFIG}" "img_folder" "${EXPERIMENT_DIR}"
    update_yaml_value "${VITPOSE_CONFIG}" "vitpose_out_folder" "${VITPOSE_OUTPUT_DIR}"
    
    # Ensure output directory exists
    create_directory "${VITPOSE_OUTPUT_DIR}"
    
    print_info "Running VitPose hand detection and segmentation..."
    print_warning "This may take several minutes depending on number of frames"
    cd "${REPO_DIR}/src/segment_hands/scripts"
    python run_vitpose.py
    
    # Verify output was created
    verify_output "${VITPOSE_OUTPUT_DIR}" "VitPose output directory"
    
    print_success "VitPose hand segmentation complete"
    print_info "Hand masks saved to: ${VITPOSE_OUTPUT_DIR}"
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 1: VitPose Hand Segmentation"
fi

###############################################################################
# Step 2: Split Episodes
###############################################################################
# PURPOSE: Organize collected data into individual episode directories.
#          Each episode represents one demonstration/manipulation sequence.
# INPUT:   Raw continuous data recordings
# OUTPUT:  Structured episode directories (e0, e1, e2, etc.)
# NOTES:   Episodes contain synchronized RGB, depth, and robot state data
###############################################################################

if ! should_skip_step 2; then
    print_header "Step 2: Splitting Episodes"
    
    PREPROCESSOR_CONFIG="${REPO_DIR}/src/phantom_touch/cfg/preprocessors.yaml"
    
    # Create timestamped backup of preprocessor configuration
    cp "${PREPROCESSOR_CONFIG}" "${PREPROCESSOR_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    print_info "Splitting continuous recording into individual episodes..."
    print_info "Target range: Episode ${START_EPISODE} to ${END_EPISODE}"
    
    cd "${REPO_DIR}/src/phantom_touch/preprocessors"
    python split_episodes.py
    
    # Verify episodes directory and check episode count
    verify_output "${EPISODES_DIR}" "Episodes directory"
    
    # Count created episodes and auto-detect episode range
    if [[ -d "${EPISODES_DIR}" ]]; then
        EPISODE_COUNT=$(find "${EPISODES_DIR}" -maxdepth 1 -type d -name "e[0-9]*" | wc -l)
        print_success "Episodes split successfully: ${EPISODE_COUNT} episodes created"
        
        # Auto-detect END_EPISODE based on actual episodes created
        if [[ ${EPISODE_COUNT} -gt 0 ]]; then
            END_EPISODE=$((EPISODE_COUNT - 1))
            print_info "Auto-detected episode range: ${START_EPISODE} to ${END_EPISODE}"
        else
            print_error "No episodes found in ${EPISODES_DIR}"
            exit 1
        fi
    fi
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 2: Split Episodes"
    print_warning "END_EPISODE remains at initial value: ${END_EPISODE}"
fi

###############################################################################
# Step 3: SAM3 Hand Mask Generation
###############################################################################
# PURPOSE: Generate precise, temporally-consistent hand segmentation masks
#          using SAM3 (Segment Anything Model 3) with text prompts.
# INPUT:   Episode RGB images + text prompt (e.g., "human hand and arm")
# OUTPUT:  High-quality binary hand masks in sam3_output/episodes/
# MODEL:   SAM3 - Latest version with text-based segmentation (no API needed)
# NOTES:   Improves upon VitPose masks with temporal consistency across frames
###############################################################################

if ! should_skip_step 3; then
    print_header "Step 3: SAM3 Hand Mask Generation"
    
    SAM3_CONFIG="${REPO_DIR}/src/sam3_session/cfg/sam3_object_by_text.yaml"
    
    # Create timestamped backup and update configuration
    cp "${SAM3_CONFIG}" "${SAM3_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Update nested YAML values using sed (for indented keys)
    print_info "Configuring SAM3 with text prompt: '${TEXT_PROMPT}'"
    sed -i "s|  experiment_name:.*|  experiment_name: ${EXPERIMENT_NAME}|" "${SAM3_CONFIG}"
    sed -i "s|  data_dir:.*|  data_dir: ${DATA_DIR}|" "${SAM3_CONFIG}"
    sed -i "s|  text_prompt:.*|  text_prompt: ${TEXT_PROMPT}|" "${SAM3_CONFIG}"
    
    # Ensure output directory exists
    create_directory "${SAM3VID_OUTPUT_DIR}"
    
    print_info "Running SAM3 hand segmentation..."
    print_info "Processing episodes ${START_EPISODE} to ${END_EPISODE}"
    print_warning "This step may take 5-10 minutes per episode"
    
    cd "${REPO_DIR}/src/sam3_session/scripts"
    python segment_objVideo_byText.py
    
    # Verify output was created
    verify_output "${SAM3VID_OUTPUT_DIR}" "SAM3 hand masks"
    
    print_success "SAM3 hand masks generated successfully"
    print_info "Masks saved to: ${SAM3VID_OUTPUT_DIR}"
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 3: SAM3 Hand Mask Generation"
fi

###############################################################################
# Step 4: 3D Hand Projection
###############################################################################
# PURPOSE: Project 2D hand segmentation masks into 3D space using depth data.
#          This creates 3D point clouds of hand positions for spatial analysis.
# INPUT:   - 2D hand masks from SAM3
#          - Depth images from Orbbec camera
#          - Camera intrinsic/extrinsic parameters
# OUTPUT:  3D hand point clouds with spatial coordinates
# NOTES:   Enables filtering of depth data to remove hand occlusions
###############################################################################

if ! should_skip_step 4; then
    print_header "Step 4: 3D Hand Projection"
    
    PROJECTION_CONFIG="${REPO_DIR}/src/segment_hands/cfg/3d_projection.yaml"
    
    # Create timestamped backup
    cp "${PROJECTION_CONFIG}" "${PROJECTION_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    print_info "Projecting 2D hand masks to 3D using depth data..."
    print_info "Using camera transform: ${CAMERA_TRANSFORM_PATH}"
    
    cd "${REPO_DIR}/src/segment_hands/scripts"
    python project_sam3hand_to_3d.py
    
    print_success "3D hand projection complete"
    print_info "3D hand data computed for all episodes"
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 4: 3D Hand Projection"
fi

###############################################################################
# Step 5: Inpainting (Hand Removal)
###############################################################################
# PURPOSE: Remove hands from RGB images to isolate the manipulated object.
#          Uses AI inpainting to fill in the area where hands were present.
# INPUT:   - Original RGB images
#          - SAM3 hand masks (defining areas to remove)
# OUTPUT:  Hand-free RGB images in inpainting_output/episodes/
# MODEL:   Deep learning inpainting model (Stable Diffusion based)
# NOTES:   Critical for clean object tracking without hand occlusions
###############################################################################

if ! should_skip_step 5; then
    print_header "Step 5: Inpainting - Removing Hands from Images"
    
    INPAINT_CONFIG="${REPO_DIR}/src/inpainting/cfg/inpaint.yaml"
    
    # Create timestamped backup and update configuration
    cp "${INPAINT_CONFIG}" "${INPAINT_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    print_info "Configuring inpainting pipeline..."
    sed -i "s|  rgb_base_path:.*|  rgb_base_path: ${EPISODES_DIR}|" "${INPAINT_CONFIG}"
    sed -i "s|  mask_base_path:.*|  mask_base_path: ${SAM3VID_OUTPUT_DIR}|" "${INPAINT_CONFIG}"
    sed -i "s|  output_path:.*|  output_path: ${INPAINTING_OUTPUT_DIR}|" "${INPAINT_CONFIG}"
    sed -i "s|  episode_start:.*|  episode_start: ${START_EPISODE}|" "${INPAINT_CONFIG}"
    sed -i "s|  episode_end:.*|  episode_end: $((END_EPISODE + 1))|" "${INPAINT_CONFIG}"
    
    # Ensure output directory exists
    create_directory "${INPAINTING_OUTPUT_DIR}"
    
    print_info "Running AI inpainting model to remove hands..."
    print_info "Processing episodes ${START_EPISODE} to ${END_EPISODE}"
    print_warning "This is computationally intensive - may take 10-20 min per episode"
    
    cd "${REPO_DIR}/src/inpainting/scripts"
    python inpaint.py "${INPAINT_CONFIG}"
    
    # Verify output was created
    verify_output "${INPAINTING_OUTPUT_DIR}" "Inpainted images"
    
    print_success "Inpainting complete - hands removed from all images"
    print_info "Hand-free images saved to: ${INPAINTING_OUTPUT_DIR}"
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 5: Inpainting"
fi

###############################################################################
# Step 6: Phantom Data Creation
###############################################################################
# PURPOSE: Assemble the final dataset structure combining all processed data.
#          Creates the "phantom touch" dataset with robot states, images,
#          and tactile information for imitation learning.
# INPUT:   - Inpainted RGB images (hands removed)
#          - Depth images
#          - Robot joint states and end-effector poses
#          - Hand 3D positions (for touch inference)
# OUTPUT:  Structured dataset in dataset/ directory ready for training
# NOTES:   This is the main dataset used for robot learning
###############################################################################

if ! should_skip_step 6; then
    print_header "Step 6: Phantom Data Creation"

    Phantom_CONFIG="${REPO_DIR}/src/phantom_touch/cfg/phantom.yaml"

    # Create timestamped backup and update episode range
    cp "${Phantom_CONFIG}" "${Phantom_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"

    print_info "Configuring phantom touch dataset creation..."
    print_info "Episode range: ${START_EPISODE} to ${END_EPISODE}"
    
    # Update episode range in the YAML file
    sed -i "s|  start:.*|  start: ${START_EPISODE}|" "${Phantom_CONFIG}"
    sed -i "s|  end:.*|  end: ${END_EPISODE}|" "${Phantom_CONFIG}"
    
    # Ensure dataset output directory exists
    create_directory "${DATASET_DIR}"
    
    print_info "Creating phantom touch dataset..."
    print_info "Combining: inpainted images + depth + robot states + touch info"
    
    cd "${REPO_DIR}/src/phantom_touch/scripts"
    python phantom_process_data.py
    
    # Verify dataset was created
    verify_output "${DATASET_DIR}" "Phantom touch dataset"
    
    print_success "Phantom dataset created successfully"
    print_info "Dataset location: ${DATASET_DIR}"
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 6: Phantom Data Creation"
fi

###############################################################################
# Step 7: 3D Object Tracking (Offline)
###############################################################################
# PURPOSE: Track the manipulated object's 6D pose (position + orientation)
#          throughout each episode using Iterative Closest Point (ICP).
# INPUT:   - Inpainted RGB images (hands removed)
#          - Depth point clouds
#          - CAD model of the object
#          - Object segmentation masks (from text prompt)
# OUTPUT:  - 6D object poses per frame in threeD_tracking_offline/
#          - Object segmentation masks in object_masks/
# ALGORITHM: ICP (Iterative Closest Point) alignment between observed point
#            cloud and CAD model, refined frame-by-frame
# NOTES:   Requires accurate CAD model and scale for best results
###############################################################################

if ! should_skip_step 7; then
    print_header "Step 7: 3D Object Tracking (Offline ICP)"
    
    TRACKING_CONFIG="${REPO_DIR}/src/render_contact_depth_patches/threeDoffline_object_tracking/cfg/threeD_tracking_offline.yaml"
    
    # Create timestamped backup and update configuration
    cp "${TRACKING_CONFIG}" "${TRACKING_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    print_info "Configuring 3D object tracking..."
    print_info "CAD Model: ${CAD_MODEL_PATH}"
    print_info "CAD Scale: ${CAD_SCALE}"
    print_info "Object Prompt: '${OBJECT_TEXT_PROMPT}'"
    
    update_yaml_value "${TRACKING_CONFIG}" "CAD_MODEL_PATH" "${CAD_MODEL_PATH}"
    update_yaml_value "${TRACKING_CONFIG}" "cad_scale" "${CAD_SCALE}"
    update_yaml_value "${TRACKING_CONFIG}" "CAMERA_TO_ROBOT_TRANSFORM_PATH" "${CAMERA_TRANSFORM_PATH}"
    update_yaml_value "${TRACKING_CONFIG}" "start_episode" "${START_EPISODE}"
    update_yaml_value "${TRACKING_CONFIG}" "end_episode" "${END_EPISODE}"
    
    # Update object segmentation prompt
    sed -i "s|  language_prompt:.*|  language_prompt: \"${OBJECT_TEXT_PROMPT}\"|" "${TRACKING_CONFIG}"
    
    # Ensure output directories exist
    create_directory "${TRACKING_OUTPUT_DIR}"
    create_directory "${OBJECT_MASKS_DIR}"
    
    print_info "Running 3D object tracking with ICP algorithm..."
    print_warning "This step is computationally intensive:"
    print_warning "  - Point cloud processing for each frame"
    print_warning "  - ICP alignment iterations"
    print_warning "  - Estimated time: 15-30 minutes per episode"
    
    cd "${REPO_DIR}/src/render_contact_depth_patches/threeDoffline_object_tracking"
    python threeD_tracking_offline.py
    
    # Verify outputs were created
    verify_output "${TRACKING_OUTPUT_DIR}" "3D tracking results"
    verify_output "${OBJECT_MASKS_DIR}" "Object segmentation masks"
    
    print_success "3D object tracking complete"
    print_info "Tracking data: ${TRACKING_OUTPUT_DIR}"
    print_info "Object masks: ${OBJECT_MASKS_DIR}"
    
    # Pause for user verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 7: 3D Object Tracking"
fi

###############################################################################
# Step 8: Render Contact Depth Patches
###############################################################################
# PURPOSE: Generate local depth patches at predicted contact points for
#          tactile learning. These patches represent what the tactile sensor
#          would observe during contact.
# INPUT:   - 3D object poses from tracking
#          - Contact point predictions (from hand proximity)
#          - Object CAD model
# OUTPUT:  Rendered depth patches representing tactile observations
# NOTES:   Creates synthetic tactile data from vision for touch-aware learning
###############################################################################

if ! should_skip_step 8; then
    print_header "Step 8: Rendering Contact Depth Patches"
    
    print_info "Rendering depth patches at contact locations..."
    print_info "These patches simulate tactile sensor observations"
    
    cd "${REPO_DIR}/src/render_contact_depth_patches"
    python render_depth_patches_phantom.py
    
    print_success "Contact depth patches rendered successfully"
    print_info "Tactile data ready for robot learning"
    
    # Pause for final verification in interactive mode
    pause_for_verification
else
    print_warning "Skipping Step 8: Render Contact Depth Patches"
fi

###############################################################################
# Pipeline Completion Summary
###############################################################################

print_header "🎉 Pipeline Complete!"
echo ""
echo -e "${GREEN}✓ All steps completed successfully!${NC}"
echo ""
echo -e "${CYAN}Output Locations:${NC}"
echo "  📁 VitPose outputs:       ${VITPOSE_OUTPUT_DIR}"
echo "  📁 SAM3 hand masks:       ${SAM3VID_OUTPUT_DIR}"
echo "  📁 Inpainted images:      ${INPAINTING_OUTPUT_DIR}"
echo "  📁 Phantom dataset:       ${DATASET_DIR}"
echo "  📁 3D object tracking:    ${TRACKING_OUTPUT_DIR}"
echo "  📁 Object masks:          ${OBJECT_MASKS_DIR}"
echo ""
echo -e "${CYAN}Dataset Summary:${NC}"
if [[ -d "${DATASET_DIR}" ]]; then
    DATASET_SIZE=$(du -sh "${DATASET_DIR}" 2>/dev/null | cut -f1)
    echo "  💾 Total dataset size:    ${DATASET_SIZE}"
fi
if [[ -d "${EPISODES_DIR}" ]]; then
    EPISODE_COUNT=$(find "${EPISODES_DIR}" -maxdepth 1 -type d -name "e[0-9]*" | wc -l)
    echo "  📊 Number of episodes:    ${EPISODE_COUNT}"
fi
echo ""
echo -e "${BLUE}ℹ Configuration backups saved with timestamps${NC}"
echo ""

# Offer to restore original config files
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
read -p "Restore original config files? (y/n) " -n 1 -r
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Restoring original configurations..."
    BACKUP_COUNT=0
    find "${REPO_DIR}" -name "*.yaml.backup.*" -type f 2>/dev/null | while read backup; do
        original="${backup%%.*}"
        if [[ -f "$original" ]]; then
            mv "$backup" "$original"
            print_debug "Restored $(basename $original)"
            BACKUP_COUNT=$((BACKUP_COUNT + 1))
        fi
    done
    print_success "Original configurations restored"
else
    print_info "Configuration backups kept for reference"
    print_info "To manually restore: find . -name '*.yaml.backup.*'"
fi

echo ""
print_header "✅ Phantom-Touch Pipeline Finished!"
echo ""
echo -e "${GREEN}All processing steps completed successfully.${NC}"
echo -e "${CYAN}Your dataset is ready for robot learning!${NC}"
echo ""
