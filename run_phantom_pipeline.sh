#!/bin/bash

###############################################################################
# Phantom-Touch Pipeline Automation Script
# 
# This script automates the entire phantom-touch data processing pipeline from
# raw Orbbec recordings to final touch-aware robot data.
#
# Prerequisites:
# 1. Orbbec data collection completed (using OBSaveToDisk)
# 2. Virtual environment activated: source .phantom-touch/bin/activate
# 3. SIEVE_API_KEY exported: export SIEVE_API_KEY="<your_key>"
#
# Usage:
#   ./run_phantom_pipeline.sh [experiment_name] [data_dir] [--skip-step <step_num>]
#
# Example:
#   ./run_phantom_pipeline.sh pick_and_place_phantom /home/epon04yc
###############################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
EXPERIMENT_NAME="${1:-pick_and_place_phantom}"
DATA_DIR="${2:-/home/epon04yc}"
MODEL_DIR="${3:-/mnt/dataset_drive/ayad/data/phantom-touch}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# CAD model configuration (adjust as needed)
CAD_MODEL_PATH="/mnt/dataset_drive/ayad/scenes_and_cad/cad_models/Strawberry.obj"
CAD_SCALE="0.001"
SIEVE_TEXT_PROMPT="human hand and arm"
OBJECT_TEXT_PROMPT="green object"

# Episode range (adjust as needed)
START_EPISODE=0
END_EPISODE=11

# Camera transform path
CAMERA_TRANSFORM_PATH="${REPO_DIR}/src/cameras/Orbbec/data/robotbase_camera_transform_orbbec_fr4.npy"

# Directories
EXPERIMENT_DIR="${DATA_DIR}/${EXPERIMENT_NAME}"
EPISODES_DIR="${EXPERIMENT_DIR}/episodes"
VITPOSE_OUTPUT_DIR="${EXPERIMENT_DIR}/vitpose_output/episodes"
SAM2HAND_OUTPUT_DIR="${EXPERIMENT_DIR}/sam2hand_output/episodes"
SAM2VID_OUTPUT_DIR="${EXPERIMENT_DIR}/sam2-vid_output/episodes"
INPAINTING_OUTPUT_DIR="${EXPERIMENT_DIR}/inpainting_output/episodes"
DATASET_DIR="${EXPERIMENT_DIR}/dataset"
TRACKING_OUTPUT_DIR="${EXPERIMENT_DIR}/threeD_tracking_offline"
OBJECT_MASKS_DIR="${EXPERIMENT_DIR}/object_masks"

# Skip flags for debugging
SKIP_STEPS=()

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
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

check_directory() {
    if [[ ! -d "$1" ]]; then
        print_error "Directory not found: $1"
        exit 1
    fi
}

create_directory() {
    if [[ ! -d "$1" ]]; then
        mkdir -p "$1"
        print_success "Created directory: $1"
    fi
}

update_yaml_value() {
    local file=$1
    local key=$2
    local value=$3
    
    if [[ -f "$file" ]]; then
        # Use sed to update YAML values (simple key: value replacement)
        if grep -q "^${key}:" "$file"; then
            sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
            print_info "Updated ${key} in $(basename $file)"
        else
            print_warning "Key '${key}' not found in $(basename $file)"
        fi
    else
        print_error "Config file not found: $file"
        exit 1
    fi
}

###############################################################################
# Parse Arguments
###############################################################################

while [[ $# -gt 3 ]]; do
    case $1 in
        --skip-step)
            SKIP_STEPS+=("$2")
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

###############################################################################
# Pipeline Setup
###############################################################################

print_header "Phantom-Touch Pipeline Configuration"
echo "Experiment Name: ${EXPERIMENT_NAME}"
echo "Data Directory: ${DATA_DIR}"
echo "Model Directory: ${MODEL_DIR}"
echo "Repository Directory: ${REPO_DIR}"
echo "Episode Range: ${START_EPISODE} to ${END_EPISODE}"
echo ""

# Verify prerequisites
print_info "Checking prerequisites..."

# Check if we're in the right directory
if [[ ! -f "${REPO_DIR}/setup.py" ]]; then
    print_error "Not in phantom-touch repository root"
    exit 1
fi

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    print_warning "Virtual environment not activated. Activate with: source .phantom-touch/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for SIEVE_API_KEY
if [[ -z "${SIEVE_API_KEY:-}" ]]; then
    print_warning "SIEVE_API_KEY not set. This is required for SAM2 hand segmentation."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if experiment directory exists
check_directory "${EXPERIMENT_DIR}"

print_success "Prerequisites check complete"
echo ""

###############################################################################
# Step 0: Update Main Paths Configuration
###############################################################################

if ! should_skip_step 0; then
    print_header "Step 0: Updating Main Configuration (cfg/paths.yaml)"
    
    PATHS_CONFIG="${REPO_DIR}/cfg/paths.yaml"
    
    # Backup original config
    cp "${PATHS_CONFIG}" "${PATHS_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Update main paths
    update_yaml_value "${PATHS_CONFIG}" "  experiment_name" "${EXPERIMENT_NAME}"
    update_yaml_value "${PATHS_CONFIG}" "  data_dir" "${DATA_DIR}"
    
    print_success "Main configuration updated"
    echo ""
else
    print_warning "Skipping Step 0"
fi

###############################################################################
# Step 1: VitPose Hand Segmentation
###############################################################################

if ! should_skip_step 1; then
    print_header "Step 1: VitPose Hand Segmentation"
    
    VITPOSE_CONFIG="${REPO_DIR}/src/segment_hands/cfg/vitpose_segmentation.yaml"
    
    # Backup and update config
    cp "${VITPOSE_CONFIG}" "${VITPOSE_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    update_yaml_value "${VITPOSE_CONFIG}" "experiment_name" "${EXPERIMENT_NAME}"
    update_yaml_value "${VITPOSE_CONFIG}" "img_folder" "${EXPERIMENT_DIR}"
    update_yaml_value "${VITPOSE_CONFIG}" "vitpose_out_folder" "${VITPOSE_OUTPUT_DIR}"
    
    create_directory "${VITPOSE_OUTPUT_DIR}"
    
    print_info "Running VitPose..."
    cd "${REPO_DIR}/src/segment_hands/scripts"
    python run_vitpose.py
    
    print_success "VitPose hand segmentation complete"
    echo ""
else
    print_warning "Skipping Step 1: VitPose"
fi

###############################################################################
# Step 2: Split Episodes
###############################################################################

if ! should_skip_step 2; then
    print_header "Step 2: Splitting Episodes"
    
    PREPROCESSOR_CONFIG="${REPO_DIR}/src/phantom_touch/cfg/preprocessors.yaml"
    
    # Backup config
    cp "${PREPROCESSOR_CONFIG}" "${PREPROCESSOR_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    print_info "Running episode splitter..."
    cd "${REPO_DIR}/src/phantom_touch/preprocessors"
    python split_episodes.py
    
    print_success "Episodes split successfully"
    echo ""
else
    print_warning "Skipping Step 2: Split Episodes"
fi

###############################################################################
# Step 3: SAM2 Hand Mask Generation
###############################################################################

if ! should_skip_step 3; then
    print_header "Step 3: SAM2 Hand Mask Generation"
    
    SAM2_CONFIG="${REPO_DIR}/src/sam2/cfg/sam2_object_by_text.yaml"
    
    # Backup and update config
    cp "${SAM2_CONFIG}" "${SAM2_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Update nested values (this is more complex for nested YAML)
    sed -i "s|  experiment_name:.*|  experiment_name: ${EXPERIMENT_NAME}|" "${SAM2_CONFIG}"
    sed -i "s|  data_dir:.*|  data_dir: ${DATA_DIR}|" "${SAM2_CONFIG}"
    sed -i "s|  text_prompt :.*|  text_prompt : ${SIEVE_TEXT_PROMPT}|" "${SAM2_CONFIG}"
    
    create_directory "${SAM2VID_OUTPUT_DIR}"
    
    print_info "Running SAM2 hand segmentation..."
    print_warning "This step requires SIEVE API and may take time per episode"
    cd "${REPO_DIR}/src/sam2/scripts"
    python segment_objVideo_byText.py
    
    print_success "SAM2 hand masks generated"
    echo ""
else
    print_warning "Skipping Step 3: SAM2 Hand Masks"
fi

###############################################################################
# Step 4: 3D Hand Projection
###############################################################################

if ! should_skip_step 4; then
    print_header "Step 4: 3D Hand Projection"
    
    PROJECTION_CONFIG="${REPO_DIR}/src/segment_hands/cfg/3d_projection.yaml"
    
    # Backup config
    cp "${PROJECTION_CONFIG}" "${PROJECTION_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    print_info "Projecting segmented hands to 3D..."
    cd "${REPO_DIR}/src/segment_hands/scripts"
    python project_sam2hand_to_3d.py
    
    print_success "3D hand projection complete"
    echo ""
else
    print_warning "Skipping Step 4: 3D Projection"
fi

###############################################################################
# Step 5: Inpainting (Hand Removal)
###############################################################################

if ! should_skip_step 5; then
    print_header "Step 5: Inpainting - Removing Hands from Images"
    
    INPAINT_CONFIG="${REPO_DIR}/src/inpainting/cfg/inpaint.yaml"
    
    # Backup and update config
    cp "${INPAINT_CONFIG}" "${INPAINT_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    sed -i "s|  rgb_base_path:.*|  rgb_base_path: ${EPISODES_DIR}|" "${INPAINT_CONFIG}"
    sed -i "s|  mask_base_path:.*|  mask_base_path: ${EPISODES_DIR}|" "${INPAINT_CONFIG}"
    sed -i "s|  output_path:.*|  output_path: ${INPAINTING_OUTPUT_DIR}|" "${INPAINT_CONFIG}"
    sed -i "s|  episode_start:.*|  episode_start: ${START_EPISODE}|" "${INPAINT_CONFIG}"
    sed -i "s|  episode_end:.*|  episode_end: $((END_EPISODE + 1))|" "${INPAINT_CONFIG}"
    
    create_directory "${INPAINTING_OUTPUT_DIR}"
    
    print_info "Running inpainting model..."
    cd "${REPO_DIR}/src/inpainting/scripts"
    python inpaint.py "${INPAINT_CONFIG}"
    
    print_success "Inpainting complete"
    echo ""
else
    print_warning "Skipping Step 5: Inpainting"
fi

###############################################################################
# Step 6: Phantom Data Creation
###############################################################################

if ! should_skip_step 6; then
    print_header "Step 6: Phantom Data Creation"

    Phantom_CONFIG="${REPO_DIR}/src/phantom_touch/cfg/phantom.yaml"

    # Backup and update config
    cp "${Phantom_CONFIG}" "${Phantom_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"

    # update episode range, the variable is episode_number and then it has two children start and end
    sed -i "s|  start:.*|  start: ${START_EPISODE}|" "${Phantom_CONFIG}"
    sed -i "s|  end:.*|  end: ${END_EPISODE}|" "${Phantom_CONFIG}"
    
    create_directory "${DATASET_DIR}"
    
    print_info "Creating phantom touch dataset..."
    cd "${REPO_DIR}/src/phantom_touch/scripts"
    python phantom_process_data.py
    
    print_success "Phantom dataset created"
    echo ""
else
    print_warning "Skipping Step 6: Phantom Data Creation"
fi

###############################################################################
# Step 7: 3D Object Tracking (Offline)
###############################################################################

if ! should_skip_step 7; then
    print_header "Step 7: 3D Object Tracking (Offline ICP)"
    
    TRACKING_CONFIG="${REPO_DIR}/src/render_contact_depth_patches/threeDoffline_object_tracking/cfg/threeD_tracking_offline.yaml"
    
    # Backup and update config
    cp "${TRACKING_CONFIG}" "${TRACKING_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    
    update_yaml_value "${TRACKING_CONFIG}" "CAD_MODEL_PATH" "${CAD_MODEL_PATH}"
    update_yaml_value "${TRACKING_CONFIG}" "cad_scale" "${CAD_SCALE}"
    update_yaml_value "${TRACKING_CONFIG}" "CAMERA_TO_ROBOT_TRANSFORM_PATH" "${CAMERA_TRANSFORM_PATH}"
    update_yaml_value "${TRACKING_CONFIG}" "start_episode" "${START_EPISODE}"
    update_yaml_value "${TRACKING_CONFIG}" "end_episode" "${END_EPISODE}"
    
    # Update sieve language prompt for object segmentation
    sed -i "s|  sieve_language_prompt:.*|  sieve_language_prompt: \"${OBJECT_TEXT_PROMPT}\"|" "${TRACKING_CONFIG}"
    
    create_directory "${TRACKING_OUTPUT_DIR}"
    create_directory "${OBJECT_MASKS_DIR}"
    
    print_info "Running 3D object tracking with ICP..."
    print_warning "This may take significant time for point cloud processing"
    cd "${REPO_DIR}/src/render_contact_depth_patches/threeDoffline_object_tracking"
    python threeD_tracking_offline.py
    
    print_success "3D object tracking complete"
    echo ""
else
    print_warning "Skipping Step 7: 3D Object Tracking"
fi

###############################################################################
# Step 8: Render Contact Depth Patches
###############################################################################

if ! should_skip_step 8; then
    print_header "Step 8: Rendering Contact Depth Patches"
    
    print_info "Rendering depth patches for contacts..."
    cd "${REPO_DIR}/src/render_contact_depth_patches"
    python render_depth_patches_phantom.py
    
    print_success "Contact depth patches rendered"
    echo ""
else
    print_warning "Skipping Step 8: Render Depth Patches"
fi

###############################################################################
# Pipeline Complete
###############################################################################

print_header "Pipeline Complete!"
echo ""
echo -e "${GREEN}✓ All steps completed successfully!${NC}"
echo ""
echo "Output locations:"
echo "  • VitPose outputs:     ${VITPOSE_OUTPUT_DIR}"
echo "  • SAM2 hand masks:     ${SAM2VID_OUTPUT_DIR}"
echo "  • Inpainted images:    ${INPAINTING_OUTPUT_DIR}"
echo "  • Phantom dataset:     ${DATASET_DIR}"
echo "  • 3D object tracking:  ${TRACKING_OUTPUT_DIR}"
echo "  • Object masks:        ${OBJECT_MASKS_DIR}"
echo ""
echo -e "${BLUE}Configuration backups saved with timestamp${NC}"
echo ""

# Offer to restore configs
read -p "Would you like to restore original config files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Restoring original configurations..."
    find "${REPO_DIR}" -name "*.yaml.backup.*" -type f | while read backup; do
        original="${backup%%.*}"
        if [[ -f "$original" ]]; then
            mv "$backup" "$original"
            print_info "Restored $(basename $original)"
        fi
    done
    print_success "Configurations restored"
fi

echo ""
print_success "Phantom-Touch pipeline finished!"
