#!/bin/bash

###############################################################################
# Phantom-Touch Pipeline Utilities
# 
# Helper script for common pipeline operations
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

show_help() {
    cat << EOF
Phantom-Touch Pipeline Utilities

Usage: ./pipeline_utils.sh [command] [options]

Commands:
  validate              Validate environment and dependencies
  clean-backups         Remove all config backup files
  restore-configs       Restore original config files from latest backups
  check-outputs         Check if output directories exist and report status
  disk-usage            Show disk usage for experiment directories
  list-episodes         List all episodes in an experiment
  count-frames          Count frames in each episode
  verify-cad            Verify CAD model exists and is valid
  help                  Show this help message

Examples:
  ./pipeline_utils.sh validate
  ./pipeline_utils.sh clean-backups
  ./pipeline_utils.sh check-outputs /home/epon04yc/pick_and_place_phantom
  ./pipeline_utils.sh list-episodes /home/epon04yc/pick_and_place_phantom
  ./pipeline_utils.sh verify-cad /path/to/model.obj

EOF
}

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

cmd_validate() {
    if [[ -f "./validate_environment.sh" ]]; then
        ./validate_environment.sh
    else
        print_error "validate_environment.sh not found"
        exit 1
    fi
}

cmd_clean_backups() {
    print_header "Cleaning Config Backups"
    
    BACKUP_COUNT=$(find . -name "*.yaml.backup.*" -type f | wc -l)
    
    if [[ $BACKUP_COUNT -eq 0 ]]; then
        print_info "No backup files found"
        return
    fi
    
    echo "Found $BACKUP_COUNT backup file(s)"
    read -p "Delete all backups? (y/n) " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        find . -name "*.yaml.backup.*" -type f -delete
        print_success "Deleted $BACKUP_COUNT backup file(s)"
    else
        print_info "Cancelled"
    fi
}

cmd_restore_configs() {
    print_header "Restoring Config Files"
    
    # Find latest backups
    find . -name "*.yaml.backup.*" -type f | while read backup; do
        original="${backup%%.*}"
        original="${original}.yaml"
        
        if [[ -f "$original" ]]; then
            cp "$backup" "$original"
            print_success "Restored: $(basename $original)"
        else
            print_error "Original not found for: $(basename $backup)"
        fi
    done
    
    print_info "Config restoration complete"
}

cmd_check_outputs() {
    EXPERIMENT_DIR="${1:-.}"
    
    if [[ ! -d "$EXPERIMENT_DIR" ]]; then
        print_error "Directory not found: $EXPERIMENT_DIR"
        exit 1
    fi
    
    print_header "Checking Output Directories"
    echo "Experiment: $EXPERIMENT_DIR"
    echo ""
    
    DIRS=(
        "episodes:Raw Episodes"
        "vitpose_output/episodes:VitPose Keypoints"
        "sam2-vid_output/episodes:SAM2 Hand Masks"
        "sam2hand_output/episodes:3D Hand Projections"
        "inpainting_output/episodes:Inpainted Images"
        "dataset:Phantom Dataset"
        "threeD_tracking_offline:Object Tracking"
        "object_masks:Object Masks"
    )
    
    for entry in "${DIRS[@]}"; do
        IFS=':' read -r dir label <<< "$entry"
        full_path="${EXPERIMENT_DIR}/${dir}"
        
        if [[ -d "$full_path" ]]; then
            count=$(find "$full_path" -type f 2>/dev/null | wc -l)
            print_success "$label: $count files"
        else
            print_error "$label: Not found"
        fi
    done
}

cmd_disk_usage() {
    EXPERIMENT_DIR="${1:-.}"
    
    if [[ ! -d "$EXPERIMENT_DIR" ]]; then
        print_error "Directory not found: $EXPERIMENT_DIR"
        exit 1
    fi
    
    print_header "Disk Usage Analysis"
    echo "Experiment: $EXPERIMENT_DIR"
    echo ""
    
    du -h --max-depth=1 "$EXPERIMENT_DIR" | sort -hr
}

cmd_list_episodes() {
    EXPERIMENT_DIR="${1:-.}"
    EPISODES_DIR="${EXPERIMENT_DIR}/episodes"
    
    if [[ ! -d "$EPISODES_DIR" ]]; then
        print_error "Episodes directory not found: $EPISODES_DIR"
        exit 1
    fi
    
    print_header "Episodes List"
    echo "Location: $EPISODES_DIR"
    echo ""
    
    for episode in "$EPISODES_DIR"/e*; do
        if [[ -d "$episode" ]]; then
            episode_name=$(basename "$episode")
            file_count=$(find "$episode" -type f | wc -l)
            size=$(du -sh "$episode" | cut -f1)
            print_info "Episode $episode_name: $file_count files, $size"
        fi
    done
}

cmd_count_frames() {
    EXPERIMENT_DIR="${1:-.}"
    EPISODES_DIR="${EXPERIMENT_DIR}/episodes"
    
    if [[ ! -d "$EPISODES_DIR" ]]; then
        print_error "Episodes directory not found: $EPISODES_DIR"
        exit 1
    fi
    
    print_header "Frame Count per Episode"
    echo ""
    
    for episode in "$EPISODES_DIR"/e*; do
        if [[ -d "$episode" ]]; then
            episode_name=$(basename "$episode")
            
            # Count Color frames
            color_count=$(find "$episode" -name "Color_*.png" -o -name "Color_*.jpg" | wc -l)
            
            # Count Depth frames
            depth_count=$(find "$episode" -name "RawDepth_*.raw" -o -name "Depth_*.bin" | wc -l)
            
            echo -e "${BLUE}$episode_name${NC}"
            echo "  Color frames: $color_count"
            echo "  Depth frames: $depth_count"
            echo ""
        fi
    done
}

cmd_verify_cad() {
    CAD_PATH="${1}"
    
    if [[ -z "$CAD_PATH" ]]; then
        print_error "Usage: ./pipeline_utils.sh verify-cad <path/to/model.obj>"
        exit 1
    fi
    
    if [[ ! -f "$CAD_PATH" ]]; then
        print_error "CAD model not found: $CAD_PATH"
        exit 1
    fi
    
    print_header "CAD Model Verification"
    echo "File: $CAD_PATH"
    echo ""
    
    # Check file size
    size=$(du -h "$CAD_PATH" | cut -f1)
    print_info "File size: $size"
    
    # Check file format
    extension="${CAD_PATH##*.}"
    print_info "Format: .$extension"
    
    # Try to get basic stats
    if [[ "$extension" == "obj" ]]; then
        vertices=$(grep -c "^v " "$CAD_PATH" 2>/dev/null || echo "0")
        faces=$(grep -c "^f " "$CAD_PATH" 2>/dev/null || echo "0")
        
        print_info "Vertices: $vertices"
        print_info "Faces: $faces"
        
        if [[ $vertices -gt 0 ]] && [[ $faces -gt 0 ]]; then
            print_success "CAD model appears valid"
        else
            print_error "CAD model may be invalid or empty"
        fi
    else
        print_info "Non-OBJ format, skipping detailed check"
    fi
}

# Main command dispatcher
COMMAND="${1:-help}"

case "$COMMAND" in
    validate)
        cmd_validate
        ;;
    clean-backups)
        cmd_clean_backups
        ;;
    restore-configs)
        cmd_restore_configs
        ;;
    check-outputs)
        cmd_check_outputs "${2:-.}"
        ;;
    disk-usage)
        cmd_disk_usage "${2:-.}"
        ;;
    list-episodes)
        cmd_list_episodes "${2:-.}"
        ;;
    count-frames)
        cmd_count_frames "${2:-.}"
        ;;
    verify-cad)
        cmd_verify_cad "${2:-}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        echo ""
        show_help
        exit 1
        ;;
esac
