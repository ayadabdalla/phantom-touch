#!/bin/bash

###############################################################################
# Phantom-Touch Environment Validation Script
# 
# Checks all prerequisites before running the pipeline
###############################################################################

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

print_check() {
    echo -e "${BLUE}[CHECK]${NC} $1"
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((ERRORS++))
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

echo "=================================="
echo "Phantom-Touch Environment Validation"
echo "=================================="
echo ""

# Check 1: Repository structure
print_check "Checking repository structure..."
if [[ -f "setup.py" ]] && [[ -d "src" ]] && [[ -d "cfg" ]]; then
    print_pass "Repository structure is correct"
else
    print_fail "Not in phantom-touch repository root"
fi

# Check 2: Virtual environment
print_check "Checking virtual environment..."
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    print_pass "Virtual environment activated: ${VIRTUAL_ENV}"
else
    print_warn "Virtual environment not activated (run: source .phantom-touch/bin/activate)"
fi

# Check 3: Python version
print_check "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" == 3.12* ]]; then
    print_pass "Python version: ${PYTHON_VERSION}"
else
    print_warn "Python version ${PYTHON_VERSION} (expected 3.12.x)"
fi

# Check 4: Required Python packages
print_check "Checking required Python packages..."
PACKAGES=("torch" "numpy" "cv2" "omegaconf" "trimesh" "open3d" "sieve")
for pkg in "${PACKAGES[@]}"; do
    if python -c "import ${pkg}" 2>/dev/null; then
        print_pass "Package '${pkg}' installed"
    else
        print_fail "Package '${pkg}' not found"
    fi
done

# Check 5: SAM3 installation
print_check "Checking SAM3..."
if python -c "from sam3.build_sam import build_sam3_video_predictor" 2>/dev/null; then
    print_pass "SAM3 installed and importable"
else
    print_fail "SAM3 not properly installed"
fi

# Check 7: Config files exist
print_check "Checking configuration files..."
CONFIGS=(
    "cfg/paths.yaml"
    "src/segment_hands/cfg/vitpose_segmentation.yaml"
    "src/phantom_touch/cfg/preprocessors.yaml"
    "src/sam3_session/cfg/sam3_object_by_text.yaml"
    "src/segment_hands/cfg/3d_projection.yaml"
    "src/inpainting/cfg/inpaint.yaml"
    "src/render_contact_depth_patches/threeDoffline_object_tracking/cfg/threeD_tracking_offline.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    if [[ -f "$cfg" ]]; then
        print_pass "Config exists: $(basename $cfg)"
    else
        print_fail "Config missing: $cfg"
    fi
done

# Check 8: Required executables
print_check "Checking system utilities..."
UTILS=("sed" "awk" "grep")
for util in "${UTILS[@]}"; do
    if command -v "$util" &> /dev/null; then
        print_pass "Utility '${util}' available"
    else
        print_fail "Utility '${util}' not found"
    fi
done

# Check 9: Model checkpoints
print_check "Checking model checkpoints..."
if [[ -d "src/hamer/_DATA" ]]; then
    print_pass "Hamer models directory exists"
else
    print_warn "Hamer models directory not found (src/hamer/_DATA)"
fi

# Check 10: Disk space
print_check "Checking disk space..."
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [[ $AVAILABLE_GB -gt 50 ]]; then
    print_pass "Sufficient disk space: ${AVAILABLE_GB}GB available"
else
    print_warn "Low disk space: ${AVAILABLE_GB}GB available (recommend 50GB+)"
fi

# Check 11: GPU availability
print_check "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [[ $GPU_COUNT -gt 0 ]]; then
        print_pass "GPU available: ${GPU_COUNT} device(s)"
    else
        print_warn "No GPU detected (pipeline will be slow)"
    fi
else
    print_warn "nvidia-smi not found (GPU highly recommended)"
fi

# Check 12: Pipeline scripts executable
print_check "Checking pipeline scripts..."
if [[ -f "run_phantom_pipeline.sh" ]]; then
    if [[ -x "run_phantom_pipeline.sh" ]]; then
        print_pass "Pipeline script is executable"
    else
        print_warn "Pipeline script not executable (run: chmod +x run_phantom_pipeline.sh)"
    fi
else
    print_fail "Pipeline script not found: run_phantom_pipeline.sh"
fi

# Summary
echo ""
echo "=================================="
echo "Validation Summary"
echo "=================================="

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}✓ All checks passed! Ready to run pipeline.${NC}"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠ ${WARNINGS} warning(s) found. Pipeline may run with limitations.${NC}"
    exit 0
else
    echo -e "${RED}✗ ${ERRORS} error(s) and ${WARNINGS} warning(s) found.${NC}"
    echo -e "${RED}Please fix errors before running the pipeline.${NC}"
    exit 1
fi
