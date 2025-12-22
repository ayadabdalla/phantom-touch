# Phantom-Touch Pipeline Automation - Visual Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     PHANTOM-TOUCH AUTOMATED PIPELINE                         │
│                           Quick Start Guide                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│  STEP 1: VALIDATE    │  ./validate_environment.sh
│  ✓ Python 3.12       │
│  ✓ Required packages │
│  ✓ GPU available     │
│  ✓ Disk space        │
│  ✓ Config files      │
└──────────────────────┘
           ↓
┌──────────────────────┐
│  STEP 2: CONFIGURE   │  cp pipeline_config.env my_experiment.env
│  • Edit paths        │  nano my_experiment.env
│  • Set episode range │  source my_experiment.env
│  • Add API key       │
│  • Choose prompts    │
└──────────────────────┘
           ↓
┌──────────────────────┐
│  STEP 3: RUN         │  ./run_phantom_pipeline.sh \
│  Automated pipeline  │    "${EXPERIMENT_NAME}" \
│  8 steps sequential  │    "${DATA_DIR}" \
│  ~1-4 hours total    │    "${MODEL_DIR}"
└──────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                              PIPELINE FLOW
═══════════════════════════════════════════════════════════════════════════════

 INPUT: Raw Orbbec Recordings
    └─ Color_*.png, Depth_*.bin, PointCloud_*.ply
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 0: Configuration Update                                            │
│ └─ Auto-updates all YAML files with your experiment paths               │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 1: VitPose Hand Segmentation                                       │
│ └─ Detects hand keypoints in RGB images                                 │
│    Output: vitpose_output/episodes/                                     │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Episode Splitting                                               │
│ └─ Splits continuous recording into discrete episodes                   │
│    Output: episodes/e0/, episodes/e1/, ...                              │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 3: SAM2 Hand Masking (SIEVE API)                                   │
│ └─ Segments hands using language prompts                                │
│    Prompt: "human on the left side of the image"                        │
│    Output: sam2-vid_output/episodes/                                    │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 4: 3D Hand Projection                                              │
│ └─ Projects 2D hand masks to 3D point clouds using depth                │
│    Output: sam2hand_output/episodes/                                    │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Inpainting (Hand Removal)                                       │
│ └─ Removes hands from RGB images using E2FGVI                           │
│    Output: inpainting_output/episodes/                                  │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Phantom Data Creation                                           │
│ └─ Creates robot-compatible dataset from processed data                 │
│    Output: dataset/e0/experiment_e0.npz                                 │
│    └─ Contains: RGB, depth, hand poses, trajectories                    │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 7: 3D Object Tracking (ICP)                                        │
│ └─ Tracks object pose over time using point cloud alignment             │
│    Uses: CAD model + depth observations                                 │
│    Output: threeD_tracking_offline/episode_*.npz                        │
│    └─ Contains: T_robot_from_cad, positions, rotations                  │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ STEP 8: Depth Patch Rendering                                           │
│ └─ Renders contact depth patches for touch sensing                      │
│    Output: Contact depth information for tactile data                   │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ↓
    FINAL OUTPUT: Complete robot training dataset with touch awareness

═══════════════════════════════════════════════════════════════════════════════
                          UTILITY COMMANDS
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│ ./pipeline_utils.sh validate                                            │
│ └─ Run environment validation checks                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ ./pipeline_utils.sh check-outputs /path/to/experiment                   │
│ └─ Verify all output directories and count files                        │
├─────────────────────────────────────────────────────────────────────────┤
│ ./pipeline_utils.sh list-episodes /path/to/experiment                   │
│ └─ List all episodes with file counts and sizes                         │
├─────────────────────────────────────────────────────────────────────────┤
│ ./pipeline_utils.sh count-frames /path/to/experiment                    │
│ └─ Count color and depth frames per episode                             │
├─────────────────────────────────────────────────────────────────────────┤
│ ./pipeline_utils.sh verify-cad /path/to/model.obj                       │
│ └─ Verify CAD model exists and check basic statistics                   │
├─────────────────────────────────────────────────────────────────────────┤
│ ./pipeline_utils.sh disk-usage /path/to/experiment                      │
│ └─ Show disk usage breakdown by directory                               │
├─────────────────────────────────────────────────────────────────────────┤
│ ./pipeline_utils.sh clean-backups                                       │
│ └─ Remove all timestamped config backup files                           │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                        CONFIGURATION REFERENCE
═══════════════════════════════════════════════════════════════════════════════

Required Environment Variables (set in .env file):

┌────────────────────────┬──────────────────────────────────────────────────┐
│ Variable               │ Description                                      │
├────────────────────────┼──────────────────────────────────────────────────┤
│ EXPERIMENT_NAME        │ Name of your experiment                          │
│ DATA_DIR               │ Base directory for experiment data               │
│ MODEL_DIR              │ Directory containing model checkpoints           │
│ START_EPISODE          │ First episode to process (0-indexed)             │
│ END_EPISODE            │ Last episode to process (inclusive)              │
│ CAD_MODEL_PATH         │ Full path to object CAD model (.obj)             │
│ CAD_SCALE              │ Scale factor (usually 0.001 for mm→m)            │
│ SIEVE_HAND_PROMPT      │ Text prompt for hand segmentation                │
│ OBJECT_PROMPT          │ Text prompt for object segmentation              │
│ SIEVE_API_KEY          │ API key from sievedata.com                       │
│ CAMERA_TRANSFORM_PATH  │ Path to camera-to-robot transform (.npy)         │
└────────────────────────┴──────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                          SKIP FLAGS REFERENCE
═══════════════════════════════════════════════════════════════════════════════

Use --skip-step <N> to skip individual steps:

./run_phantom_pipeline.sh experiment /data --skip-step 1 --skip-step 3

┌──────┬────────────────────────────┬─────────────────────────────────────┐
│ Step │ Name                       │ When to Skip                        │
├──────┼────────────────────────────┼─────────────────────────────────────┤
│  1   │ VitPose                    │ Already have hand keypoints         │
│  2   │ Episode Splitting          │ Episodes already split              │
│  3   │ SAM2 Hand Masking          │ Have pre-generated masks            │
│  4   │ 3D Hand Projection         │ Already projected to 3D             │
│  5   │ Inpainting                 │ Already have inpainted images       │
│  6   │ Phantom Data Creation      │ Dataset already created             │
│  7   │ 3D Object Tracking         │ Have object tracking data           │
│  8   │ Depth Patch Rendering      │ Patches already rendered            │
└──────┴────────────────────────────┴─────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                       COMMON ISSUES & SOLUTIONS
═══════════════════════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────┐
│ Issue: SIEVE API rate limiting                                             │
│ Solution: Run step 3 separately with delays between episodes               │
│          OR use pre-generated masks and --skip-step 3                      │
├────────────────────────────────────────────────────────────────────────────┤
│ Issue: GPU out of memory                                                   │
│ Solution: Process fewer episodes at once, reduce batch sizes in configs    │
├────────────────────────────────────────────────────────────────────────────┤
│ Issue: ICP alignment failures                                              │
│ Solution: Check CAD_SCALE, verify camera transform, adjust ICP params      │
├────────────────────────────────────────────────────────────────────────────┤
│ Issue: Missing dependencies                                                │
│ Solution: Run ./validate_environment.sh to see what's missing              │
├────────────────────────────────────────────────────────────────────────────┤
│ Issue: Config changes not taking effect                                    │
│ Solution: Check for .backup files, restore originals, re-run pipeline      │
└────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                          FILES CREATED
═══════════════════════════════════════════════════════════════════════════════

Scripts (3):
  • run_phantom_pipeline.sh      (456 lines) - Main automation
  • validate_environment.sh       (184 lines) - Environment checker
  • pipeline_utils.sh             (296 lines) - Utility commands

Configuration (2):
  • pipeline_config.env           - Configuration template
  • example_config.env            - Working example

Documentation (3):
  • PIPELINE_GUIDE.md             (200 lines) - Quick reference
  • AUTOMATION_SUMMARY.md         (263 lines) - Detailed docs
  • VISUAL_GUIDE.md               (this file) - Visual guide

Total: 1,399+ lines of automation code + comprehensive documentation

═══════════════════════════════════════════════════════════════════════════════
                         SUCCESS CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

Before running pipeline:
 □ Virtual environment activated
 □ SIEVE_API_KEY exported
 □ Configuration file created and sourced
 □ CAD model exists and verified
 □ Camera transform available
 □ Sufficient disk space (50GB+ recommended)
 □ GPU available (strongly recommended)
 □ Environment validation passed

After pipeline completion:
 □ All 8 steps completed without errors
 □ Output directories exist and populated
 □ Dataset NPZ files created
 □ Object tracking results generated
 □ Config backups created
 □ Original configs restored (optional)

═══════════════════════════════════════════════════════════════════════════════

For detailed information, see:
  • README.md               - Setup and overview
  • PIPELINE_GUIDE.md       - Quick reference and troubleshooting
  • AUTOMATION_SUMMARY.md   - Complete automation documentation

For help:
  ./run_phantom_pipeline.sh --help
  ./pipeline_utils.sh help
  ./validate_environment.sh --help

═══════════════════════════════════════════════════════════════════════════════
```
