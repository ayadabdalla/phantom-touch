# Phantom-Touch Pipeline Quick Reference

## Prerequisites Checklist

- [ ] Orbbec data collected using `OBSaveToDisk`
- [ ] Virtual environment activated: `source .phantom-touch/bin/activate`
- [ ] SIEVE API key exported: `export SIEVE_API_KEY="your_key"`
- [ ] Experiment name decided
- [ ] CAD model available for object tracking
- [ ] Camera-to-robot transform calibrated

## Quick Start

### 1. Validate Environment
```bash
./validate_environment.sh
```

### 2. Configure Pipeline
```bash
# Copy template
cp pipeline_config.env my_experiment.env

# Edit configuration
nano my_experiment.env

# Source configuration
source my_experiment.env
```

### 3. Run Pipeline
```bash
./run_phantom_pipeline.sh "${EXPERIMENT_NAME}" "${DATA_DIR}" "${MODEL_DIR}"
```

## Common Configuration Values

### Experiment Names
- `experimental_phantomn_touch_collection`
- `pick_and_place_phantom`
- `soft_objects`
- `white_cloth_exp`

### Data Directories
- Local: `/home/epon04yc`
- Dataset drive: `/mnt/dataset_drive/ayad/data`

### Text Prompts
**Hand segmentation:**
- `"human on the left side of the image"`
- `"person's hand"` 
- `"left hand"`

**Object segmentation:**
- `"strawberry fruit"`
- `"red cup"`
- `"white cloth"`
- `"grape cluster"`

### CAD Models
Location: `/mnt/dataset_drive/ayad/scenes_and_cad/cad_models/`
- `Strawberry.obj`
- `Cup.obj`
- `Grape.obj`

### Common Scale Factors
- Most objects: `0.001` (mm to m)
- Pre-scaled models: `1.0`

## Troubleshooting

### Skip Problematic Steps
```bash
# Skip VitPose (already done)
./run_phantom_pipeline.sh experiment /path/to/data --skip-step 1

# Skip SAM2 (using pre-generated masks)
./run_phantom_pipeline.sh experiment /path/to/data --skip-step 3

# Skip multiple steps
./run_phantom_pipeline.sh experiment /path/to/data --skip-step 1 --skip-step 3 --skip-step 5
```

### Manual Step Execution

If you need to run steps individually:

**Step 1: VitPose**
```bash
cd src/segment_hands/scripts
python run_vitpose.py
```

**Step 2: Split Episodes**
```bash
cd src/phantom_touch/preprocessors
python split_episodes.py
```

**Step 3: SAM2 Hand Masks**
```bash
cd src/sam2/scripts
python segment_objVideo_byText.py
```

**Step 4: 3D Projection**
```bash
cd src/segment_hands/scripts
python project_sam2hand_to_3d.py
```

**Step 5: Inpainting**
```bash
cd src/inpainting/scripts
python inpaint.py ../cfg/inpaint.yaml
```

**Step 6: Phantom Data**
```bash
cd src/phantom_touch/scripts
python phantom_process_data.py
```

**Step 7: Object Tracking**
```bash
cd src/render_contact_depth_patches/threeDoffline_object_tracking
python threeD_tracking_offline.py
```

**Step 8: Depth Patches**
```bash
cd src/render_contact_depth_patches
python render_depth_patches_phantom.py
```

## Output Directory Structure

After pipeline completion:
```
${EXPERIMENT_NAME}/
├── episodes/               # Raw recordings split by episode
│   ├── e0/
│   ├── e1/
│   └── ...
├── vitpose_output/         # Hand keypoints
│   └── episodes/
├── sam2-vid_output/        # Hand segmentation masks
│   └── episodes/
├── inpainting_output/      # Images with hands removed
│   └── episodes/
├── dataset/                # Final phantom dataset
│   ├── e0/
│   │   └── *.npz
│   └── ...
├── threeD_tracking_offline/ # Object pose tracking results
│   └── episode_*.npz
└── object_masks/           # Object segmentation masks
```

## Common Issues

### SIEVE API Rate Limiting
- Solution: Run SAM2 step separately with delays
- Alternative: Use pre-generated masks and skip step 3

### Out of Memory (GPU)
- Reduce batch sizes in configs
- Process fewer episodes at once
- Use CPU fallback (slower)

### ICP Alignment Failures
- Check CAD model scale factor
- Verify camera-to-robot transform
- Adjust ICP parameters in config

### Missing Camera Transform
- Generate using calibration script
- Default location: `src/cameras/Orbbec/data/robotbase_camera_transform_orbbec_fr4.npy`

## Configuration Files Reference

| Step | Config File | Key Parameters |
|------|-------------|----------------|
| Main | `cfg/paths.yaml` | `experiment_name`, `data_dir` |
| 1 | `src/segment_hands/cfg/vitpose_segmentation.yaml` | `img_folder`, `crop` |
| 2 | `src/phantom_touch/cfg/preprocessors.yaml` | `split_threshold` |
| 3 | `src/sam2/cfg/sam2_object_by_text.yaml` | `text_prompt` |
| 4 | `src/segment_hands/cfg/3d_projection.yaml` | `shape` |
| 5 | `src/inpainting/cfg/inpaint.yaml` | `episode_start`, `episode_end` |
| 7 | `threeDoffline_object_tracking/cfg/threeD_tracking_offline.yaml` | `CAD_MODEL_PATH`, `cad_scale` |

## Tips

1. **Always validate environment first**: `./validate_environment.sh`
2. **Use config backups**: Pipeline saves timestamped backups
3. **Monitor disk space**: Each episode can be several GB
4. **Test with 1-2 episodes first**: Use smaller episode ranges initially
5. **Keep SIEVE API key secure**: Don't commit to git
6. **Check logs**: Each step prints progress and errors
7. **GPU recommended**: CPU processing is 10-100x slower
