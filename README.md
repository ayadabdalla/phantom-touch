# Phantom Touch Setup Guide

## Clone the Repository
```bash
git clone git@github.com:ayadabdalla/phantom-touch.git
cd phantom-touch
```

## Set Up Virtual Environment
```bash
python3.12 -m venv .phantom-touch
source .phantom-touch/bin/activate
```

## Install Dependencies
1. Install pre-requirements:
    ```bash
    pip install -r requirements/pre-requirements.txt
    ```
    *(Ensure compatibility with `cu12.6` and its corresponding `torch` and `torchvision` versions.)*

2. Install cython
   ```bash
    pip install cython
    ```
3. Install xtcocotools
    ```bash
    pip install xtcocotools
    ```
4. Install main requirements:
    ```bash
    pip install --no-build-isolation -r requirements/requirements.txt
    ```

5. Install `sam2`:
    ```bash
    pip install sam2
    ```
    *Note: Ignore any `iopath` conflict.*

## Additional Setup
1. Download `vitpose_model.py`:
    ```bash
    wget https://raw.githubusercontent.com/geopavlakos/hamer/refs/heads/main/vitpose_model.py -P .phantom-touch/lib/python3.12/site-packages/hamer
    ```

2. Modify the following files:
    - **File**: `.phantom-touch/lib/python3.12/site-packages/chumpy/ch.py`  
      **Function**: `depends_on`  
      **Line**: 1203  
      **Change**: Replace `getargs` with `getfullargspec`.

    - **File**: `.phantom-touch/lib/python3.12/site-packages/sam2/utils/misc.py`  
      **Function**: `load_video_frames_from_jpg_images`  
      **Line**: 248  
      **Change**: Remove `int()` typecasting.

3. Install the package in editable mode:
    ```bash
    pip install -e .
    ```

4. Add `hamer`, `vitpose`, and `mano` checkpoints and models to the `hamer` directory (not inside the virtual environment):
    ```bash
    cd src/hamer
    cp -r /mnt/dataset_drive/ayad/phantom-touch/models/hamer/_DATA/ ./
    ```

## Install Third-Party Dependencies
1. Create a `third-party` directory in `hamer` directory:
    ```bash
    mkdir third-party
    cd third-party
    ```

2. Clone the `ViTPose` repository:
    ```bash
    git clone git@github.com:ViTAE-Transformer/ViTPose.git
    ```

## SIEVE API Setup
1. Create a free account on [SIEVE](https://www.sieve.com).
2. Export your API key:
    ```bash
    export SIEVE_API_KEY="<your key>"
    ```
## Running the Project
In order to run both sam2 and hamer workflows, all you need is a path of a list of png images in the `sam2sieve.images_path` in the config file in `sam2` directory and a text prompt. First run sam2 workflow, which creates an rgb video as a side product, then extract the jpg frames from it using the script `utils/extract_frames.py` and changing the `video_path` and `output_folder` to the appropriate paths, and run hamer on the `jpg_frames`.
### Run SAM2
1. Adapt the config file input and output paths in `sam2` directory if needed.
1. Use the `segment_objVideo_by_Text` script available in `sam2/scripts`:
    ```bash
    python phantom-touch/src/sam2/scripts/segment_objVideo_by_Text.py
    ```
### Run Hamer
1. In `phantom-touch`, go to the `hamer` directory:
    ```bash
    cd src/hamer
    ```
2. Adapt the following command to your paths and run:
    ```bash
    python scripts/segment_hands.py \
        --img_folder /mnt/dataset_drive/ayad/phantom-touch/data/recordings/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/png_output/color_sample/ \
        --out_folder /mnt/dataset_drive/ayad/phantom-touch/data/output/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/hamer_output/ \
        --checkpoint ~/phantom-touch/src/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt
    ```