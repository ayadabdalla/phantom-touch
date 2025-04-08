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

6. Install `phantom-touch` in editable mode:
    ```bash
    pip install -e .
    ```
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
      **Line**: 246
      **Change** Add `.png` extension for flexebility: [".jpg", ".jpeg", ".JPG", ".JPEG",".png", ".PNG"]
      **Line**: 248  
      **Change**: Remove `int()` typecasting.

3. Add `hamer`, `vitpose`, and `mano` checkpoints and models to the `hamer` directory in `phantom-touch` (not inside the virtual environment):
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
1. Create a free account on [SIEVE](https://www.sievedata.com/)
2. Export your API key:
    ```bash
    export SIEVE_API_KEY="<your key>"
    ```
## Running the Project
In order to run both sam2 and hamer workflows, all you need is `sam2sieve.images_path` in the config file in `sam2/conf` directory and a `sam2sieve.text_prompt`. In order to run hamer, all you need is `img_folder` in the config file in `hamer/conf` directory.
Note: Take note of the paths of final and intermediary outputs, and make sure to have the models checkpoints and config for sam2.
### Run SAM2
1. Adapt the config file input and output paths in `sam2` directory if needed.
2. In `phantom-touch`, go to the `sam2` directory:
    ```bash
    cd src/sam2
3. Adapt the `config` in `conf` to your paths and run:
    ```bash
    python scripts/segment_objVideo_by_Text.py
    ```
### Run Hamer
1. In `phantom-touch`, go to the `hamer` directory:
    ```bash
    cd src/hamer
    ```
2. Adapt the `config` in `conf` to your paths and run:
    ```bash
    python scripts/segment_hands.py
    ```
