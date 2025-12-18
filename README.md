# Phantom Touch Setup Guide
Convert human demos to robot data adapted from https://phantom-human-videos.github.io/.

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

## Collecting Data from Orbbec RGB-D Camera

To collect RGB, depth, raw depth, and point cloud aligned data from the Orbbec RGB-D camera, follow these steps.  
*Note: Change the number of iterations to the desired frame number (TODO: set in config).*

*Note: Use high speed usb-c to usb-c short cable for the camera*


### Method 1: Build and Run Orbbec (currently private)

1.    Clone the Orbbec repository and build the `OBSaveToDisk` executable:

        ```bash
        git clone git@github.com:ayadabdalla/Orbbec.git
        cd Orbbec
        mkdir build
        cd build
        cmake ..
        cmake --build . --target OBSaveToDisk
        ```
2. Create a directory for your experiment, then run the executable:

    ```bash
    mkdir <experiment_path>
    cd <experiment_path>
    ./<relative path to build/bin>/OBSaveToDisk
    ```
3.    Wait for frame initialization then press `r`
### Method 2: Replacing saveToDisk.cpp using OrbbecSDK

1. Clone the OrbbecSDK Repository

    ```bash

    git clone https://github.com/orbbec/OrbbecSDK.git
    ```
2. Locate and Replace File

    Find the `saveToDisk.cpp` file in `src/cameras/Orbbec` of your current project.

    Replace the `saveToDisk.cpp` file in the `examples` directory of OrbbecSDK with the one from your project.

3. Build the Executable

    Follow the same build instructions as in Method 1:

    ```bash
    cd OrbbecSDK
    mkdir build
    cd build
    cmake ..
    cmake --build . --target OBSaveToDisk
    ```
4. Create a directory for your experiment, then run the executable:

    ```bash
    mkdir <experiment_path>
    cd <experiment_path>
    ./<relative path to build/bin>/OBSaveToDisk
    ```
5. Wait for frame initialization then press `r`


# Running phantom-touch pipeline (slimming ongoing)
To collect data, one has to first run the orbbec c++ script as described in the collection step using method 1.

First adapt the config file in the following location to include your addresses and experiment name:
```
cfg/paths.yaml
```
Next, to get the vitpose hands run the following:
```
cd src/segment_hands/scripts
python segment_hands.py
```
after adapting the config files in: 
```
src/segment_hands/cfg/vitpose_segmentation.yaml
```

Next, to split the episodes and move the recordings into individual episodes folders accordingly, go to the preprocessor in phantom-touch and run split_episodes, after adapting the config files:

```
src/phantom-touch/cfg/preprocessors.yaml
```

```
cd src/phantom-touch/preprocessors
python split_episodes.py
```
Next, we need to get the masks for the intended hand, to do so, we run sam2, after adapting the config:
Before doing that, one has to get a frame with an obvious free hand at the beginning of each episode.
```
src/sam2/cfg/sam2_object_by_text.yaml
```
```
cd src/sam2/scripts
python segment_objVideo_byText.py
```

Next, we need to project the segmented hands into 3D, to do so, we run and adapt:
```
src/phantom-touch/cfg/3d_projection.yaml
```
```
python src/phantom-touch/scripts/project_sam2hand_to_3d.py
```
Next, we need to run the inpainting model to remove the intended hand, to do so, adapt and run:

```
src/inpainting/cfg/inpaint.yaml
```

```
python src/inpainting/scripts/inpaint.py
```

#TODO: describe the installation steps to run the inpainting in details from https://github.com/MCG-NKU/E2FGVI.git.


Next, to run the phantom data creation, run:
```
cd src/phantom-touch/scripts/
python phantom_process_data.py
```

Since 3d object tracking data are not readily available in the recording, we run and adapt the following to track the object offline:

```
src/threeDoffline_tracking/cfg/threeD_tracking_offline.yaml
```
```
python src/threeDoffline_tracking/three_tracking_offline.py
```
In order to add touch data, we need first to render depth patches for the contacts with the object. To do so:

```
cd render_contact_depth_patches
python render_depth_patches_phantom.py
```
To be continued...