git clone git@github.com:ayadabdalla/phantom-touch.git

cd phantom-touch

python -m venv .phantom-touch

source .phantom-touch/bin/activate

pip install -r requirements/pre-requirements.txt
(cu12.6 with its torch and torchvision counterparts)

pip install --no-build-isolation -r requirements/requirements.txt 

pip install sam2
note: ignore iopath conflict

wget https://github.com/geopavlakos/hamer/blob/main/vitpose_model.py -P .phantom-touch/lib/python3.12/site-packages/hamer



In .phantom-touch/lib/python3.12/site-packages/chumpy/ch.py In depends_on In line 1203 in _depends_on, change getargs to getfullargspec

In .phantom-touch/lib/python3.12/site-packages/sam2/utils/misc.py In load_video_frames_from_jpg_images In line 248 remove int() typecasting

pip install -e .

add hamer,vitpose and mano checkpoints and model to hamer directory, not the one in the venv/
cd src/hamer
cp -r /mnt/dataset_drive/ayad/phantom-touch/hamer/_DATA/ ./

#Create a free account on SIEVE
export SIEVE_API_KEY=<"your key">

mkdir third-party
cd third-party
git clone git@github.com:ViTAE-Transformer/ViTPose.git

To run hamer
cd ..
adapt the following to your paths
python scripts/segment_hands.py --img_folder /mnt/dataset_drive/ayad/phantom-touch/data/recordings/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/png_output/color_sample/ --out_folder /mnt/dataset_drive/ayad/phantom-touch/data/output/white_cloth_exp/white_nonreflective_cloth_light_on_ambient_light/hamer_output/ --checkpoint ~/phantom-touch/src/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt 

To run sam2
run segment_objVideo_by_Text available in phantom-touch/src/sam2/scripts