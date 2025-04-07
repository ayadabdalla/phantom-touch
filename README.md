python -m venv .phantom-touch

source .phantom-touch/bin/activate

pip install -r requirements/pre-requirements.txt

pip install cython

pip install --no-build-isolation xtcocotools

pip install --no-build-isolation -r requirements/requirements.txt 

add vitpose_model.py to your hamer directory in <virtualenv>/lib/python<version>/site-packages

pip install sam2 and ignore iopath conflict

add hamer,vitpose and mano checkpoints and model to hamer directory, not the one in the venv/

edit some function in the ch module (screenshot)

change the int in loading frames in sam2 in venv, see screenshot

#Create a free account on SIEVE

export SIEVE_API_KEY=<"your key">