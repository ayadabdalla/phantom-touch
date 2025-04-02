python -m venv .phantom-touch

source .phantom-touch/bin/activate

pip install -r requirements/pre-requirements.txt

pip install cython

pip install --no-build-isolation xtcocotools

pip install --no-build-isolation -r requirements/requirements.txt 
