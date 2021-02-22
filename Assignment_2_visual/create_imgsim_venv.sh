#!/usr/bin/env bash

VENVNAME=img_sim

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install matplotlib
pip install opencv-python
pip install re
pip install pandas


python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
