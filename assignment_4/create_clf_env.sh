#!/usr/bin/env bash

VENVNAME=clf_env

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

pip install ipython
pip install jupyter
pip install seaborn
pip install scikit_learn
pip install opencv-python
pip install sklearn
pip install joblib

python -m ipykernel install --user --name=$VENVNAME

test -f requirements.txt && pip install -r requirements.txt

deactivate
echo "build $VENVNAME"
