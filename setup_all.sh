#!/usr/bin/env bash

## This script creates Python virtualenv and installs libraries for analysis.
## No deep learning framework will be installed.
##
## The first argument is recognised as the name of virtualenv.

if [[ -z "$1" ]]; then
  VENV="venv"
else
  VENV=$1
fi

echo "-- Create a Python environment"
python -m venv $VENV

echo "-- Activate the Python environment: $VENV"
source $VENV/bin/activate

echo "-- Upgrade pip"
pip install --upgrade pip

echo "-- Install adhoc and required libraries"
pip install https://github.com/stdiff/adhoc/archive/v0.4.zip

echo "-- Install jupyter lab and jupytext"
pip install jupyter==1.0.0
pip install jupyterlab==1.2.6
pip install jupytext==0.8.6

echo "-- Install jupyterlab_spellchecker"
jupyter labextension install @ijmbarr/jupyterlab_spellchecker

echo "-- Install jupyterlab_templates"
pip install jupyterlab_templates
jupyter labextension install jupyterlab_templates
jupyter serverextension enable --py jupyterlab_templates

echo "-- Install further libraries"
pip install watermark==2.0.2
pip install scikit-image==0.16.2 pillow==6.2.2

echo "-- create requirements.txt"
pip freeze > requirements.txt

echo "-- Deactivate the Python environment"
deactivate

echo "-- Create some directories"
mkdir data
mkdir lib
mkdir notebook
mkdir test
mkdir tmp

